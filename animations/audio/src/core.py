"""Audio engine: sample loading, pitch mapping, sampler instruments, DSP helpers, buses, reverb."""
import os, re, glob, json, math, random
from fractions import Fraction
from functools import lru_cache
import numpy as np, soundfile as sf
from scipy import signal

SR = 48000
ROOT = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(ROOT, 'libs')
VSCO = os.path.join(LIB, 'VSCO-2-CE')
VCSL = os.path.join(LIB, 'VCSL')
SP = os.path.join(ROOT, 'sp')
RNG = np.random.default_rng(7)

def rnd(a=0.0, b=1.0):
    return float(RNG.uniform(a, b))

# ---------------------------------------------------------------- loading
_cache = {}
def load(path, mono=False):
    key = (path, mono)
    if key in _cache:
        return _cache[key]
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.T
    if x.shape[0] == 1:
        x = np.vstack([x, x])
    elif x.shape[0] > 2:
        x = x[:2]
    if sr != SR:
        f = Fraction(SR, sr).limit_denominator(1000)
        x = signal.resample_poly(x, f.numerator, f.denominator, axis=1).astype(np.float32)
    # trim leading silence (keeps attacks tight)
    a = np.abs(x).max(0)
    pk = a.max() + 1e-9
    idx = np.argmax(a > pk * 0.003)
    x = x[:, max(0, idx - 48):]
    x = x / pk * 0.9
    if mono:
        x = x.mean(0, keepdims=True).repeat(2, 0)
    _cache[key] = x
    return x

NOTE_RE = re.compile(r'(?<![A-Za-z])([A-G])(#|b)?(-?\d)(?![0-9])')
NAMES = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
def note_from_name(fn):
    base = os.path.basename(fn)
    ms = list(NOTE_RE.finditer(base))
    if not ms:
        return None
    m = ms[0]
    n = NAMES[m.group(1)] + (1 if m.group(2) == '#' else -1 if m.group(2) == 'b' else 0)
    return n + (int(m.group(3)) + 1) * 12

def midi(name):
    """'D3' -> 50 ; accepts ints"""
    if isinstance(name, (int, np.integer)):
        return int(name)
    m = re.match(r'([A-G])(#|b)?(-?\d)', name)
    n = NAMES[m.group(1)] + (1 if m.group(2) == '#' else -1 if m.group(2) == 'b' else 0)
    return n + (int(m.group(3)) + 1) * 12

def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)

def detect_f0(x):
    y = x.mean(0)
    n = len(y)
    seg = y[int(n * 0.15): int(n * 0.15) + 16384]
    if len(seg) < 4096:
        seg = y[:16384]
    seg = seg * np.hanning(len(seg))
    # harmonic product spectrum
    N = 1 << 17
    sp = np.abs(np.fft.rfft(seg, N))
    hps = sp.copy()
    for h in (2, 3, 4):
        d = sp[::h]
        hps[:len(d)] *= d
    f = np.fft.rfftfreq(N, 1 / SR)
    lo, hi = np.searchsorted(f, 25), np.searchsorted(f, 2500)
    k = lo + np.argmax(hps[lo:hi])
    return f[k]

# ---------------------------------------------------------------- pitch shift (resample)
def shift(x, semis):
    if abs(semis) < 1e-3:
        return x
    r = 2 ** (semis / 12.0)
    fr = Fraction(r).limit_denominator(160)
    return signal.resample_poly(x, fr.denominator, fr.numerator, axis=1).astype(np.float32)

def speed(x, r):
    fr = Fraction(r).limit_denominator(200)
    return signal.resample_poly(x, fr.denominator, fr.numerator, axis=1).astype(np.float32)

# ---------------------------------------------------------------- envelopes / filters
def fade(x, fin=0.005, fout=0.05):
    x = x.copy()
    n = x.shape[1]
    a, b = int(fin * SR), int(fout * SR)
    if a > 0:
        a = min(a, n); x[:, :a] *= np.linspace(0, 1, a, dtype=np.float32)
    if b > 0:
        b = min(b, n); x[:, n - b:] *= np.linspace(1, 0, b, dtype=np.float32) ** 1.5
    return x

def sos(kind, f, order=2):
    if kind == 'bp':
        return signal.butter(order, [max(f[0], 10) / (SR / 2), min(f[1], SR / 2 - 100) / (SR / 2)], 'bandpass', output='sos')
    return signal.butter(order, min(max(f, 10), SR / 2 - 100) / (SR / 2), kind, output='sos')

def lp(x, f, order=2): return signal.sosfilt(sos('low', f, order), x, axis=-1).astype(np.float32)
def hp(x, f, order=2): return signal.sosfilt(sos('high', f, order), x, axis=-1).astype(np.float32)
def bp(x, lo, hi, order=2): return signal.sosfilt(sos('bp', (lo, hi), order), x, axis=-1).astype(np.float32)

def peak(x, f, gain_db, q=1.0):
    """peaking EQ (RBJ biquad)"""
    A = 10 ** (gain_db / 40); w = 2 * np.pi * f / SR; al = np.sin(w) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w), 1 - al * A]; a = [1 + al / A, -2 * np.cos(w), 1 - al / A]
    return signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=-1).astype(np.float32)

def sweep_filter(x, f0, f1, kind='low', blocks=64, q=0.7):
    """time-varying filter (block-wise state-variable approximation) for whooshes"""
    n = x.shape[1]; out = np.zeros_like(x); bl = int(np.ceil(n / blocks))
    zi = None
    for i in range(blocks):
        a, b = i * bl, min(n, (i + 1) * bl)
        if a >= b: break
        f = f0 * (f1 / f0) ** (i / max(1, blocks - 1))
        s = sos(kind if kind != 'bp' else 'bp', f if kind != 'bp' else (f / 1.6, f * 1.6), 2)
        if zi is None or zi.shape[0] != s.shape[0]:
            zi = np.zeros((s.shape[0], 2, 2))
        y, zi2 = signal.sosfilt(s, x[:, a:b], axis=-1, zi=np.transpose(zi, (0, 1, 2)))
        out[:, a:b] = y; zi = zi2
    return out.astype(np.float32)

def env_ar(n, a, r, curve=2.0):
    e = np.ones(n, np.float32)
    A = min(n, int(a * SR)); R = min(n - A, int(r * SR))
    if A: e[:A] = np.linspace(0, 1, A) ** curve
    if R: e[n - R:] = np.linspace(1, 0, R) ** curve
    return e

def noise(sec, color='white', seed=None):
    n = int(sec * SR); rg = np.random.default_rng(seed if seed is not None else RNG.integers(1 << 30))
    w = rg.standard_normal((2, n)).astype(np.float32)
    if color == 'pink':
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]; a = [1, -2.494956002, 2.017265875, -0.522189400]
        w = signal.lfilter(b, a, w, axis=1).astype(np.float32) * 6
    elif color == 'brown':
        w = signal.lfilter([1], [1, -0.995], w, axis=1).astype(np.float32) * 0.08
    return w

def stereoize(m, width=0.3, seed=None):
    """mono (n,) -> decorrelated stereo"""
    if m.ndim == 2: return m
    d = int(SR * 0.011)
    L = m; R = np.concatenate([np.zeros(d, np.float32), m[:-d]]) * width + m * (1 - width)
    return np.vstack([L, R]).astype(np.float32)

def pan(x, p):
    """p in [-1,1] equal power"""
    a = (p + 1) * np.pi / 4
    return np.vstack([x[0] * np.cos(a) * 1.414, x[1] * np.sin(a) * 1.414]).astype(np.float32)

def db(v): return 10 ** (v / 20)

# ---------------------------------------------------------------- buses
class Bus:
    def __init__(self, dur):
        self.n = int(dur * SR) + SR
        self.x = np.zeros((2, self.n), np.float32)
    def add(self, sig, t, gain=1.0, p=0.0):
        if sig is None or sig.shape[1] == 0: return
        if p: sig = pan(sig, p)
        i = int(round(t * SR))
        if i >= self.n: return
        j = 0
        if i < 0: j = -i; i = 0
        m = min(sig.shape[1] - j, self.n - i)
        if m <= 0: return
        self.x[:, i:i + m] += sig[:, j:j + m] * gain

# ---------------------------------------------------------------- impulse responses
def make_ir(rt60=2.8, pre=0.02, damp=6000, er=12, width=1.0, seed=3, bright=0.0):
    rg = np.random.default_rng(seed)
    n = int((rt60 * 1.15 + pre) * SR)
    t = np.arange(n) / SR
    ir = np.zeros((2, n), np.float32)
    # late tail: decorrelated noise, frequency dependent decay (3 bands)
    bands = [(20, 400, 1.15), (400, 3000, 1.0), (3000, 16000, 0.55 + bright)]
    for lo, hi, k in bands:
        nz = rg.standard_normal((2, n)).astype(np.float32)
        nz = bp(nz, lo, hi, 2)
        dec = np.exp(-6.91 * t / (rt60 * k)).astype(np.float32)
        ir += nz * dec
    ir[:, :int(pre * SR)] = 0
    on = int(pre * SR); ramp = int(0.03 * SR)
    ir[:, on:on + ramp] *= np.linspace(0, 1, ramp)
    # early reflections
    for i in range(er):
        d = pre * 0.4 + rg.uniform(0.003, 0.09)
        g = rg.uniform(0.2, 0.6) * (1 - d * 5)
        k = int(d * SR)
        if k < n:
            ir[0, k] += g * rg.choice([-1, 1]); ir[1, min(n - 1, k + rg.integers(0, 60))] += g * rg.choice([-1, 1])
    ir = lp(ir, damp, 1)
    if width < 1:
        m = ir.mean(0); ir = np.vstack([m + (ir[0] - m) * width, m + (ir[1] - m) * width])
    ir /= np.sqrt((ir ** 2).sum() / 2)
    return ir.astype(np.float32)

def convolve(x, ir):
    out = np.vstack([signal.oaconvolve(x[c], ir[c])[:x.shape[1]] for c in range(2)]).astype(np.float32)
    return out

# ---------------------------------------------------------------- sampler instruments
VEL_RE = re.compile(r'(?:_v|_vl|dyn)(\d)')
class Sampler:
    """maps sample files to MIDI notes; plays notes with pitch-shift, looped sustain, release"""
    def __init__(self, pattern, name='', octave_fix=None, vel_layers=None, release=0.35, loop=True, gain=1.0, exclude=None, detect=False):
        files = sorted(glob.glob(pattern))
        if exclude: files = [f for f in files if not re.search(exclude, f)]
        self.name = name; self.release = release; self.loop = loop; self.gain = gain
        self.map = {}  # note -> list of (vel, path)
        for f in files:
            n = note_from_name(f)
            if n is None: continue
            vm = VEL_RE.search(os.path.basename(f))
            v = int(vm.group(1)) if vm else 1
            self.map.setdefault(n, []).append((v, f))
        if not self.map:
            raise RuntimeError('no samples for ' + pattern)
        # octave correction using pitch detection on a few samples
        if octave_fix is None:
            offs = []
            for n in list(self.map)[:: max(1, len(self.map) // 5)][:5]:
                x = load(self.map[n][0][1])
                f0 = detect_f0(x)
                dm = 69 + 12 * math.log2(f0 / 440)
                offs.append(round((dm - n) / 12) * 12)
            octave_fix = int(np.median(offs))
        self.fix = octave_fix
        self.map = {n + self.fix: v for n, v in self.map.items()}
        self.notes = np.array(sorted(self.map))
        self.vels = sorted({v for l in self.map.values() for v, _ in l})
        self._pcache = {}

    def pick(self, note, vel):
        i = np.argmin(np.abs(self.notes - note)); src = int(self.notes[i])
        layers = self.map[src]
        vs = sorted({v for v, _ in layers})
        # vel 0..1 -> layer
        target = vs[min(len(vs) - 1, int(vel * len(vs) - 1e-6))]
        cands = [p for v, p in layers if v == target]
        return src, cands[int(RNG.integers(len(cands)))]

    def render(self, note, dur, vel=0.7, attack=0.0, release=None):
        note = midi(note)
        src, path = self.pick(note, vel)
        key = (path, note - src)
        if key not in self._pcache:
            self._pcache[key] = shift(load(path), note - src)
        x = self._pcache[key]
        rel = self.release if release is None else release
        need = int((dur + rel) * SR)
        if need <= x.shape[1] or not self.loop:
            y = x[:, :need].copy()
        else:
            # crossfade looping of the steady part
            n = x.shape[1]; a0, a1 = int(n * 0.3), int(n * 0.85)
            xf = int(0.35 * SR)
            if a1 - a0 < xf * 2: a0, a1 = int(n * 0.1), n - 10
            y = np.zeros((2, need), np.float32)
            y[:, :a1] = x[:, :a1][:, :need] if a1 <= need else x[:, :need]
            pos = a1
            w = np.linspace(0, 1, xf, dtype=np.float32)
            while pos < need:
                s = int(RNG.integers(a0, max(a0 + 1, a1 - xf * 2)))
                seg = x[:, s:a1]
                m = min(seg.shape[1], need - pos + xf)
                st = pos - xf
                y[:, st:st + xf] = y[:, st:st + xf] * np.sqrt(1 - w[:min(xf, need - st)]) + seg[:, :min(xf, need - st)] * np.sqrt(w[:min(xf, need - st)])
                if m > xf:
                    e = min(st + m, need)
                    y[:, st + xf:e] = seg[:, xf:xf + (e - st - xf)]
                pos = st + m
        e = env_ar(y.shape[1], attack, rel, 1.5)
        return (y * e * self.gain).astype(np.float32)

def S(*parts):
    return os.path.join(*parts)
