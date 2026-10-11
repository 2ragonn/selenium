"""Procedural SFX track for 'What If Humans Could Fly?' (43.26 s, 48 kHz stereo).
Every sound is synthesized from scratch and placed on the animation's exact cut/event times
(read from timeline.json, dumped from the HTML)."""
import json, sys, wave
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

SR = 48000
DUR = 43.26
N = int(round(DUR * SR))
rng = np.random.default_rng(20261011)
TL = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'timeline.json'))
OUT = sys.argv[2] if len(sys.argv) > 2 else 'sfx.wav'
S = {s['name']: s['t0'] for s in TL['shots']}
SHOTS = TL['shots']

def W(n, k=1):
    c = 0
    for w in TL['words']:
        if w['n'] == n:
            c += 1
            if c == k:
                return w['t0']
    raise KeyError(n)

bus = np.zeros((2, N))
rv = np.zeros((2, N))

# ---------------------------------------------------------------- basics
def T(d): return np.arange(int(d * SR)) / SR
def noise(n): return rng.standard_normal(n)
def clamp(v, a=0.0, b=1.0): return min(max(v, a), b)
def lerp(a, b, t): return a + (b - a) * t
def prog(t, a, b): return clamp((t - a) / (b - a))
def ioC(t): return 4 * t ** 3 if t < .5 else 1 - (-2 * t + 2) ** 3 / 2

def pink(n):
    X = np.fft.rfft(rng.standard_normal(n)); f = np.fft.rfftfreq(n); f[0] = f[1]
    y = np.fft.irfft(X / np.sqrt(f), n); return y / (np.std(y) + 1e-12)

_sos = {}
def _get(kind, f, o):
    key = (kind, f, o)
    if key not in _sos:
        wn = np.array(f) / (SR / 2) if isinstance(f, tuple) else f / (SR / 2)
        _sos[key] = butter(o, wn, btype=kind, output='sos')
    return _sos[key]
def lp(x, f, o=2): return sosfilt(_get('low', float(min(f, SR * 0.45)), o), x)
def hp(x, f, o=2): return sosfilt(_get('high', float(f), o), x)
def bp(x, lo, hi, o=2): return sosfilt(_get('band', (float(lo), float(min(hi, SR * 0.45))), o), x)

def sweep_bp(x, fc, q=1.2, blk=256):
    y = np.zeros_like(x); zi = np.zeros((1, 2))
    for i in range(0, len(x), blk):
        c = float(np.clip(fc[i], 40, SR * 0.4)); bw = c / q
        lo = max(25.0, c - bw / 2); hi = min(SR * 0.45, c + bw / 2)
        s = butter(1, [lo / (SR / 2), hi / (SR / 2)], btype='band', output='sos')
        y[i:i + blk], zi = sosfilt(s, x[i:i + blk], zi=zi)
    return y

def sweep_lp(x, fc, blk=256):
    y = np.zeros_like(x); zi = np.zeros((1, 2))
    for i in range(0, len(x), blk):
        c = float(np.clip(fc[i], 40, SR * 0.4))
        s = butter(2, c / (SR / 2), btype='low', output='sos')
        y[i:i + blk], zi = sosfilt(s, x[i:i + blk], zi=zi)
    return y

def slow_noise(n, rate, smooth=True):
    k = max(4, int(n / SR * rate) + 4)
    pts = rng.uniform(0, 1, k); xs = np.linspace(0, n, k)
    y = np.interp(np.arange(n), xs, pts)
    return lp(y, rate * 1.5, 1) if smooth else y

def fade(y, a=0.002, r=0.004):
    na, nr = min(len(y), int(a * SR)), min(len(y), int(r * SR))
    if na > 0: y[:na] *= np.linspace(0, 1, na)
    if nr > 0: y[-nr:] *= np.linspace(1, 0, nr)
    return y

def norm(y, p=1.0):
    m = np.max(np.abs(y)) + 1e-12; return y * (p / m)

def db(x): return 10 ** (x / 20)

def place(sig, t, g=1.0, pan=0.0, rev=0.0):
    sig = fade(np.array(sig, dtype=float))
    n = len(sig); i = int(round(t * SR))
    p = np.full(n, float(pan)) if np.isscalar(pan) else np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(pan)), np.asarray(pan, dtype=float))
    th = (np.clip(p, -1, 1) + 1) * np.pi / 4
    st = np.vstack([sig * np.cos(th), sig * np.sin(th)]) * g
    a, b = max(0, i), min(N, i + n)
    if b <= a: return
    bus[:, a:b] += st[:, a - i:b - i]
    if rev > 0: rv[:, a:b] += st[:, a - i:b - i] * rev

def pan_sweep(n, p0, p1): return np.linspace(p0, p1, n)

# ---------------------------------------------------------------- generators
def impact(size=1.0, f0=95, f1=36, dec=0.9, bright=1.0):
    t = T(dec * 1.6); n = len(t)
    f = f1 + (f0 - f1) * np.exp(-t / 0.06)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (dec * 0.45))
    body = lp(noise(n), 900) * np.exp(-t / 0.07) * 0.9
    click = hp(noise(n), 2500) * np.exp(-t / 0.006) * 0.55 * bright
    mid = bp(noise(n), 160, 650) * np.exp(-t / 0.2) * 0.55
    return np.tanh((sub + body + click + mid) * 1.4 * size) / 1.1

def whoosh(dur=0.5, f0=300, f1=2600, peak=0.62, q=1.3, air=0.35):
    n = int(dur * SR); u = np.linspace(0, 1, n)
    fc = f0 * (f1 / f0) ** u
    y = sweep_bp(noise(n), fc, q)
    y = norm(y) + air * norm(lp(noise(n), 500))
    e = np.where(u < peak, (u / peak) ** 2.2, ((1 - u) / (1 - peak)) ** 1.7)
    return norm(y * e)

def passby(dur=0.6, flo=250, fhi=2200, q=1.0):
    n = int(dur * SR); u = np.linspace(0, 1, n)
    e = np.exp(-((u - 0.5) / 0.2) ** 2)
    fc = flo + (fhi - flo) * e * np.where(u < 0.5, 1.0, 0.75)
    y = sweep_bp(noise(n), fc, q) * e
    return norm(y)

def riser(dur=1.0, f0=200, f1=3500, tone=True):
    n = int(dur * SR); u = np.linspace(0, 1, n)
    fc = f0 * (f1 / f0) ** (u ** 1.3)
    y = norm(sweep_bp(noise(n), fc, 1.6))
    if tone:
        f = 110 * (8 ** (u ** 1.5))
        ph = 2 * np.pi * np.cumsum(f) / SR
        saw = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.33 * np.sin(3 * ph)) * (1 + 0.3 * np.sin(2 * np.pi * (6 + 18 * u) * u))
        y = y + 0.35 * norm(saw)
    return norm(y * u ** 2.2)

def ping(f, dec=0.25):
    t = T(dec * 5)
    y = (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2.01 * f * t) * np.exp(-t / (dec * 0.35))) * np.exp(-t / dec)
    return fade(y, 0.0015, 0.01)

HI = [1174.66, 1318.51, 1479.98, 1760.0, 1975.53, 2349.32, 2637.02, 2959.96, 3520.0, 3951.07]
MID = [587.33, 659.25, 739.99, 880.0, 987.77, 1174.66, 1318.51, 1479.98]

def sparkle(t0, count, spread, g=0.12, notes=HI, dec=0.16, rev=0.45, pan_range=0.9):
    for _ in range(count):
        tt = t0 + spread * rng.uniform(0, 1) ** 1.3
        place(ping(rng.choice(notes) * (1 + rng.uniform(-.004, .004)), dec * rng.uniform(0.6, 1.5)), tt,
              g * rng.uniform(0.35, 1.0), rng.uniform(-pan_range, pan_range), rev)

def pop(f0=900, f1=260, dec=0.06):
    t = T(dec * 5); f = f1 + (f0 - f1) * np.exp(-t / 0.012)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / dec)
    y += hp(noise(len(t)), 3000) * np.exp(-t / 0.003) * 0.3
    return y

def bell(f, dec=1.2, bright=1.0):
    t = T(dec * 3.5); y = np.zeros(len(t))
    for r, a, dk in [(1, 1, 1), (2.0, .45, .55), (2.76, .35 * bright, .4), (4.07, .22 * bright, .28), (5.4, .14 * bright, .18)]:
        y += a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / (dec * dk))
    return fade(y, 0.001, 0.02)

def horn(dur=0.35, f1=415, f2=523):
    t = T(dur); vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t)
    y = np.tanh(4 * np.sin(2 * np.pi * f1 * vib * t)) + 0.8 * np.tanh(4 * np.sin(2 * np.pi * f2 * vib * t))
    y = bp(y, 300, 2800)
    e = np.minimum(1, t / 0.015) * np.minimum(1, (dur - t) / 0.04)
    return norm(y * e)

def thunder(size=1.0, length=2.6):
    t = T(length); n = len(t)
    crack = hp(noise(n), 1200) * np.exp(-t / 0.035) * 1.2 + bp(noise(n), 300, 3500) * np.exp(-t / 0.22) * 0.7
    am = slow_noise(n, 9) ** 2 * np.exp(-t / (0.9 * size)) * np.minimum(1, t / 0.08)
    rumble = lp(lp(noise(n), 160), 160) * am * 6.0
    sub = np.sin(2 * np.pi * np.cumsum(38 + 20 * np.exp(-t / 0.3)) / SR) * np.exp(-t / 0.7) * 0.7
    return np.tanh((crack + norm(rumble) * 1.1 + sub) * size)

def heartbeat():
    t = T(0.55); y = np.zeros(len(t))
    for off, a in [(0, 1.0), (0.19, 0.65)]:
        tt = np.clip(t - off, 0, None); m = t >= off
        f = 42 + 30 * np.exp(-tt / 0.03)
        y += m * a * np.sin(2 * np.pi * np.cumsum(f * m) / SR) * np.exp(-tt / 0.09)
    return y + lp(noise(len(t)), 200) * np.exp(-t / 0.05) * 0.3

def tick(f_lo=2500, f_hi=7000, dec=0.004):
    t = T(0.03); return bp(noise(len(t)), f_lo, f_hi) * np.exp(-t / dec)

def blip(f, dur=0.08, shape=3.0):
    t = T(dur); y = np.tanh(shape * np.sin(2 * np.pi * f * t))
    return y * np.exp(-t / (dur * 0.35))

def glitch(dur=0.2):
    n = int(dur * SR); y = np.zeros(n); i = 0
    while i < n:
        L = int(rng.uniform(0.008, 0.03) * SR); seg = np.arange(min(L, n - i)) / SR; k = rng.uniform()
        if k < 0.4: s = np.sign(np.sin(2 * np.pi * rng.uniform(150, 2500) * seg))
        elif k < 0.75:
            hold = int(rng.integers(4, 40)); s = np.repeat(rng.uniform(-1, 1, len(seg) // hold + 1), hold)[:len(seg)]
        else: s = np.zeros(len(seg))
        y[i:i + len(seg)] = s * rng.uniform(0.4, 1); i += len(seg)
    return lp(y, 9000) * 0.8

def laser(dur=0.32, f0=300, f1=2600):
    t = T(dur); u = t / dur
    f = f0 * (f1 / f0) ** (u ** 0.6) * (1 + 0.02 * np.sin(2 * np.pi * 30 * t))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.3 * np.sin(4 * np.pi * np.cumsum(f) / SR)
    return y * np.minimum(1, u / 0.05) * (1 - u) ** 1.2

def power_down(dur=0.7, f0=1100, f1=160):
    t = T(dur); u = t / dur; f = f1 + (f0 - f1) * (1 - u) ** 2
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)
    return lp(y, 3000) * np.minimum(1, u / 0.03) * (1 - u) ** 0.8

def clang(f=310, dec=0.7):
    t = T(dec * 3); y = np.zeros(len(t))
    for r, a in [(1, 1), (2.32, .7), (3.86, .5), (5.1, .4), (6.7, .3), (8.9, .2)]:
        y += a * np.sin(2 * np.pi * f * r * t + rng.uniform(0, 6)) * np.exp(-t / (dec / (1 + 0.25 * r)))
    return y * 0.6

def flap():
    t = T(0.18); e = np.exp(-t / 0.018) + 0.7 * np.exp(-np.clip(t - 0.06, 0, None) / 0.02) * (t > 0.06)
    return bp(noise(len(t)), 700, 4500) * e

def footstep(soft=1.0):
    t = T(0.12)
    return (lp(noise(len(t)), 350) * np.exp(-t / 0.035) * 1.2 + bp(noise(len(t)), 1500, 6000) * np.exp(-t / 0.012) * 0.4 * soft)

def boing(f=260, dur=0.45):
    t = T(dur); f_ = f * (1 + 0.45 * np.sin(2 * np.pi * 11 * t) * np.exp(-t / 0.12)) * (1 + 0.6 * np.exp(-t / 0.03))
    return np.sin(2 * np.pi * np.cumsum(f_) / SR) * np.exp(-t / 0.16)

def rip(dur=0.35):
    n = int(dur * SR); t = np.arange(n) / SR
    am = (rng.uniform(0, 1, n // 300 + 1).repeat(300)[:n] > 0.35).astype(float)
    y = sweep_bp(noise(n), 900 * 4 ** (t / dur), 0.8) * lp(am, 400, 1)
    return norm(y) * np.minimum(1, t / 0.02) * (1 - t / dur) ** 0.6

def seagull():
    t = T(0.42); u = t / 0.42
    f = np.interp(u, [0, 0.25, 0.55, 1], [1350, 2100, 1650, 1250]) * (1 + 0.015 * np.sin(2 * np.pi * 28 * t))
    y = np.tanh(2.5 * np.sin(2 * np.pi * np.cumsum(f) / SR))
    return bp(y, 900, 5000) * np.sin(np.pi * u) ** 0.7

def rocket(dur=0.9):
    n = int(dur * SR); u = np.linspace(0, 1, n)
    y = sweep_lp(noise(n), 250 * 14 ** u) * (np.minimum(1, u / 0.05)) * (1 - u) ** 0.5
    rumble = lp(noise(n), 120) * 3 * (1 - u)
    return norm(y + rumble)

def alarm(dur=0.5):
    t = T(dur); f = np.where((t * 8).astype(int) % 2 == 0, 880, 660)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.3 * np.sin(4 * np.pi * np.cumsum(f) / SR)
    return y * np.minimum(1, t / 0.01) * np.minimum(1, (dur - t) / 0.05)

def chord_pad(dur, notes, cutoff=1400, det=0.004):
    t = T(dur); y = np.zeros(len(t))
    for f in notes:
        for d in (-det, 0, det):
            ph = 2 * np.pi * f * (1 + d) * t + rng.uniform(0, 6)
            y += np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)
    y = lp(y, cutoff); u = t / dur
    return norm(y) * np.minimum(1, u / 0.25) * np.minimum(1, (1 - u) / 0.25)

# ---------------------------------------------------------------- beds
def bed(t0, t1, gen, g, pan=0.0, rev=0.0):
    n = int((t1 - t0) * SR)
    if n > 0: place(gen(n), t0, g, pan, rev)

def wind(n, lo=250, hi=900, gust_rate=1.2, depth=0.6):
    m = slow_noise(n, 0.7)
    y = sweep_bp(noise(n), lo + (hi - lo) * m, 1.6)
    g = (1 - depth) + depth * slow_noise(n, gust_rate) ** 1.5
    return norm(y * g)

def rain(n):
    y = bp(pink(n), 900, 9000) * 0.9
    drops = (rng.uniform(0, 1, n) > 0.9975).astype(float) * rng.uniform(0.3, 1, n)
    y += bp(drops, 2000, 8000) * 6
    return norm(y)

def ocean(n):
    t = np.arange(n) / SR; sw = 0.55 + 0.45 * np.sin(2 * np.pi * 0.55 * t - 1.2)
    low = lp(pink(n), 500) * sw
    hiss = bp(pink(n), 2500, 9000) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.55 * t - 1.9)) ** 2
    return norm(low + 0.6 * norm(hiss) * np.std(low))

def murmur(n, bands=8):
    y = np.zeros(n)
    for _ in range(bands):
        c = rng.uniform(350, 2600)
        y += bp(noise(n), c * 0.85, c * 1.15) * slow_noise(n, rng.uniform(3.5, 7), smooth=False) ** 2
    return norm(lp(y, 3500))

def hum(n, f=110, cutoff=900):
    t = np.arange(n) / SR; y = np.zeros(n)
    for k in range(1, 8): y += np.sin(2 * np.pi * f * k * t * (1 + 0.0015 * np.sin(2 * np.pi * 0.3 * t))) / k
    return norm(lp(y, cutoff))

def crackle(n):
    imp = (rng.uniform(0, 1, n) > 0.996) * rng.uniform(-1, 1, n)
    return norm(bp(imp, 1800, 9000) * 1.0 + bp(noise(n), 3000, 9000) * 0.02)

# =================================================================== CUES
def cut_fx(sh, nxt_t0):
    t, tr = sh['t0'], sh['tr']
    if tr == 'flash':
        place(impact(0.6, 120, 50, 0.45, 1.4), t, 0.35, 0, 0.35)
        sparkle(t, 6, 0.12, 0.1)
        place(hp(noise(int(0.25 * SR)), 4000) * np.exp(-T(0.25) / 0.06), t, 0.12, 0, 0.6)
    elif tr == 'zoom':
        place(whoosh(0.32, 400, 4200, 0.92, 1.4, 0.2), t - 0.29, 0.38, pan_sweep(int(0.32 * SR), -0.2, 0.2), 0.2)
        place(impact(0.7, 85, 40, 0.5, 0.6), t, 0.4, 0, 0.25)
    elif tr == 'slice':
        place(whoosh(0.3, 1200, 6000, 0.5, 1.1, 0.15), t - 0.15, 0.36, pan_sweep(int(0.3 * SR), 0.8, -0.8), 0.25)
        place(tick(1500, 6000, 0.01), t, 0.3, 0, 0.2)
    elif tr == 'whip':
        place(whoosh(0.28, 500, 3800, 0.55, 0.9, 0.5), t - 0.15, 0.48, pan_sweep(int(0.28 * SR), -0.9, 0.9), 0.15)
    elif tr == 'glitch':
        place(glitch(0.2), t, 0.22, rng.uniform(-0.3, 0.3), 0.1)
        place(impact(0.5, 140, 60, 0.25, 1.0), t, 0.25, 0, 0.2)

for i, sh in enumerate(SHOTS):
    if i: cut_fx(sh, SHOTS[i + 1]['t0'] if i + 1 < len(SHOTS) else DUR)

def hit(t, size=1.0, g=0.8, rev=0.35, f0=95, dec=0.9):
    place(impact(size, f0, 36, dec), t + 0.015, g, 0, rev)

# ---- 1 HOOK -------------------------------------------------------
tF = W('fly')
bed(0, S['sCurb'] + 0.1, lambda n: wind(n, 400, 1600, 2.5, 0.4), 0.16, 0, 0.1)
place(riser(1.0, 300, 3000, False), 0.0, 0.08, 0, 0.1)
hit(W('what'), 0.9, 0.75); sparkle(0.0, 5, 0.15, 0.08)
hit(W('if'), 0.6, 0.45, f0=120, dec=0.5)
hit(W('humans'), 1.0, 0.8)
hit(W('could'), 0.7, 0.5, f0=110, dec=0.6)
hit(tF, 1.3, 1.0, 0.6, 80, 1.4); sparkle(tF, 18, 0.45, 0.14); place(chord_pad(1.0, [293.66, 369.99, 440, 587.33], 2500), tF, 0.1, 0, 0.5)
for k in range(5): place(passby(0.45, 300, 2200), 0.05 + k * 0.32, 0.12, rng.uniform(-0.8, 0.8))
place(riser(0.3, 600, 6000, True), tF + 0.03, 0.3, 0, 0.2)
place(whoosh(0.55, 300, 5000, 0.45, 1.2, 0.3), tF + 0.1, 0.45, 0, 0.3)

# ---- 2 CURB -------------------------------------------------------
tY, tS, tO, tT, tCu = W('you'), W('step'), W('off'), W('the'), W('curb')
bed(S['sCurb'], 3.1, lambda n: norm(lp(pink(n), 700)) * 0.6 + 0.4 * murmur(n, 4), 0.05, 0, 0.1)
for ft in [tY + 0.04, tY + 0.35, tS + 0.12]: place(footstep(), ft, 0.4, 0.1, 0.08)
place(chord_pad(1.0, [146.83, 220, 293.66], 900), tT - 0.1, 0.16, 0, 0.4)
for k, f in enumerate([587.33, 739.99, 880, 1174.66, 1479.98]): place(bell(f, 0.5, 0.6), tT + k * 0.06, 0.08, -0.3 + k * 0.15, 0.5)
place(whoosh(0.6, 150, 900, 0.5, 1.0, 0.6), tT - 0.05, 0.3, 0, 0.3)
place(impact(0.6, 160, 60, 0.35, 0.5), tCu, 0.3, 0, 0.4); sparkle(tCu, 14, 0.35, 0.12)
for k in range(3): place(blip(2600, 0.05, 1.5), tCu + 0.05 + k * 0.09, 0.05, 0.4, 0.1)

# ---- 3 RISE -------------------------------------------------------
s0, s1 = S['sRise'], S['sTop']
place(riser(s1 - s0 + 0.05, 150, 4000, True), s0, 0.28, 0, 0.15)
bed(s0, s1, lambda n: wind(n, 300, 1400, 3, 0.4) * np.linspace(0.4, 1, n), 0.2)
place(whoosh(0.6, 250, 2500, 0.4, 1.1, 0.4), W('rising') - 0.1, 0.35, 0, 0.2)
for k in range(4): place(passby(0.35, 300, 1800), s0 + 0.3 + k * 0.28, 0.12, (-0.7, 0.6, -0.5, 0.7)[k])

# ---- 4 TOP-DOWN ---------------------------------------------------
s0, s1 = S['sTop'], S['sRoofs']
bed(s0, s1 + 0.05, lambda n: wind(n, 200, 700, 0.8, 0.5), 0.14, 0, 0.2)
place(pop(1500, 900, 0.05), W('streets') + 0.05, 0.18, 0.2, 0.2)
place(power_down(0.55, 900, 220), W('shrink'), 0.12, 0, 0.3)
place(whoosh(0.5, 300, 120, 0.3, 1.2, 0.5), W('shrink'), 0.2, 0, 0.2)
tR = W('rooftops'); place(laser(0.45, 400, 2400), tR, 0.08, 0, 0.5); sparkle(tR, 16, 0.5, 0.1, MID + HI[:4])
place(whoosh(0.6, 800, 5000, 0.35, 1.3, 0.1), tR, 0.22, pan_sweep(int(0.6 * SR), -0.5, 0.5), 0.3)

# ---- 5 ROOFTOPS SLIDE ---------------------------------------------
s0, s1 = S['sRoofs'], S['sBeach']
bed(s0, s1, lambda n: wind(n, 500, 2200, 4, 0.3), 0.18)
for k in range(5): place(passby(0.3, 200, 1500, 0.9), s0 + 0.08 + k * 0.17, 0.2, pan_sweep(int(0.3 * SR), 0.8, -0.8))
place(whoosh(0.4, 350, 3000, 0.5, 1.0, 0.4), W('past') - 0.1, 0.3, pan_sweep(int(0.4 * SR), 0.6, -0.6))

# ---- 6 BEACH ------------------------------------------------------
s0, s1 = S['sBeach'], S['sMap']
bed(s0, s1 + 0.05, ocean, 0.2, 0, 0.1)
place(seagull(), s0 + 0.25, 0.08, -0.6, 0.4); place(seagull(), s0 + 1.1, 0.06, 0.7, 0.5)
tG = W('glitter'); sparkle(tG - 0.05, 40, 0.9, 0.12, HI, 0.2, 0.55)
place(chord_pad(1.2, [587.33, 739.99, 880, 1108.73], 4000), tG, 0.05, 0, 0.6)

# ---- 7 MAP --------------------------------------------------------
s0, s1 = S['sMap'], S['sRoutes']
bed(s0, s1, lambda n: hum(n, 55, 300), 0.05)
place(pop(500, 160, 0.09), s0 + 0.2, 0.35, -0.6, 0.15); place(pop(560, 170, 0.09), s0 + 0.3, 0.35, 0.6, 0.15)
place(pop(1200, 600, 0.05), W('your') - 0.05, 0.2, -0.4, 0.1)
t = W('your') - 0.05; tEnd = W('straight')
while t < tEnd: place(tick(2000, 6000, 0.003), t, 0.08, -0.3, 0.05); t += 0.07
tSt = W('straight')
place(laser(0.35, 250, 3000), tSt, 0.22, pan_sweep(int(0.35 * SR), -0.6, 0.6), 0.3)
place(whoosh(0.5, 400, 3500, 0.4, 1.2, 0.2), tSt + 0.05, 0.3, pan_sweep(int(0.5 * SR), -0.6, 0.6), 0.2)
place(bell(1318.5, 0.7), tSt + 0.17, 0.14, 0.4, 0.4); place(bell(1760, 0.7), tSt + 0.27, 0.12, 0.4, 0.4)
place(rip(0.2), tSt + 0.15, 0.08, -0.4, 0.1)
place(pop(1400, 700, 0.05), W('line'), 0.18, 0.1, 0.2)

# ---- 8 ROUTES (night) ---------------------------------------------
s0, s1 = S['sRoutes'], S['sPads']
bed(s0, s1, lambda n: chord_pad(n / SR, [73.42, 110, 146.83, 220], 700), 0.12, 0, 0.3)
tR2 = W('rooftops', 2)
for k in range(30): place(blip(rng.choice(MID + HI[:4]), 0.07, 2.0), tR2 - 0.06 + k * 0.026, 0.05, rng.uniform(-0.8, 0.8), 0.4)
place(chord_pad(0.9, [293.66, 369.99, 440, 554.37, 739.99], 3000), W('routes'), 0.1, 0, 0.6); sparkle(W('routes'), 10, 0.4, 0.08)

# ---- 9a PADS WAVE -------------------------------------------------
s0 = S['sPads']; tE = W('every')
for k in range(46):
    tt = tE + 0.02 + 0.55 * np.sqrt(rng.uniform())
    place(pop(rng.uniform(700, 1500), 250, 0.05), tt, 0.17, rng.uniform(-0.9, 0.9), 0.12)
for k in range(10): place(tick(1500, 4000, 0.003), s0 + 0.15 + k * 0.08, 0.06, 0.0)

# ---- 9b PAD CLOSE-UP ----------------------------------------------
s0 = S['sPad']; tTouch = W('pad') + 0.04
place(whoosh(tTouch - s0 + 0.05, 3000, 300, 0.85, 1.1, 0.5), s0, 0.32, 0, 0.2)
for k in range(8): place(blip(1760, 0.04, 1.2), s0 + k * 0.08, 0.025, -0.6 + k * 0.15, 0.2)
place(impact(1.0, 110, 40, 0.7), tTouch, 0.7, 0, 0.35)
place(bell(220, 1.0, 0.4), tTouch, 0.12, 0, 0.5); sparkle(tTouch, 12, 0.3, 0.1)

# ---- 10 SKY FILLS UP FAST -----------------------------------------
s0, s1 = S['sFill'], S['sJam']
for k in range(34):
    tt = s0 + (s1 - s0 - 0.15) * (k / 34) ** 0.65
    place(passby(rng.uniform(0.25, 0.5), 200, rng.uniform(1200, 3000)), tt, 0.06 + 0.1 * k / 34, pan_sweep(int(0.5 * SR), rng.choice([-0.9, 0.9]), 0) * 1, 0.05)
t = s0 + 0.1; dt = 0.13
while t < s1 - 0.05: place(tick(2500, 7000, 0.0025), t, 0.07, 0.15); dt = max(0.028, dt * 0.88); t += dt
bed(s0, s1, lambda n: murmur(n, 10) * np.linspace(0.1, 1, n) ** 1.5, 0.08)
place(riser(s1 - s0, 300, 2500, True), s0, 0.12)
hit(W('fast'), 1.0, 0.65, 0.35)

# ---- 11 SKY JAM ---------------------------------------------------
s0, s1 = S['sJam'], S['sStorm']
bed(s0, s1, lambda n: hum(n, 46, 400) * 0.7 + 0.3 * norm(lp(pink(n), 300)), 0.1)
bed(s0, s1, lambda n: hum(n, 120, 2500), 0.015, 0)
for (o, x), (f1, f2) in zip([(0.1, 420), (0.32, 1650), (0.55, 700), (0.78, 1760), (1.0, 380), (1.2, 1420), (1.38, 980)],
                            [(415, 523), (370, 466), (440, 554), (392, 494), (349, 440), (415, 523), (466, 587)]):
    place(horn(rng.uniform(0.22, 0.4), f1, f2), s0 + o, 0.16, (x - 1080) / 1080 * 0.8, 0.15)
hit(W('jam'), 0.7, 0.4, 0.3, 120, 0.5)

# ---- 12 STORM -----------------------------------------------------
s0, s1 = S['sStorm'], S['sBurn']
tSt, tSh, tSi = W('storm'), W('shoves'), W('sideways')
bed(s0, s1 + 0.15, rain, 0.16, 0, 0.1)
bed(s0, s1 + 0.15, lambda n: wind(n, 200, 1300, 2.5, 0.7), 0.2, 0, 0.1)
for l, sz, pn in [(tSt - 0.12, 1.2, -0.4), (tSh + 0.02, 0.9, 0.5), (tSi + 0.02, 1.0, 0.1)]:
    place(thunder(sz, 2.0), l, 0.9 * sz, pn, 0.4)
for tt in (tSh, tSi):
    place(whoosh(0.7, 120, 900, 0.35, 0.9, 0.8), tt - 0.1, 0.55, pan_sweep(int(0.7 * SR), -0.8, 0.8), 0.2)
    place(impact(0.8, 70, 35, 0.5, 0.4), tt + 0.05, 0.35, 0.3, 0.2)

# ---- 13 ARMS BURN -------------------------------------------------
s0, s1 = S['sBurn'], S['sRules']
k = int(np.ceil(s0 * 2.6))
while k / 2.6 < s1: place(heartbeat(), k / 2.6, 0.5, 0, 0.1); k += 1
bed(s0, s1, crackle, 0.1, 0, 0.1)
bed(s0, s1, lambda n: norm(sweep_lp(noise(n), np.linspace(400, 2500, n))) * np.linspace(0.3, 1, n), 0.07)
tB = W('burn'); hit(tB, 0.7, 0.4, 0.3, 120, 0.5)
t = tB
while t < s1 - 0.05: place(blip(1980, 0.07, 1.2), t, 0.06, 0.3, 0.1); t += 0.285

# ---- 14 FLIGHT RULES ----------------------------------------------
s0, s1 = S['sRules'], S['sLanes']
place(whoosh(0.35, 200, 1800, 0.7, 1.1, 0.3), s0, 0.25, 0, 0.1)
for i, (tr_, f) in enumerate([(W('we') + 0.02, 1174.66), (W('flight'), 1318.51), (W('rules') - 0.08, 1479.98)]):
    place(whoosh(0.2, 600, 3000, 0.5, 1.2, 0.1), tr_, 0.1, -0.5, 0.1)
    place(bell(f, 0.45), tr_ + 0.25, 0.13, 0.5, 0.35); place(pop(1600, 900, 0.03), tr_ + 0.24, 0.12, 0.5, 0.1)
tSp = W('rules') + 0.16
place(impact(0.9, 140, 55, 0.35, 1.3), tSp, 0.6, 0.2, 0.25); place(flap(), tSp, 0.25, 0.2, 0.1)
for k in range(6): place(passby(0.5, 250, 1400), s0 + 0.1 + k * 0.22, 0.05, pan_sweep(int(0.5 * SR), -0.8, 0.8))

# ---- 15/16 AIR LANES + SPEED LIMITS -------------------------------
def lanePos(t):
    t0, tl = 21.17, W('limits')
    if t < tl: return (t - t0) * 3000
    p = (tl - t0) * 3000; n = 40; dt = (t - tl) / n
    for i in range(n):
        tt = tl + (i + 0.5) * dt; p += lerp(3000, 850, ioC(prog(tt, tl, tl + 0.7))) * dt
    return p
s0, s1 = S['sLanes'], S['sSports']
bed(s0, s1, lambda n: hum(n, 55, 1200) * 0.6 + 0.4 * chord_pad(n / SR, [110, 164.81, 220], 900), 0.11, 0, 0.2)
prev = np.floor(lanePos(s0) / 1000); t = s0
while t < s1:
    c = np.floor(lanePos(t) / 1000)
    if c != prev: place(passby(0.36, 150, 1100, 0.8), t - 0.18, 0.18, 0, 0.1); prev = c
    t += 0.002
tBu = W('build') - 0.04
for k in range(14): place(blip([220, 277.18, 329.63, 440, 554.37, 659.25, 880][k % 7] * (1 + (k // 7)), 0.09, 2.5), tBu + k * 0.04, 0.045, -0.6 + k * 0.09, 0.3)
place(riser(0.6, 300, 4000, True), tBu, 0.12, 0, 0.2)
tA = W('air'); place(pop(1300, 600, 0.05), tA, 0.2, 0, 0.2); sparkle(tA, 8, 0.3, 0.09)
tSp, tLi = W('speed'), W('limits')
place(pop(900, 300, 0.07), S['sLimit'] + 0.08, 0.2, 0, 0.1)
place(whoosh(0.25, 500, 4000, 0.9, 1.0, 0.3), tSp - 0.2, 0.3, 0, 0.1)
place(impact(1.2, 90, 36, 1.0), tSp + 0.07, 0.85, 0, 0.4); place(clang(300, 0.8), tSp + 0.07, 0.22, 0, 0.5)
place(power_down(0.8, 1400, 180), tLi, 0.2, 0, 0.3)

# ---- 17 SPORTS ----------------------------------------------------
s0, s1 = S['sSports'], S['sCourier']
bed(s0, s1 + 0.1, lambda n: murmur(n, 12), 0.1, 0, 0.3)
tGoal = W('vertical') + 0.28
for tt, pn in [(s0, -0.6), (s0 + 0.42, 0.6), (s0 + 0.72, -0.5)]:
    place(impact(0.5, 160, 70, 0.15, 1.2), tt + 0.01, 0.3, pn, 0.2)
    place(passby(0.4, 400, 3000), tt + 0.02, 0.18, pan_sweep(int(0.4 * SR), pn, -pn))
place(whoosh(0.3, 600, 4000, 0.9, 1.2, 0.2), tGoal - 0.27, 0.25)
place(impact(1.0, 100, 40, 0.8), tGoal, 0.6, 0, 0.4)
place(norm(bp(noise(int(1.3 * SR)), 600, 4000)) * np.minimum(1, T(1.3) / 0.1) * np.exp(-T(1.3) / 0.6), tGoal, 0.22, 0, 0.5)
place(horn(0.5, 233, 294), tGoal + 0.05, 0.1, 0, 0.4); sparkle(tGoal, 20, 0.6, 0.1)
for k in range(10): place(pop(rng.uniform(1500, 2500), 900, 0.02), tGoal + rng.uniform(0, 0.25), 0.06, rng.uniform(-0.7, 0.7))

# ---- 18 COURIERS --------------------------------------------------
s0, s1 = S['sCourier'], S['sWeekend']
bed(s0, s1, lambda n: wind(n, 600, 2400, 3, 0.3), 0.14)
bed(s0, s1, lambda n: hum(n, 38, 250), 0.08)
place(passby(0.55, 200, 2600, 0.9), s0 - 0.05, 0.45, pan_sweep(int(0.55 * SR), -0.9, 0.2), 0.1)
place(horn(0.3, 392, 494), s0 + 0.35, 0.12, -0.6, 0.2); place(horn(0.45, 349, 440), s0 + 0.85, 0.1, 0.5, 0.2)
tSk = W('skip'); place(bell(1567.98, 0.5), tSk + 0.12, 0.16, 0, 0.3); place(bell(2093, 0.6), tSk + 0.24, 0.14, 0, 0.3)
place(pop(1500, 800, 0.04), tSk + 0.38, 0.14, 0.5, 0.2)
place(pop(420, 180, 0.07), W('traffic'), 0.18, -0.4, 0.1)

# ---- 19 WEEKEND ---------------------------------------------------
s0, s1 = S['sWeekend'], S['sGlide']
tW_ = W('weekend')
k = int(np.ceil((s0 * 3.4 - np.pi / 2) / (2 * np.pi)))
while True:
    tt = (np.pi / 2 + 2 * np.pi * k) / 3.4
    if tt > s1: break
    if tt > s0: place(passby(0.6, 300, 2400), tt - 0.3, 0.2, pan_sweep(int(0.6 * SR), 0.8, -0.8), 0.15)
    k += 1
place(pop(700, 300, 0.06), s0 + 0.1, 0.15, 0, 0.1)
place(flap(), tW_, 0.3, 0, 0.15); place(flap(), tW_ + 0.13, 0.22, 0, 0.15)
place(impact(0.6, 200, 80, 0.2, 1.5), tW_ + 0.13, 0.25, 0, 0.3)
for k in range(16): place(pop(rng.uniform(1200, 2600), 700, 0.025), tW_ + 0.14 + rng.uniform(0, 0.3), 0.07, rng.uniform(-0.8, 0.8))
sparkle(tW_ + 0.13, 18, 0.5, 0.11)

# ---- 20 LONG GLIDE (golden hour) ----------------------------------
s0, s1 = S['sGlide'], S['sNotEq']
bed(s0, s1, lambda n: wind(n, 250, 800, 0.6, 0.4), 0.12, 0, 0.2)
bed(s0, s1, ocean, 0.07)
place(chord_pad(s1 - s0 + 0.3, [146.83, 220, 277.18, 369.99, 440], 1600), s0, 0.13, 0, 0.5)
place(seagull(), s0 + 0.5, 0.05, 0.6, 0.6); place(seagull(), s0 + 1.2, 0.04, -0.5, 0.6)
for k in range(12): place(ping(rng.choice(HI[:6]), 0.1), s0 + 0.15 + k * 0.1, 0.035, -0.2 + k * 0.05, 0.5)

# ---- 21 NOT EQUAL -------------------------------------------------
s0, s1 = S['sNotEq'], S['sBars']
place(impact(1.2, 70, 32, 1.4, 0.4), s0 + 0.02, 0.8, 0, 0.5)
place(chord_pad(1.4, [73.42, 77.78, 110, 116.54], 500), s0, 0.2, 0, 0.4)
tI = W("isn't"); place(impact(1.1, 110, 40, 0.8, 1.2), tI + 0.02, 0.8, 0, 0.4); place(glitch(0.25), tI, 0.15, 0.2, 0.1)
place(rip(0.4), W('equally'), 0.25, pan_sweep(int(0.4 * SR), -0.5, 0.5), 0.15)
bed(s0, s1, rain, 0.05, 0.5)

# ---- 22a BARS -----------------------------------------------------
s0, s1 = S['sBars'], S['sSoar']
for i, v in enumerate([1.0, 0.34, 0.16, 0.48, 0.6, 0.8]):
    tt = s0 + 0.04 + i * 0.07; d = 0.35
    f = 220 * 2 ** (2.5 * v * np.linspace(0, 1, int(d * SR)))
    y = np.tanh(2 * np.sin(2 * np.pi * np.cumsum(f) / SR)) * np.exp(-T(d) / 0.2)
    place(lp(y, 3000), tt, 0.07, -0.8 + i * 0.32, 0.2)
tSf = W('safe')
for k in range(3): place(blip(98, 0.14, 4), tSf + k * 0.16, 0.14, 0, 0.1)

# ---- 22b SOAR FOR HOURS -------------------------------------------
s0, s1 = S['sSoar'], S['sRisk']
bed(s0, s1, lambda n: wind(n, 400, 1800, 2, 0.4), 0.2)
for tt, p0 in [(s0 - 0.05, -0.7), (s0 + 0.6, 0.6)]:
    place(passby(0.8, 250, 2800, 0.8), tt, 0.32, pan_sweep(int(0.8 * SR), p0, -p0), 0.2)
tH = W('hours'); prev = 0; t = s0
while t < s1:
    hrs = lerp(0, 4.2, prog(t, s0, tH + 0.4) ** 2); c = int(hrs * 12)
    if c != prev: place(tick(3000, 8000, 0.002), t, 0.08, -0.6); prev = c
    t += 0.002
place(chord_pad(1.2, [293.66, 369.99, 440, 587.33], 2600), tH - 0.1, 0.07, 0, 0.5)

# ---- 22c OTHERS CAN'T RISK IT -------------------------------------
s0, s1 = S['sRisk'], S['sWho']
bed(s0, s1, lambda n: wind(n, 300, 1000, 1.5, 0.75), 0.24, 0, 0.15)
bed(s0, s1, rain, 0.05)
for i in range(6):
    for kk in range(-2, 3):
        tt = (1780 - i * 700 + kk * 3560) / 2800
        if s0 < tt < s1: place(passby(0.4, 300, 2000), tt - 0.2, 0.09, pan_sweep(int(0.4 * SR), -0.9, 0.9))
tC = W("can't"); place(alarm(0.9), tC, 0.06, 0.5, 0.3)
place(impact(0.9, 75, 34, 1.0, 0.5), W('risk') + 0.02, 0.6, 0, 0.35)

# ---- 23 WHO CONTROLS THE SKY? -------------------------------------
s0, s1 = S['sWho'], S['sCrowd']
bed(s0, s1 + 0.2, lambda n: chord_pad(n / SR, [55, 82.41, 110, 130.81], 600), 0.18, 0, 0.3)
for i in range(6):
    place(laser(0.3, 800 + i * 150, 2500 + i * 200) * 0.5, s0 + 0.02 + i * 0.08, 0.05, -0.7 + i * 0.28, 0.3)
    if i in (0, 1, 2, 5): place(tick(1000, 4000, 0.006), s0 + 0.32 + i * 0.08, 0.2, -0.7 + i * 0.28, 0.1)
for tt, sz in [(W('who'), 1.1), (W('controls'), 1.0), (W('sky', 3) - 0.24, 1.3)]:
    place(impact(sz, 70, 30, 1.1, 0.7), tt + 0.02, 0.75, 0, 0.6)
place(glitch(0.15), W('controls') + 0.01, 0.12, 0.3)

# ---- 24 CROWDED PUBLIC SPACE --------------------------------------
s0, s1 = S['sCrowd'], S['sCTA']
place(whoosh(0.9, 150, 2500, 0.6, 1.0, 0.5), s0, 0.32, 0, 0.2)
tCr = W('crowded')
bed(s0, s1, lambda n: murmur(n, 14) * np.clip(np.linspace(-0.2, 1.4, n), 0.15, 1), 0.16, 0, 0.25)
for k in range(14): place(passby(rng.uniform(0.3, 0.6), 250, rng.uniform(1200, 2600)), s0 + 0.4 + k * 0.14, 0.07, pan_sweep(int(0.6 * SR), rng.choice([-0.9, 0.9]), rng.uniform(-0.5, 0.5)))
for k, tt in enumerate(np.linspace(tCr - 0.2, W('space'), 4)[1:]): place(blip([660, 880, 1175][k], 0.08, 2), tt, 0.08, 0, 0.2)
place(alarm(0.25), W('space') + 0.02, 0.05, 0.3, 0.2)

# ---- 25 WILDEST "WHAT IF" -----------------------------------------
s0, s1 = S['sCTA'], S['sComments']
place(riser(s1 - s0 - 0.1, 200, 5000, True), s0, 0.12, 0, 0.2)
hit(W("what's"), 0.7, 0.45, 0.35, 120, 0.5)
hit(W('wildest'), 1.1, 0.8, 0.5)
hit(W('what', 2), 1.1, 0.8, 0.5)
hit(W('question'), 0.9, 0.6, 0.45); sparkle(W('question'), 14, 0.4, 0.1)
k = int(np.ceil(s0 * 2.6 / np.pi))
while k * np.pi / 2.6 < s1:
    tt = k * np.pi / 2.6
    if tt > s0: place(passby(0.55, 300, 2600), tt - 0.27, 0.16, pan_sweep(int(0.55 * SR), -0.7 if k % 2 else 0.7, 0.7 if k % 2 else -0.7))
    k += 1

# ---- 26 COMMENTS --------------------------------------------------
s0, s1 = S['sComments'], S['sIdea']
tD, tCo = W('drop'), W('comments')
for i, f in enumerate([700, 820, 940, 1080]): place(pop(f, f * 0.4, 0.06), tD - 0.05 + i * 0.17 + 0.08, 0.22, 0.2 - i * 0.1, 0.15)
t = tCo - 0.1
while t < tCo + 0.42: place(tick(1800, 5000, 0.002), t, 0.06, -0.2); t += rng.uniform(0.025, 0.045)
place(boing(240), tCo - 0.08, 0.15, 0.6, 0.2)
place(bell(1975.5, 0.4), tCo + 0.05, 0.08, 0.7, 0.3)

# ---- 27 YOUR IDEA COULD BECOME OUR... ------------------------------
s0, s1 = S['sIdea'], S['sEnd']
tI2, tBe = W('idea'), W('become', 2)
place(pop(600, 250, 0.08), s0 + 0.05, 0.25, 0, 0.15)
place(bell(1046.5, 1.0), tI2, 0.2, 0, 0.5); place(bell(1568, 0.9), tI2 + 0.05, 0.12, 0, 0.5)
place(norm(np.sin(2 * np.pi * 120 * T(0.12)) + 0.4 * noise(int(0.12 * SR)) * 0.2) * np.exp(-T(0.12) / 0.05), tI2, 0.06, 0)
sparkle(tI2, 16, 0.5, 0.1)
place(whoosh(0.32, 500, 4000, 0.5, 1.1, 0.4), tBe - 0.05, 0.3, pan_sweep(int(0.32 * SR), -0.4, 0.4), 0.2)
sparkle(tBe + 0.15, 20, 0.4, 0.12)
place(riser(s1 - tBe, 300, 6000, True), tBe, 0.22, 0, 0.2)

# ---- 28 NEXT VIDEO / GO! ------------------------------------------
tN, tV, tGo = W('next'), W('video'), W('go', 2)
place(impact(1.4, 85, 32, 1.4, 1.2), tN + 0.015, 1.0, 0, 0.6)
place(impact(1.3, 100, 36, 1.2, 1.2), tV + 0.015, 0.9, 0, 0.6)
place(pop(900, 400, 0.06), tV + 0.1, 0.25, 0, 0.2)
place(rocket(0.9), tGo - 0.05, 0.55, 0, 0.3)
place(impact(1.2, 120, 40, 0.8, 1.5), tGo + 0.01, 0.8, 0, 0.5)
sparkle(tN, 20, 0.6, 0.12)

# =================================================================== MIX
def make_ir(sec=2.4):
    n = int(sec * SR); t = np.arange(n) / SR; ir = np.zeros((2, n))
    for c in range(2):
        x = noise(n) * np.exp(-t / 0.55)
        x = lp(x, 6000) * 0.6 + lp(x, 1500) * 0.4
        x[:int(0.012 * SR)] = 0
        ir[c] = x
    return ir / np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True)) * 0.9

ir = make_ir()
wet = np.vstack([fftconvolve(rv[c], ir[c])[:N] for c in range(2)])
mix = bus + 0.55 * wet
mix = np.vstack([hp(mix[c], 28, 2) for c in range(2)])
# gentle bus compression (RMS follower) then soft limiter
env = np.sqrt(lp(np.max(mix ** 2, axis=0), 8, 1).clip(1e-9))
ref = np.percentile(env, 99.5)
gain = np.where(env > ref * 0.5, (ref * 0.5 / env) ** 0.35, 1.0)
mix = mix * gain
mix = mix / (np.max(np.abs(mix)) + 1e-12) * 1.25
mix = np.tanh(mix) * 0.7          # peak ~ -3.5 dBFS
fo = int(0.03 * SR); mix[:, -fo:] *= np.linspace(1, 0, fo)
dither = (rng.uniform(-0.5, 0.5, mix.shape) + rng.uniform(-0.5, 0.5, mix.shape)) / 32768
pcm = np.clip(np.round((mix + dither) * 32767), -32768, 32767).astype('<i2')
with wave.open(OUT, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.T.tobytes())
pk = 20 * np.log10(np.max(np.abs(mix)) + 1e-12); rms = 20 * np.log10(np.sqrt(np.mean(mix ** 2)) + 1e-12)
print(f'wrote {OUT}: {N / SR:.2f}s, 48kHz 16-bit stereo, peak {pk:.1f} dBFS, rms {rms:.1f} dBFS')
