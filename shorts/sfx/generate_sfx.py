#!/usr/bin/env python3
"""
Sound design for "What If Dinosaurs Never Went Extinct?" (9:16 Shorts animation).

Every effect is synthesized procedurally (no samples), then placed on the exact
timeline of what-if-dinosaurs-never-went-extinct.html and mixed to a
48 kHz / 24-bit stereo WAV that is exactly 49.6 s long.

    python3 generate_sfx.py [output.wav]

Requires numpy + scipy.
"""
import sys
import wave
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d

SR = 48000
DUR = 49.6
N = int(round(SR * DUR))
_rng = np.random.default_rng(2026)


def rng(seed=None):
    return np.random.default_rng(seed) if seed is not None else _rng


# ----------------------------------------------------------------------------
# core helpers
# ----------------------------------------------------------------------------
def tt(d):
    return np.arange(max(1, int(d * SR))) / SR


def db(x):
    return 10 ** (x / 20)


def norm(x, peak=1.0):
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def fade(x, fi=0.003, fo=0.02):
    x = x.copy()
    n = x.shape[-1]
    a, b = min(n, int(fi * SR)), min(n, int(fo * SR))
    if a > 0:
        x[..., :a] *= np.linspace(0, 1, a)
    if b > 0:
        x[..., -b:] *= np.linspace(1, 0, b)
    return x


def env_exp(d, tau, attack=0.002):
    t = tt(d)
    e = np.exp(-t / tau)
    if attack > 0:
        e *= np.clip(t / attack, 0, 1)
    return e


def env_ar(d, attack, release, curve=2.0):
    t = tt(d)
    up = np.clip(t / max(attack, 1e-4), 0, 1) ** curve
    dn = np.clip((d - t) / max(release, 1e-4), 0, 1) ** curve
    return up * dn


def _sos(kind, f, order=2):
    nyq = SR / 2
    if kind == 'band':
        lo, hi = max(10, f[0]), min(f[1], nyq * 0.95)
        return signal.butter(order, [lo / nyq, hi / nyq], 'band', output='sos')
    return signal.butter(order, min(max(f, 10), nyq * 0.95) / nyq, kind, output='sos')


def lp(x, f, order=2):
    return signal.sosfilt(_sos('low', f, order), x, axis=-1)


def hp(x, f, order=2):
    return signal.sosfilt(_sos('high', f, order), x, axis=-1)


def bp(x, f1, f2, order=2):
    return signal.sosfilt(_sos('band', (f1, f2), order), x, axis=-1)


def reson(x, f, q):
    """RBJ constant-peak band-pass (resonator)."""
    w = 2 * np.pi * f / SR
    al = np.sin(w) / (2 * q)
    b = np.array([al, 0, -al])
    a = np.array([1 + al, -2 * np.cos(w), 1 - al])
    return signal.lfilter(b / a[0], a / a[0], x)


def noise(d, seed=None):
    return rng(seed).standard_normal(int(d * SR))


def pink(d, seed=None):
    w = noise(d, seed)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return norm(signal.lfilter(b, a, w))


def phase(freq):
    return 2 * np.pi * np.cumsum(freq) / SR


def sine(freq, d=None):
    if np.isscalar(freq):
        freq = np.full(len(tt(d)), float(freq))
    return np.sin(phase(freq))


def saw(freq, d=None, harm=40):
    """Band-limited-ish additive saw with time-varying frequency."""
    if np.isscalar(freq):
        freq = np.full(len(tt(d)), float(freq))
    ph = phase(freq)
    out = np.zeros_like(ph)
    for k in range(1, harm + 1):
        mask = (freq * k) < SR * 0.45
        out += mask * np.sin(k * ph) / k
    return out


def square(freq, d=None, harm=11):
    if np.isscalar(freq):
        freq = np.full(len(tt(d)), float(freq))
    ph = phase(freq)
    out = np.zeros_like(ph)
    for k in range(1, harm + 1, 2):
        out += ((freq * k) < SR * 0.45) * np.sin(k * ph) / k
    return out


def expsweep(f0, f1, d, curve=1.0):
    u = tt(d) / d
    return f0 * (f1 / f0) ** (u ** curve)


def stereo(x, pan=0.0):
    """Equal-power pan; pan may be a scalar or an array (-1..1)."""
    if x.ndim == 2:
        return x
    th = (np.asarray(pan) + 1) * np.pi / 4
    return np.stack([x * np.cos(th), x * np.sin(th)]) * np.sqrt(2)


def widen(x, amt=0.5, ms=12):
    """Mono -> wide stereo using a short decorrelating delay on one side."""
    d = int(ms * SR / 1000)
    r = np.concatenate([np.zeros(d), x])[:len(x)]
    return np.stack([x, (1 - amt) * x + amt * r])


def shaped_noise(d, fc, bw, amp, seed=None, nfft=2048, hop=512):
    """Noise sculpted in the STFT domain: a Gaussian band (log-frequency) whose
    centre `fc`, width `bw` (octaves) and level `amp` follow callables of time."""
    r = rng(seed)
    n = int(d * SR)
    nfr = n // hop + 4
    tf = np.arange(nfr) * hop / SR
    freqs = np.fft.rfftfreq(nfft, 1 / SR)
    lf = np.log2(np.maximum(freqs, 15))
    FC = np.log2(np.maximum(np.asarray(fc(tf), float), 15))[:, None]
    BW = np.maximum(np.asarray(bw(tf), float), 0.05)[:, None]
    mag = np.exp(-0.5 * ((lf[None, :] - FC) / BW) ** 2) * np.asarray(amp(tf), float)[:, None]
    spec = mag * np.exp(1j * r.uniform(0, 2 * np.pi, mag.shape))
    frames = np.fft.irfft(spec, nfft, axis=1) * np.hanning(nfft)[None, :]
    out = np.zeros(nfr * hop + nfft)
    for i in range(nfr):
        out[i * hop:i * hop + nfft] += frames[i]
    out = out[nfft // 2:nfft // 2 + n]
    return norm(out)


def bitcrush(x, bits=6, hold=4):
    y = np.repeat(x[::hold], hold)[:len(x)]
    q = 2 ** (bits - 1)
    return np.round(y * q) / q


def sat(x, drive=2.0):
    return np.tanh(x * drive) / np.tanh(drive)


def ir_reverb(rt=1.8, d=2.4, seed=7, bright=6500):
    """Stereo synthetic impulse response (decorrelated exponential noise)."""
    t = tt(d)
    e = np.exp(-6.9 * t / rt)
    L = lp(noise(d, seed) * e, bright)
    R = lp(noise(d, seed + 1) * e, bright)
    pre = int(0.012 * SR)
    L[:pre] = 0
    R[:pre] = 0
    for k, (dl, g) in enumerate([(0.017, .5), (0.023, .4), (0.031, .33), (0.043, .25)]):
        i = int(dl * SR)
        (L if k % 2 == 0 else R)[i] += g
    return np.stack([L, R]) / np.sqrt(np.sum(L ** 2))


# ----------------------------------------------------------------------------
# mixing buses
# ----------------------------------------------------------------------------
MIX = np.zeros((2, N))
VERB = np.zeros((2, N))


def place(x, t, gain_db=-6.0, pan=0.0, rev=0.15, end=None):
    """Drop a mono/stereo sound at time t (seconds); optional hard end time."""
    s = stereo(x, pan) if x.ndim == 1 else x
    s = s * db(gain_db)
    i0 = int(round(t * SR))
    if end is not None:
        s = fade(s[:, :max(1, int((end - t) * SR))], 0, 0.01)
    if i0 < 0:
        s = s[:, -i0:]
        i0 = 0
    if i0 >= N or s.shape[1] == 0:
        return
    n = min(s.shape[1], N - i0)
    MIX[:, i0:i0 + n] += s[:, :n]
    if rev > 0:
        VERB[:, i0:i0 + n] += s[:, :n] * rev


def scatter(fn, t0, t1, rate, gain_db, seed, pan_spread=0.8, rev=0.15, jitter=0.6):
    """Place many short grains from fn(rng) with random timing between t0..t1."""
    r = rng(seed)
    t = t0
    while t < t1:
        place(fn(r), t, gain_db + r.uniform(-4, 1), r.uniform(-pan_spread, pan_spread), rev)
        t += (1 / rate) * (1 + r.uniform(-jitter, jitter))


# ----------------------------------------------------------------------------
# instruments (each returns a mono or stereo array, roughly unit peak)
# ----------------------------------------------------------------------------
def sub_boom(d=1.4, f0=110, f1=36, tau=0.45, bend=0.12):
    t = tt(d)
    f = f1 + (f0 - f1) * np.exp(-t / bend)
    return sine(f) * env_exp(d, tau, 0.001)


def hit(kind='mid', seed=None):
    r = rng(seed)
    if kind == 'big':
        d = 2.2
        x = 1.0 * sub_boom(d, 130, 34, .6, .1)
        x += .55 * lp(noise(d, r.integers(1e9)), 700) * env_exp(d, .12)
        x += .35 * hp(noise(d, r.integers(1e9)), 2500) * env_exp(d, .01)
        x = sat(x, 2.2)
        x = x + .5 * braam(55, d, seed=r.integers(1e9))
        return widen(norm(x), .5)
    if kind == 'small':
        d = .45
        x = .7 * sub_boom(d, 180, 60, .09, .03)
        x += .6 * bp(noise(d, r.integers(1e9)), 300, 3000) * env_exp(d, .03)
        x += .4 * hp(noise(d, r.integers(1e9)), 4000) * env_exp(d, .006)
        return widen(norm(sat(x, 1.8)), .3)
    if kind == 'ice':
        d = 1.6
        x = .7 * sub_boom(d, 120, 45, .25)
        tink = sum(np.sin(2 * np.pi * f * tt(d)) * env_exp(d, r.uniform(.15, .5)) * r.uniform(.2, .5)
                   for f in r.uniform(2400, 7500, 9))
        x += .5 * tink + .4 * hp(noise(d, r.integers(1e9)), 5000) * env_exp(d, .05)
        return widen(norm(x), .7)
    if kind == 'warm':
        d = 2.0
        x = .8 * sub_boom(d, 90, 45, .5, .15)
        x += .45 * lp(noise(d, r.integers(1e9)), 500) * env_exp(d, .1)
        chord = sum(saw(f, d, 20) for f in (110, 164.8, 220, 277.2)) * env_exp(d, .7, .01)
        x += .35 * lp(chord, 1400)
        return widen(norm(x), .6)
    if kind == 'stone':
        d = 1.2
        x = .9 * sub_boom(d, 120, 40, .35)
        grit = bp(noise(d, r.integers(1e9)), 200, 2500) * env_exp(d, .09)
        x += .8 * sat(grit * 3, 2) * .4
        return widen(norm(sat(x, 2)), .4)
    # mid
    d = 1.2
    x = .9 * sub_boom(d, 140, 42, .32, .07)
    x += .6 * lp(noise(d, r.integers(1e9)), 1200) * env_exp(d, .06)
    x += .35 * hp(noise(d, r.integers(1e9)), 3000) * env_exp(d, .008)
    return widen(norm(sat(x, 2)), .4)


def braam(root=55, d=2.0, seed=None):
    r = rng(seed)
    x = np.zeros(len(tt(d)))
    for f in (root, root * 1.5, root * 2):
        for det in (-0.4, 0.0, 0.45):
            x += saw(f * 2 ** (det / 12 + r.uniform(-.01, .01)), d, 30)
    e = env_exp(d, .9, .025)
    x = lp(x, 900) * e
    return norm(sat(x * 1.5, 2.5))


def whoosh(d=0.5, peak=0.7, f0=250, f1=3500, f2=500, bw=1.1, pan_from=-0.8, pan_to=0.8, seed=None):
    tp = d * peak

    def fc(t):
        return np.where(t < tp, f0 * (f1 / f0) ** (np.clip(t / tp, 0, 1) ** 1.5),
                        f1 * (f2 / f1) ** np.clip((t - tp) / max(d - tp, 1e-3), 0, 1))

    def amp(t):
        return np.where(t < tp, np.clip(t / tp, 0, 1) ** 2.5, np.exp(-np.clip(t - tp, 0, None) / ((d - tp) * .35 + 1e-3)))

    x = shaped_noise(d, fc, lambda t: np.full_like(t, bw), amp, seed)
    p = np.linspace(pan_from, pan_to, len(x))
    return fade(stereo(x, p), .002, .03)


def riser(d=1.5, f0=200, f1=6000, tone=True, seed=None):
    t = tt(d)
    u = t / d
    x = shaped_noise(d, lambda tf: f0 * (f1 / f0) ** (tf / d) ** 1.6, lambda tf: 0.6 + 0 * tf,
                     lambda tf: (tf / d) ** 2.2, seed)
    if tone:
        f = expsweep(80, 900, d, 1.8)
        trem = 0.6 + 0.4 * np.sin(phase(4 + 26 * u ** 2))
        x = x + .45 * norm(saw(f, d, 18)) * u ** 2 * trem
    return widen(norm(x), .6)


def subdrop(d=0.9, f0=150, f1=32):
    return sine(expsweep(f0, f1, d, .6)) * env_exp(d, d * .55, .004)


def reverse_swell(d=0.6, seed=None):
    x = shaped_noise(d, lambda t: 2000 + 6000 * (t / d), lambda t: 0.9 + 0 * t, lambda t: (t / d) ** 3, seed)
    x = x + .6 * sine(expsweep(60, 160, d)) * (tt(d) / d) ** 3
    return widen(norm(x), .7)


def glitch(d=0.24, seed=None):
    r = rng(seed)
    n = int(d * SR)
    L, R = np.zeros(n), np.zeros(n)
    i, prev = 0, None
    while i < n:
        seg = int(r.uniform(.012, .04) * SR)
        k = r.integers(0, 5)
        if k == 0 or prev is None:
            s = square(r.uniform(180, 2600), seg / SR, 9)
        elif k == 1:
            s = noise(seg / SR, r.integers(1e9))
        elif k == 2:
            s = prev
        elif k == 3:
            s = sine(expsweep(r.uniform(300, 3000), r.uniform(80, 400), seg / SR))
        else:
            s = np.zeros(seg)
        s = bitcrush(norm(s[:seg]) * .9, r.integers(3, 7), r.integers(2, 9)) if np.any(s) else s
        s = fade(s, .001, .002)
        p = r.uniform(-1, 1)
        m = min(len(s), n - i)
        L[i:i + m] += s[:m] * np.cos((p + 1) * np.pi / 4)
        R[i:i + m] += s[:m] * np.sin((p + 1) * np.pi / 4)
        prev = s
        i += seg
    return norm(np.stack([L, R]))


def roar(d=1.3, f0=82, seed=None, bright=1.0):
    r = rng(seed)
    t = tt(d)
    u = t / d
    jit = lp(r.standard_normal(len(t)), 9) * 18
    f = f0 * (1 + .45 * np.sin(np.pi * u ** .8)) * (1 + 0.04 * jit / (np.std(jit) + 1e-9))
    src = saw(f, d, 45) + .6 * saw(f * .5, d, 45)
    growl = 1 + .7 * np.sin(phase(26 + 10 * np.sin(np.pi * u))) * (.5 + .5 * lp(r.standard_normal(len(t)), 40) / 3)
    src = src * growl
    form = (reson(src, 380, 3) + .9 * reson(src, 820, 4) + .55 * reson(src, 1650 * bright, 5) + .3 * reson(src, 2900 * bright, 6))
    breath = bp(noise(d, r.integers(1e9)), 250, 3500) * (.5 + .5 * np.sin(phase(31)))
    x = norm(form) + .45 * norm(breath)
    e = np.clip(t / .12, 0, 1) * np.clip((d - t) / .4, 0, 1) ** 1.5 * (.75 + .25 * np.sin(np.pi * u))
    x = sat(x * e * 2.2, 2.5)
    x = lp(x, 6500) + .5 * sub_boom(d, 70, 45, .6, .3) * e
    return widen(norm(x), .5)


def screech(d=0.5, f0=1100, seed=None):
    r = rng(seed)
    t = tt(d)
    u = t / d
    f = f0 * (1 + .35 * np.sin(np.pi * u)) * (1 + .03 * np.sin(phase(18)))
    src = saw(f, d, 14) * (1 + .5 * np.sin(phase(55)))
    x = reson(src, 1800, 3) + .7 * reson(src, 3200, 4) + .3 * bp(noise(d, r.integers(1e9)), 2000, 6000)
    e = np.clip(t / .03, 0, 1) * np.clip((d - t) / .2, 0, 1)
    return norm(sat(x * e * 2, 2))


def heartbeat(gain2=0.7):
    d = .7
    x = np.zeros(len(tt(d)))
    for off, g in ((0, 1.0), (.17, gain2)):
        i = int(off * SR)
        th = sine(expsweep(62, 38, .25)) * env_exp(.25, .07, .003)
        th += .25 * lp(noise(.25, 3 + i), 180) * env_exp(.25, .03)
        x[i:i + len(th)] += g * th
    return norm(lp(x, 400))


def stomp(seed=None):
    r = rng(seed)
    d = 3.0
    x = 1.0 * sub_boom(d, 90, 26, 1.1, .18)
    x += .8 * lp(noise(d, r.integers(1e9)), 450) * env_exp(d, .18)
    x += .4 * hp(noise(d, r.integers(1e9)), 1800) * env_exp(d, .02)
    x = sat(x * 1.4, 2.6)
    deb = np.zeros(len(x))
    for _ in range(90):
        i = int(r.exponential(.35) * SR) + int(.05 * SR)
        if i >= len(x) - 4000:
            continue
        g = bp(noise(.06, r.integers(1e9)), r.uniform(150, 600), r.uniform(900, 3000)) * env_exp(.06, .012)
        deb[i:i + len(g)] += g * r.uniform(.05, .3) * np.exp(-i / SR / .8)
    return widen(norm(x + deb), .5)


def footstep_heavy(seed=None):
    r = rng(seed)
    d = .9
    x = sub_boom(d, 75, 34, .22, .05) + .5 * lp(noise(d, r.integers(1e9)), 350) * env_exp(d, .06)
    return norm(sat(x, 1.8))


def step_light(r):
    d = .09
    x = bp(noise(d, r.integers(1e9)), r.uniform(250, 500), r.uniform(1800, 4000)) * env_exp(d, .018)
    return norm(x + .4 * sine(r.uniform(90, 140), d) * env_exp(d, .02))


def bird(r):
    d = r.uniform(.25, .5)
    reps = r.integers(2, 4)
    out = np.zeros(int(d * SR))
    seg = len(out) // reps
    base = r.uniform(2600, 4200)
    for k in range(reps):
        ln = int(seg * .7)
        u = np.arange(ln) / ln
        f = base * (1 + .3 * np.sin(np.pi * u) * r.choice([-1, 1]))
        out[k * seg:k * seg + ln] += np.sin(phase(f)) * np.sin(np.pi * u) ** 2
    return norm(out)


def insects(d, seed=None):
    x = shaped_noise(d, lambda t: 6500 + 0 * t, lambda t: .35 + 0 * t, lambda t: 1 + 0 * t, seed)
    am = .55 + .45 * np.sign(np.sin(phase(np.full(len(x), 38.0))))
    return widen(norm(lp(x * am, 9000)), .9)


def crackle(d, rate, f1=1200, f2=5000, ring=False, seed=None):
    r = rng(seed)
    n = int(d * SR)
    out = np.zeros(n)
    k = int(rate * d)
    for _ in range(k):
        i = r.integers(0, max(1, n - 3000))
        if ring:
            gd = .05
            g = np.sin(2 * np.pi * r.uniform(f1, f2) * tt(gd)) * env_exp(gd, r.uniform(.004, .02))
        else:
            gd = r.uniform(.002, .008)
            g = bp(noise(gd, r.integers(1e9)), f1, f2) * env_exp(gd, gd / 3)
        out[i:i + len(g)] += g * r.uniform(.1, 1) ** 2
    return widen(norm(out), .8)


def bubbles(d, rate, seed=None):
    r = rng(seed)
    n = int(d * SR)
    out = np.zeros(n)
    for _ in range(int(rate * d)):
        gd = r.uniform(.025, .07)
        f0 = r.uniform(300, 900)
        g = np.sin(phase(expsweep(f0, f0 * r.uniform(1.8, 2.6), gd))) * env_ar(gd, .004, gd * .8, 1)
        i = r.integers(0, max(1, n - len(g)))
        out[i:i + len(g)] += g * r.uniform(.2, 1)
    return widen(norm(lp(out, 2500)), .7)


def tick(f=2800, d=.05, seed=None):
    x = np.sin(2 * np.pi * f * tt(d)) * env_exp(d, .006)
    x += .5 * hp(noise(d, seed), 3000) * env_exp(d, .0015)
    return norm(x)


def blip(f, d=.09, kind='sine', tau=.03):
    x = (square(f, d, 7) if kind == 'square' else sine(f, d)) * env_exp(d, tau, .002)
    return norm(x)


def ping(f, d=.6, tau=.18):
    t = tt(d)
    x = (np.sin(2 * np.pi * f * t) + .3 * np.sin(2 * np.pi * f * 2.01 * t) * np.exp(-t / (tau * .5))) * env_exp(d, tau)
    return norm(x)


def pop(f0=900, f1=180, d=.09):
    x = sine(expsweep(f0, f1, d, .5)) * env_exp(d, .03, .001)
    return norm(x + .2 * hp(noise(d), 2000) * env_exp(d, .002))


def bell(f, d=1.6, partials=((1, 1, 1.2), (2.0, .5, .8), (3.01, .35, .5), (4.2, .2, .35), (5.43, .12, .25))):
    t = tt(d)
    x = sum(a * np.sin(2 * np.pi * f * m * t) * np.exp(-t / dec) for m, a, dec in partials)
    return norm(x * np.clip(t / .003, 0, 1))


def chime_arp(notes, step=.07, d=1.8):
    out = np.zeros(int((d + step * len(notes)) * SR))
    for k, f in enumerate(notes):
        b = bell(f, d)
        i = int(k * step * SR)
        out[i:i + len(b)] += b * (1 - .05 * k)
    return widen(norm(out), .6)


def sparkle(d, rate, seed=None, lo=3500, hi=10000):
    r = rng(seed)
    n = int(d * SR)
    out = np.zeros(n)
    for _ in range(int(rate * d)):
        gd = .25
        g = np.sin(2 * np.pi * r.uniform(lo, hi) * tt(gd)) * env_exp(gd, r.uniform(.03, .1))
        i = r.integers(0, max(1, n - len(g)))
        out[i:i + len(g)] += g * r.uniform(.1, 1) ** 2
    return widen(norm(out), .9)


def glass_crack(seed=None):
    r = rng(seed)
    d = 1.0
    x = .9 * hp(noise(d, r.integers(1e9)), 2000) * env_exp(d, .005)
    for f in r.uniform(1800, 6500, 6):
        x += r.uniform(.15, .4) * np.sin(2 * np.pi * f * tt(d)) * env_exp(d, r.uniform(.05, .15))
    x += .6 * sub_boom(d, 160, 60, .12, .02)
    return widen(norm(x), .5)


def shatter(seed=None):
    r = rng(seed)
    d = 2.2
    n = int(d * SR)
    x = .9 * bp(noise(d, r.integers(1e9)), 1800, 12000) * env_exp(d, .12)
    for _ in range(160):
        gd = .3
        g = np.sin(2 * np.pi * r.uniform(2200, 11000) * tt(gd)) * env_exp(gd, r.uniform(.02, .09))
        i = int(min(r.exponential(.25), 1.6) * SR)
        if i + len(g) < n:
            x[i:i + len(g)] += g * r.uniform(.05, .5) * np.exp(-i / SR / .7)
    x += .9 * sub_boom(d, 120, 32, .5, .08)
    return widen(norm(sat(x, 1.5)), .9)


def rewind(d=.65, seed=None):
    t = tt(d)
    u = t / d
    f = 180 * (2400 / 180) ** (u ** 1.4)
    x = saw(f, d, 10) * (.5 + .5 * np.sign(np.sin(phase(18 + 60 * u))))
    x = bp(x, 300, 5000) + .3 * hp(noise(d, seed), 3000)
    return widen(norm(x * env_ar(d, .02, .05, 1)), .5)


def typing(n_keys, rate, seed=None):
    r = rng(seed)
    d = n_keys / rate + .2
    out = np.zeros(int(d * SR))
    for k in range(n_keys):
        gd = .05
        g = .7 * hp(noise(gd, r.integers(1e9)), 2500) * env_exp(gd, .0025) + .5 * sine(r.uniform(160, 260), gd) * env_exp(gd, .012)
        i = int((k / rate + r.uniform(-.01, .01)) * SR)
        i = max(0, i)
        out[i:i + len(g)] += g * r.uniform(.6, 1)
    return norm(out)


def slash(seed=None):
    w = whoosh(.28, .6, 1200, 7000, 2500, .7, -.6, .6, seed)
    t = tt(.9)
    ring = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((2150, .5), (3320, .35), (5040, .2))) * env_exp(.9, .25)
    out = np.zeros((2, max(w.shape[1], len(ring))))
    out[:, :w.shape[1]] += w
    out[:, :len(ring)] += .3 * stereo(ring)
    return norm(out)


def siren(d, rate=1.6):
    t = tt(d)
    f = 620 + 260 * (.5 - .5 * np.cos(2 * np.pi * rate * t))
    x = bp(square(f, None, 7), 400, 3000)
    return norm(x * env_ar(d, .05, .1, 1))


def whale(d=1.4, f0=210, seed=None):
    t = tt(d)
    u = t / d
    f = f0 * (1 + .9 * np.sin(np.pi * u ** .7)) * (1 + .015 * np.sin(phase(5.5)))
    x = sine(f) + .4 * sine(f * 2) + .15 * sine(f * 3)
    return norm(lp(x, 1400) * env_ar(d, .25, .4, 1.5))


def pad(d, freqs, cutoff=1800, attack=.3, release=.4, seed=None):
    r = rng(seed)
    x = np.zeros(len(tt(d)))
    for f in freqs:
        for det in (-.07, .07):
            x += saw(f * 2 ** (det / 12), d, 16) * (.7 + .3 * np.sin(phase(np.full(len(x), r.uniform(.2, .6)))))
    return widen(norm(lp(x, cutoff) * env_ar(d, attack, release, 1.2)), .8)


def drone(d, f=41.2, seed=None):
    x = sine(f, d) + .5 * sine(f * 2.003, d) + .25 * sine(f * 3.01, d)
    x += .5 * lp(pink(d, seed), 160)
    return widen(norm(x * env_ar(d, .2, .3, 1)), .6)


def wind(d, fc=700, bw=1.3, seed=None, gust=.5):
    r = rng(seed)
    ph1, ph2 = r.uniform(0, 6, 2)
    x = shaped_noise(d, lambda t: fc * (1 + .5 * np.sin(t * 1.3 + ph1)), lambda t: bw + 0 * t,
                     lambda t: (1 - gust) + gust * (.5 + .5 * np.sin(t * 2.1 + ph2)), seed)
    return widen(norm(x), .9)


def rumble(d, seed=None):
    x = lp(pink(d, seed), 110, 4) + .4 * sine(38, d) * (.6 + .4 * np.sin(phase(np.full(len(tt(d)), .7))))
    return widen(norm(x), .4)


def horn(seed=None):
    d = .45
    x = square(392, d, 9) + square(494, d, 9)
    return norm(lp(x, 1600) * env_ar(d, .02, .08, 1))


def scribble(d, seed=None):
    r = rng(seed)
    x = bp(noise(d, r.integers(1e9)), 1800, 4200)
    mod = np.clip(lp(r.standard_normal(len(x)), 14) * 6, 0, 1)
    return widen(norm(x * mod), .6)


def zap_down(d=.7):
    f = expsweep(1800, 110, d, .7) * (1 + .06 * np.sin(phase(np.full(len(tt(d)), 28.0))))
    x = .6 * square(f, None, 9) + sine(f)
    return widen(norm(lp(x, 5000) * env_ar(d, .01, .12, 1)), .4)


def flutter(d, rate=14, seed=None):
    r = rng(seed)
    n = int(d * SR)
    out = np.zeros(n)
    t = 0.0
    while t < d - .05:
        gd = .04
        g = bp(noise(gd, r.integers(1e9)), 400, 2600) * env_ar(gd, .006, .03, 1)
        i = int(t * SR)
        out[i:i + len(g)] += g * r.uniform(.4, 1)
        t += 1 / (rate * r.uniform(.7, 1.3))
    return widen(norm(out), .8)


def door_creak(d=1.2, seed=None):
    r = rng(seed)
    f = 70 + 25 * lp(r.standard_normal(len(tt(d))), 6) / .05
    f = np.clip(f, 40, 160)
    x = bp(saw(f, None, 30), 180, 1400) * (.6 + .4 * np.abs(lp(r.standard_normal(len(f)), 30)) * 4)
    return norm(sat(x * env_ar(d, .15, .3, 1), 1.5))


def confetti(seed=None):
    d = 1.8
    x = np.zeros((2, int(d * SR)))
    p = pop(1600, 300, .08)
    x[:, :len(p)] += stereo(p)
    s = sparkle(1.6, 90, seed, 4000, 11000)
    x[:, int(.03 * SR):int(.03 * SR) + s.shape[1]] += .7 * s
    rust = hp(noise(1.5, seed), 3500) * env_exp(1.5, .5)
    x[:, int(.05 * SR):int(.05 * SR) + len(rust)] += .25 * widen(rust)
    return norm(x)


def stinger(seed=None):
    d = 2.6
    chord = (220, 277.18, 329.63, 440, 554.37, 659.25)
    x = sum(bell(f, d) for f in chord) * .4
    x += .6 * lp(sum(saw(f, d, 16) for f in chord[:4]), 2200) * env_exp(d, .9, .01)
    x = norm(x) + .8 * sub_boom(d, 110, 40, .6)
    return widen(norm(x), .8)


def hum(d, f=60, seed=None):
    x = saw(f, d, 12) * .4 + .2 * sine(9000, d)
    return norm(lp(x, 2500))


# ----------------------------------------------------------------------------
# timeline — mirrors CUTS in the HTML (start time, transition into the cut)
# ----------------------------------------------------------------------------
CUTS = [(0, 'none'), (.85, 'flash'), (1.9, 'glitch'), (3.12, 'whip'), (4.6, 'zoom'), (6.14, 'zoom'),
        (7.6, 'flash'), (9.24, 'zoom'), (10.3, 'flash'), (11.1, 'whip'), (11.96, 'glitch'), (13.2, 'flash'),
        (14.62, 'whip'), (15.7, 'flash'), (17.24, 'whip'), (18.3, 'glitch'), (19.3, 'whip'), (20.22, 'flash'),
        (21.3, 'whip'), (22.25, 'zoom'), (23.36, 'flash'), (25.3, 'zoom'), (26.44, 'flash'), (27.5, 'whip'),
        (28.55, 'zoom'), (29.58, 'whip'), (31.0, 'zoom'), (32.4, 'whip'), (33.7, 'flash'), (35.0, 'whip'),
        (36.5, 'glitch'), (38.06, 'whip'), (39.7, 'none'), (40.9, 'flash'), (42.3, 'zoom'), (43.86, 'glitch'),
        (45.9, 'whip'), (47.4, 'zoom')]


def transitions():
    for ci, (t, tr) in enumerate(CUTS):
        sd = 1 if ci % 2 else -1          # same direction rule as render() in the HTML
        if tr == 'whip':
            d, pk = .42, .78
            place(whoosh(d, pk, 280, 4200, 700, 1.1, .9 * sd, -.9 * sd, seed=1000 + ci), t - d * pk, -8, rev=.08)
        elif tr == 'zoom':
            place(reverse_swell(.32, 1100 + ci), t - .32, -17, rev=.05)
            place(subdrop(.8), t, -9, rev=0)
            place(hit('small', 1200 + ci), t, -16, rev=.1)
        elif tr == 'glitch':
            place(glitch(.24, 1300 + ci), t, -12, rev=.05)
            place(hit('small', 1400 + ci), t, -14, rev=.05)
        elif tr == 'flash':
            place(reverse_swell(.28, 1500 + ci), t - .28, -18, rev=.05)
            place(hit('mid', 1600 + ci), t, -10, rev=.2)
            place(widen(hp(noise(.4, 1700 + ci), 5000) * env_exp(.4, .07)), t, -20, rev=.2)


def beats(t0, t1, w):
    """Times where the HTML's pulse max(0, sin(t*w))^k peaks inside [t0, t1)."""
    out, n = [], 0
    while True:
        t = (np.pi / 2 + 2 * np.pi * n) / w
        if t >= t1:
            return out
        if t >= t0:
            out.append(t)
        n += 1


BED_BOOST = 5.0   # ambience beds sit a little hotter so quiet scenes stay audible under the voice


def bed(x, t0, t1, gain_db, rev=.1, pan=0.0):
    place(fade(x if x.ndim == 2 else stereo(x, pan), .04, .06), t0, gain_db + BED_BOOST, rev=rev, end=t1)


def cues():
    pent = [523.25, 587.33, 659.25, 783.99, 880, 1046.5, 1174.66, 1318.51]

    # 1a  HOOK — eye snaps open, WHAT IF slam
    place(hit('big', 1), .03, -1, rev=.3)
    bed(drone(1.0, 41.2, 2), 0, .9, -17)
    eye = shaped_noise(.36, lambda t: 250 * 10 ** (t / .36), lambda t: .7 + 0 * t, lambda t: np.sin(np.pi * t / .36) ** 2, 3)
    place(widen(eye), .04, -15, rev=.25)
    # 1b  DINOSAURS — roar, letters drop
    place(roar(1.15, 78, 11), .95, -2, rev=.35)
    bed(wind(1.1, 500, 1.4, 12), .85, 1.9, -22)
    for i in range(9):
        place(tick(1700 + i * 110, .04, 100 + i), 1.05 + i * .025, -24, pan=-.6 + i * .15, rev=.05)
    place(hit('small', 5), 1.24, -12)
    # 1c  NEVER WENT ~~EXTINCT~~ ?
    bed(hum(1.25, 60, 13), 1.9, 3.12, -30)
    bed(drone(1.3, 36.7, 14), 1.9, 3.12, -16)
    place(hit('stone', 15), 2.22, -4, rev=.25)
    place(slash(16), 2.47, -8, rev=.2)
    place(pop(400, 1400, .14), 2.74, -13)
    place(hit('small', 17), 2.8, -11)

    # 2a  THE ASTEROID — fire roar crescendo, alert beeps
    bed(rumble(1.6, 20), 3.12, 4.6, -10)
    fire = shaped_noise(1.55, lambda t: 250 + 900 * (t / 1.55), lambda t: 1.4 + 0 * t, lambda t: .25 + .75 * (t / 1.55) ** 1.5, 21)
    bed(widen(fire), 3.12, 4.6, -7, rev=.15)
    bed(crackle(1.5, 70, 800, 4000, seed=22), 3.12, 4.6, -15)
    for k in range(3):
        place(blip(1318.5, .1, 'square', .04), 3.14 + .5 * k, -20, rev=.05)
        place(blip(987.8, .1, 'square', .04), 3.26 + .5 * k, -21, rev=.05)
    place(hit('small', 23), 3.32, -10)
    # 2b  ...silently past Earth — MISSED!
    bed(drone(1.6, 32.7, 24), 4.6, 6.14, -20)
    place(whoosh(1.5, .55, 110, 900, 180, 1.6, .9, -.9, seed=25), 4.62, -15, rev=.4)
    place(hit('mid', 26), 5.33, -3, rev=.3)
    place(braam(49, 1.2, 27), 5.33, -13, rev=.3)

    # 3a  dive into the imagined world
    place(riser(1.46, 200, 7500, True, 30), 6.14, -9, rev=.2, end=7.6)
    place(sparkle(1.2, 25, 31), 6.3, -22)
    # 3b  prehistoric jungle — insects, birds, volcano, T-rex steps
    bed(insects(1.7, 32), 7.6, 9.24, -28)
    scatter(bird, 7.7, 9.1, 3.0, -22, 33, rev=.3)
    bed(rumble(1.7, 34), 7.6, 9.24, -17)
    for k in range(3):
        place(footstep_heavy(35 + k), 7.8 + k * .63, -9, pan=-.25 + .2 * k, rev=.2)
    place(screech(.55, 1300, 36), 8.15, -20, pan=.5, rev=.45)
    place(hit('small', 37), 8.0, -12)

    # 4a  millions of years blur by — accelerating clock + riser
    t, k = 9.24, 0
    while t < 10.28:
        place(tick(3200 if k % 2 else 2400, .04, 400 + k), t, -13, pan=-.3 if k % 2 else .3, rev=.05)
        t += max(.028, .16 * .82 ** k)
        k += 1
    place(riser(1.06, 300, 5000, True, 41), 9.24, -9, end=10.3)
    whir = sine(expsweep(220, 900, 1.06)) * (.55 + .45 * np.sin(phase(np.full(len(tt(1.06)), 6.05))))
    bed(whir, 9.24, 10.3, -24)
    place(hit('small', 42), 9.33, -10)
    # 4b  ice spreads
    place(hit('ice', 43), 10.32, -6, rev=.35)
    bed(crackle(.8, 170, 2500, 9000, ring=True, seed=44), 10.3, 11.1, -12)
    bed(wind(.85, 1800, .9, 45), 10.3, 11.1, -20)
    # 4c  deserts grow
    bed(wind(.9, 900, 1.6, 46, .7), 11.1, 11.96, -10)
    sand = shaped_noise(.9, lambda t: 7000 + 0 * t, lambda t: .6 + 0 * t, lambda t: .6 + .4 * np.sin(t * 9), 146)
    bed(widen(sand), 11.1, 11.96, -17)
    place(hit('small', 47), 11.2, -10)

    # 5a  ADAPT — DNA arpeggio + scramble chatter + lock
    for k in range(16):
        place(blip(pent[(k * 3) % 8] * (2 if k % 2 else 1), .08, 'sine', .025), 11.98 + k * .075, -16, pan=np.sin(k * .9) * .7, rev=.15)
    bed(drone(1.24, 55, 147), 11.96, 13.2, -19)
    place(riser(.68, 400, 6000, False, 148), 11.98, -17, end=12.66)
    scatter(lambda r: blip(r.uniform(1500, 4000), .03, 'square', .008), 12.11, 12.6, 35, -20, 48, rev=.05)
    place(hit('small', 49), 12.66, -9)
    place(blip(1760, .25, 'sine', .08), 12.66, -16, rev=.2)
    # 5b  FEATHERS
    bed(flutter(1.35, 16, 50), 13.25, 14.62, -11)
    place(whoosh(.6, .4, 400, 3000, 800, 1.2, -.5, .5, seed=51), 13.4, -13)
    place(sparkle(1.0, 30, 52), 13.5, -22)
    place(hit('small', 53), 13.45, -10)

    # 6a  SHRINK
    place(zap_down(.68), 14.74, -11, rev=.25)
    place(blip(1568, .3, 'sine', .1), 15.42, -16, rev=.2)
    bed(drone(1.1, 55, 54), 14.62, 15.7, -24)
    # 6b  FROZEN SEAS — underwater, bubbles, whale-like call, aurora shimmer
    bed(widen(norm(lp(pink(1.6, 55), 500))), 15.7, 17.24, -17)
    bed(bubbles(1.55, 22, 56), 15.72, 17.24, -18)
    place(whale(1.3, 190, 57), 15.85, -14, rev=.6)
    place(sparkle(1.4, 12, 58, 3000, 7000), 15.7, -26)
    place(hit('ice', 59), 16.0, -12, rev=.4)

    # 7a  HUMANS ARRIVE — dawn wind, footsteps
    bed(wind(1.1, 600, 1.3, 60, .4), 17.24, 18.3, -15)
    bed(pad(1.06, [98, 146.83, 196], 1200, .2, .3, 160), 17.24, 18.3, -19)
    scatter(step_light, 17.3, 18.25, 9, -17, 61, rev=.1)
    scatter(bird, 17.3, 18.2, 2, -27, 62, rev=.3)
    place(hit('small', 63), 17.46, -10)
    # 7b  TERRIFIED — roar + heartbeat
    place(roar(1.0, 70, 64, .9), 18.38, -1, rev=.35)
    for b in beats(18.3, 19.3, 9.5):
        place(heartbeat(), b - .03, -9, rev=.05)
    bed(rumble(1.0, 65), 18.3, 19.3, -14)
    # 7c  HUNTED — running, raptor screech, lock-on beeps, racing heart
    scatter(step_light, 19.3, 20.2, 14, -18, 66, rev=.05)
    for k in range(3):
        place(footstep_heavy(166 + k), 19.36 + k * .3, -15, pan=.3, rev=.1)
    place(screech(.45, 900, 67), 19.45, -12, pan=.4, rev=.3)
    for k in range(4):
        place(blip(2093, .06, 'square', .02), 19.36 + k * .25, -21)
    for b in (19.32, 19.74, 20.14):
        place(heartbeat(), b, -12)

    # 8a  BY GIANTS — falling foot + stomp
    place(whoosh(.32, .95, 120, 900, 400, 1.3, 0, 0, seed=69), 20.22, -8)
    place(stomp(70), 20.52, 0, rev=.3)
    scatter(step_light, 20.55, 21.25, 12, -20, 71)
    # 8b  BUT SOME COMMUNITIES — warm reveal + campfire
    place(reverse_swell(.5, 72), 21.3, -14)
    bed(pad(1.0, [110, 164.81, 220, 277.18], 1600, .25, .3, 73), 21.3, 22.25, -18)
    bed(crackle(.95, 70, 900, 4500, seed=74), 21.4, 22.25, -20)
    place(hit('small', 75), 21.58, -11)
    # 8c  COEXIST — village, gentle giant call
    bed(crackle(1.15, 70, 900, 4500, seed=76), 22.25, 23.36, -21)
    place(whale(1.0, 95, 77), 22.4, -16, rev=.5)
    place(hit('warm', 78), 22.47, -7, rev=.3)
    scatter(bird, 22.4, 23.3, 2, -27, 79)
    bed(pad(1.15, [110, 138.59, 164.81, 220], 1800, .2, .3, 80), 22.25, 23.36, -19)

    # 9a  GENTLE PLANT EATERS — wonder pad, leaves, chime on the touch
    bed(wind(1.95, 900, 1.0, 81, .3), 23.36, 25.3, -25)
    bed(pad(1.95, [130.81, 196, 261.63, 329.63], 2400, .3, .4, 82), 23.36, 25.3, -14)
    bed(flutter(1.9, 5, 182), 23.36, 25.3, -27)
    place(chime_arp([523.25, 659.25, 783.99, 1046.5, 1318.51], .08), 24.41, -12, rev=.5)
    place(sparkle(1.0, 40, 83), 24.45, -18)
    place(hit('small', 84), 23.81, -14)
    # 9b  COME WITH ME — portal riser with accelerating pulses
    place(riser(1.14, 150, 9000, True, 85), 25.3, -8, rev=.2, end=26.44)
    t, k = 25.32, 0
    while t < 26.4:
        place(sub_boom(.25, 110, 55, .05, .02), t, -15)
        t += max(.05, .14 * .9 ** k)
        k += 1

    # 10a  IT'S MODERN DAY — digital stutter + city
    place(glitch(.35, 87), 26.44, -10)
    city = widen(norm(lp(pink(1.1, 88), 380, 4) + .15 * lp(saw(55, 1.1, 10), 300)))
    bed(city, 26.44, 27.5, -19)
    place(horn(89), 27.0, -26, pan=-.6, rev=.5)
    place(blip(1046.5, .1, 'square', .03), 26.75, -22)
    # 10b  FEATHERED RUNNER — sprinting steps, speed wind
    for k in range(4):
        place(footstep_heavy(190 + k), 27.55 + k * .286, -15, rev=.05)
        place(step_light(rng(290 + k)), 27.69 + k * .286, -19)
    bed(wind(1.05, 1500, 1.4, 91, .2), 27.5, 28.55, -12)
    bed(flutter(1.0, 20, 92), 27.5, 28.55, -22)
    place(hit('small', 93), 27.72, -10)
    # 10c  TO WORK — sign pop, rider pass-by, clock
    place(hit('small', 94), 28.65, -9)
    place(pop(600, 1400, .1), 28.62, -14)
    place(whoosh(.7, .5, 200, 2500, 300, 1.2, -1, 1, seed=95), 28.75, -9)
    place(tick(2600, .05, 96), 28.8, -20)
    place(tick(2000, .05, 97), 29.3, -20)

    # 11a  GIANTS STROLL PAST SKYSCRAPERS
    bed(widen(norm(lp(pink(1.45, 98), 380, 4))), 29.58, 31.0, -20)
    for k, tk in enumerate((29.7, 30.62)):
        place(footstep_heavy(198 + k), tk, -5, rev=.25)
    place(whale(1.2, 70, 99), 30.0, -18, rev=.5)
    place(horn(100), 30.5, -27, pan=.5, rev=.5)
    scatter(step_light, 29.6, 31.0, 6, -30, 101)
    # 11b  HUGE DOORWAYS
    place(door_creak(1.2, 102), 31.05, -14, rev=.4)
    place(hit('small', 103), 31.2, -9)
    for k in range(10):
        place(tick(1500 + k * 150, .03, 500 + k), 31.25 + k * .045, -20)
    place(blip(1760, .2, 'sine', .07), 31.72, -16)
    for tk in (31.4, 32.3):
        place(footstep_heavy(int(tk * 10)), tk, -10, rev=.3)

    # 12a  REINFORCED BRIDGES — blueprint drawing
    bed(scribble(.9, 104), 32.4, 33.3, -12)
    scatter(lambda r: blip(r.choice([880, 1174.66, 1318.5, 1760]), .05, 'square', .015), 32.45, 33.2, 10, -19, 105)
    place(hit('small', 106), 32.55, -9)
    place(ping(1567.98, .5, .12), 32.85, -16)
    for tk in (32.8, 33.5):
        place(footstep_heavy(int(tk * 10)), tk, -14)
    # 12b  BUT PEACE IS FRAGILE — glass crack
    bed(pad(.4, [261.63, 329.63, 392], 2000, .05, .05, 107), 33.7, 34.05, -22)
    place(glass_crack(108), 34.05, -4, rev=.4)
    bed(crackle(.75, 90, 2500, 8000, True, 109), 34.07, 34.82, -12)
    place(sparkle(.8, 40, 110, 3000, 9000), 34.15, -18)
    place(hit('small', 111), 34.25, -10)
    # 12c  HERDS NEED LAND — map pings, herd rumble
    bed(rumble(1.5, 112), 35.0, 36.5, -12)
    for tk in (35.15, 35.75, 36.3):
        place(footstep_heavy(int(tk * 10)), tk, -15, pan=.3, rev=.3)
    place(ping(1318.51, .4), 35.2, -16)
    place(ping(987.77, .4), 35.3, -16)
    scatter(lambda r: blip(r.uniform(1800, 2600), .04, 'sine', .012), 35.1, 36.4, 6, -22, 113)
    place(hit('small', 114), 35.25, -10)
    # 12d  PREDATORS NEED FOOD — growl + heartbeat
    for b in beats(36.5, 38.06, 9.0):
        place(heartbeat(), b - .03, -10, rev=.05)
    place(roar(1.3, 52, 115, .7), 36.75, -9, rev=.3)
    place(hit('small', 116), 36.6, -9)
    bed(drone(1.56, 34.6, 117), 36.5, 38.06, -16)

    # 13a  ONE BORDER DISPUTE — siren + tension riser + warning beeps
    bed(siren(1.64, 1.6), 38.06, 39.7, -17)
    place(riser(1.64, 100, 3000, True, 118), 38.06, -9, end=39.7)
    bed(drone(1.64, 36.7, 161), 38.06, 39.7, -16)
    for b in beats(38.06, 39.7, 10):
        place(blip(988, .07, 'square', .02), b, -15)
    place(hit('small', 119), 38.16, -10)
    # 13b  COULD SHATTER EVERYTHING
    place(shatter(120), 39.78, -1, rev=.35)
    place(hit('big', 121), 39.78, -5, rev=.3)
    place(hit('mid', 122), 40.2, -8)
    place(hit('small', 123), 40.4, -10)
    # 13c  ONE MISSED ASTEROID — tape rewind + pass-by
    place(rewind(.6, 124), 40.92, -11)
    place(whoosh(1.3, .5, 200, 1800, 300, 1.3, -.9, .9, seed=125), 40.95, -12)
    bed(crackle(1.3, 50, 800, 4000, seed=126), 40.95, 42.3, -22)
    place(hit('small', 127), 41.15, -10)
    # 13d  OUR WORLD — tiny planet wonder
    bed(pad(1.56, [146.83, 220, 293.66, 369.99], 2600, .2, .4, 128), 42.3, 43.86, -14)
    place(pop(300, 900, .15), 42.35, -15)
    place(sparkle(1.4, 25, 129), 42.4, -21)
    place(hit('small', 130), 42.65, -10)

    # 14a  WHAT'S YOUR WILDEST WHAT IF?
    place(pop(400, 1200, .14), 44.0, -12)
    place(bell(880, 1.2), 44.05, -18, rev=.4)
    place(whoosh(.6, .5, 300, 2200, 400, 1.1, -.8, .8, seed=131), 44.35, -18)
    place(whoosh(.6, .5, 300, 2200, 400, 1.1, .8, -.8, seed=132), 45.25, -18)
    bed(pad(2.04, [220, 277.18, 329.63, 415.3], 2000, .3, .4, 133), 43.86, 45.9, -16)
    place(hit('mid', 134), 44.26, -7)
    # 14b  DROP IT IN THE COMMENTS — card pop, typing, hearts, bouncing arrow
    place(pop(700, 1500, .1), 45.95, -12)
    for k, ts in enumerate((46.15, 46.57, 46.98)):
        place(typing(6, 30, 600 + k), ts, -11)
    scatter(lambda r: pop(r.uniform(900, 1400), r.uniform(1800, 2600), .06), 46.0, 47.3, 4, -18, 135)
    bed(pad(1.5, [196, 246.94, 293.66], 2000, .15, .2, 162), 45.9, 47.4, -17)
    for tb in np.arange(np.ceil(45.9 * 7 / np.pi), np.floor(47.4 * 7 / np.pi) + 1) * np.pi / 7:
        place(blip(196, .1, 'sine', .04), tb, -16)
    place(hit('small', 136), 46.15, -10)
    # 14c  YOUR IDEA → OUR NEXT VIDEO — idea ding, morph, confetti, stinger
    place(bell(1318.51, 1.6), 47.45, -11, rev=.4)
    place(sparkle(.6, 30, 137), 47.45, -20)
    place(whoosh(.45, .7, 300, 4000, 900, 1.0, -.5, .5, seed=138), 48.1, -12)
    place(pop(500, 1300, .12), 48.4, -12)
    place(confetti(139), 48.45, -8, rev=.3)
    place(hit('mid', 140), 48.5, -8)
    place(stinger(141), 48.5, -7, rev=.45)
    bed(pad(2.2, [220, 277.18, 329.63, 440], 2400, .3, .5, 142), 47.4, 49.6, -21)


# ----------------------------------------------------------------------------
# master
# ----------------------------------------------------------------------------
def k_weight(x):
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=-1), axis=-1)


def lufs(x):
    y = k_weight(x)
    blk, hop = int(.4 * SR), int(.1 * SR)
    ms = np.array([np.sum(np.mean(y[:, i:i + blk] ** 2, axis=1)) for i in range(0, y.shape[1] - blk, hop)])
    ld = -0.691 + 10 * np.log10(ms + 1e-12)
    g = ms[ld > -70]
    rel = -0.691 + 10 * np.log10(np.mean(g)) - 10
    g = ms[(ld > -70) & (ld > rel)]
    return -0.691 + 10 * np.log10(np.mean(g))


def glue(x, thr_db=-22.0, ratio=2.5, win=.04, smooth=.08):
    """Gentle RMS compressor that narrows the gap between hits and ambience."""
    from scipy.ndimage import uniform_filter1d
    lvl = 10 * np.log10(uniform_filter1d(np.mean(x ** 2, axis=0), int(win * SR)) + 1e-12)
    gr = np.maximum(0, lvl - thr_db) * (1 - 1 / ratio)
    a = np.exp(-1 / (smooth * SR))
    gr = signal.lfilter([1 - a], [1, -a], gr)
    return x * db(-gr)


def limiter(x, ceil=db(-2.0), look=.003, rel=.12):
    L = int(look * SR)
    need = np.minimum(1.0, ceil / np.maximum(maximum_filter1d(np.max(np.abs(x), axis=0), 2 * L + 1), 1e-12))
    k = np.exp(-1 / (rel * SR))
    g = np.empty_like(need)
    cur = 1.0
    for i, v in enumerate(need):
        cur = v if v < cur else v + (cur - v) * k
        g[i] = cur
    g = minimum_filter1d(g, 2 * L + 1)
    g = np.convolve(g, np.ones(2 * L + 1) / (2 * L + 1), mode='same')
    return np.clip(x * g, -ceil, ceil)


def write_wav(path, x):
    pcm = np.clip(np.round(x.T * (2 ** 23 - 1)), -2 ** 23, 2 ** 23 - 1).astype('<i4')
    raw = pcm.reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(SR)
        w.writeframes(raw)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'what-if-dinosaurs-sfx.wav'
    transitions()
    cues()
    wet = signal.fftconvolve(VERB, ir_reverb(1.7), axes=1)[:, :N]
    mix = hp(MIX + .9 * wet, 28, 2)
    mix = glue(mix)
    mix *= db(-19.0 - lufs(mix))                 # SFX bed sits under a -14 LUFS voice-over
    mix = limiter(mix)
    mix = fade(mix, .004, .3)
    write_wav(out, mix)
    print(f'wrote {out}: {mix.shape[1] / SR:.3f}s, {SR} Hz, 24-bit stereo, '
          f'{lufs(mix):.1f} LUFS, peak {20 * np.log10(np.max(np.abs(mix))):.2f} dBFS')


if __name__ == '__main__':
    main()
