#!/usr/bin/env python3
"""
Procedural sound design for the Short "What If Every Human Disappeared?"
Builds what-if-humans-disappeared-sfx.wav  (46.48 s, 48 kHz, 24-bit stereo)
synced to the shot list / events of what-if-humans-disappeared.html.

Everything is synthesized from noise + oscillators (no samples), so it is
royalty-free. Re-run to regenerate:  python3 what-if-humans-disappeared-sfx.py
Requires: numpy, scipy.
"""
import os, wave
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d

SR = 48000
DUR = 46.48
N = int(round(DUR * SR))
R = np.random.default_rng(2026)
FX = np.zeros((2, N))     # hits, whooshes, foley
BED = np.zeros((2, N))    # ambiences / drones (ducked under the voice-over)
SEND = np.zeros((2, N))   # reverb send

# ----------------------------------------------------------------- basics
def S(t): return int(round(t * SR))
def ns(d): return max(1, int(round(d * SR)))
def tax(n): return np.arange(n) / SR
def wn(n): return R.standard_normal(n)
def norm(x):
    m = np.max(np.abs(x)) if x.size else 0
    return x / m if m > 1e-12 else x

def _sos(kind, f, q=2.0, order=2):
    ny = SR / 2
    if kind == 'lp':
        return signal.butter(order, min(max(f, 20), ny * .98) / ny, 'low', output='sos')
    if kind == 'hp':
        return signal.butter(order, min(max(f, 10), ny * .98) / ny, 'high', output='sos')
    bw = f / q
    lo, hi = max(15, f - bw / 2), min(ny * .97, f + bw / 2)
    if hi <= lo * 1.01: hi = lo * 1.02
    return signal.butter(max(1, order // 2), [lo / ny, hi / ny], 'band', output='sos')

def lp(x, f, o=2): return signal.sosfilt(_sos('lp', f, order=o), x)
def hp(x, f, o=2): return signal.sosfilt(_sos('hp', f, order=o), x)
def bp(x, f, q=2, o=2): return signal.sosfilt(_sos('bp', f, q, o), x)

def sweep(x, ffun, kind='bp', q=2.0, blk=256):
    """time-varying filter; ffun(k) -> cutoff for normalized time k in [0,1]"""
    n = len(x); y = np.empty(n); zi = None; nb = (n + blk - 1) // blk
    for b in range(nb):
        sos = _sos(kind, float(ffun(b / max(1, nb - 1))), q)
        if zi is None: zi = np.zeros((sos.shape[0], 2))
        seg = x[b * blk:(b + 1) * blk]
        y[b * blk:b * blk + len(seg)], zi = signal.sosfilt(sos, seg, zi=zi)
    return y

def expf(f0, f1, pw=1.0): return lambda k: f0 * (f1 / f0) ** (k ** pw)

def pink(n):
    X = np.fft.rfft(wn(n)); f = np.fft.rfftfreq(n, 1 / SR); f[0] = f[1]
    y = np.fft.irfft(X / np.sqrt(f), n); return y / (np.std(y) + 1e-12)

def brown(n):
    y = hp(np.cumsum(wn(n)), 18); return y / (np.std(y) + 1e-12)

def smooth_rand(n, rate, lo=-1.0, hi=1.0):
    pts = int(n / SR * rate) + 3
    v = R.uniform(lo, hi, pts)
    return np.interp(np.linspace(0, pts - 2, n), np.arange(pts), v)

def phase(f): return 2 * np.pi * np.cumsum(f) / SR
def farr(f, n): return np.full(n, float(f)) if np.isscalar(f) else np.asarray(f, float)
def sine(f, n): return np.sin(phase(farr(f, n)))
def saw(f, n): return 2 * ((np.cumsum(farr(f, n)) / SR) % 1.0) - 1
def sqr(f, n): return np.tanh(6 * np.sin(phase(farr(f, n))))

def expdec(n, tau, att=0.002):
    e = np.exp(-tax(n) / tau); na = min(n, max(1, int(att * SR)))
    e[:na] *= np.linspace(0, 1, na); return e

def hump(n, peak=.7, a=2.0, b=2.0):
    k = np.linspace(0, 1, n)
    return np.where(k < peak, (k / max(peak, 1e-6)) ** a, ((1 - k) / max(1 - peak, 1e-6)) ** b)

def fade(x, fi=.005, fo=.02):
    x = x.copy(); n = x.shape[-1]
    a, b = min(n, ns(fi)), min(n, ns(fo))
    x[..., :a] *= np.linspace(0, 1, a); x[..., n - b:] *= np.linspace(1, 0, b); return x

def pan2(x, pan=0.0):
    if x.ndim == 2: return x
    a = (np.clip(np.asarray(pan, float), -1, 1) + 1) * np.pi / 4
    return np.vstack([x * np.cos(a), x * np.sin(a)])

def place(x, t, g=1.0, pan=0.0, bus='fx', verb=0.0):
    x = pan2(np.asarray(x, float), pan) * g
    i = S(t); j0 = max(0, -i); i0 = max(0, i); n = min(x.shape[1] - j0, N - i0)
    if n <= 0: return
    tgt = FX if bus == 'fx' else BED
    tgt[:, i0:i0 + n] += x[:, j0:j0 + n]
    if verb: SEND[:, i0:i0 + n] += x[:, j0:j0 + n] * verb

def db(v): return 10 ** (v / 20)

# ------------------------------------------------------------ generators
def whoosh(d=.6, f0=300, f1=3000, q=1.6, peak=.65, p0=-.6, p1=.6):
    n = ns(d)
    L = sweep(pink(n), expf(f0, f1), 'bp', q); Rr = sweep(pink(n), expf(f0 * 1.1, f1 * 1.1), 'bp', q)
    e = hump(n, peak, 2.2, 1.6); x = np.vstack([L * e, Rr * e]); x = norm(x)
    pan = np.linspace(p0, p1, n); a = (pan + 1) * np.pi / 4
    return np.vstack([x[0] * np.cos(a) * 1.41, x[1] * np.sin(a) * 1.41])

def whip(d=.34, p0=-.9, p1=.9):
    return whoosh(d, 500, 6500, 1.1, .55, p0, p1) + .6 * whoosh(d, 120, 900, 1.2, .5, p0, p1)

def rev_swell(d=.45, f0=1500, f1=9000):
    n = ns(d); x = sweep(wn(n), expf(f0, f1), 'hp', 1) * np.linspace(0, 1, n) ** 3
    y = sweep(wn(n), expf(f0, f1), 'hp', 1) * np.linspace(0, 1, n) ** 3
    return norm(np.vstack([x, y]))

def impact(size=1.0, f0=95, f1=32, bright=3500):
    d = 1.4 * size + .3; n = ns(d); t = tax(n)
    f = f1 + (f0 - f1) * np.exp(-t / .07)
    sub = np.sin(phase(f)) * expdec(n, .32 * size + .05, .0008)
    body = lp(wn(n), bright) * expdec(n, .05 + .03 * size, .0004)
    click = hp(wn(n), 2500) * expdec(n, .005, .0001)
    return norm(np.tanh(1.8 * (sub + .55 * body + .45 * click)))

def riser(d, f0=200, f1=4000, tone=True, curve=2.4):
    n = ns(d); k = np.linspace(0, 1, n); e = .08 + .92 * k ** curve
    chans = []
    for det in (1.0, 1.012):
        nz = sweep(wn(n), expf(f0 * det, f1 * det, 1.5), 'bp', 2.2)
        x = nz / (np.std(nz) + 1e-9) * .5
        if tone:
            ff = f0 * .5 * det * (f1 / f0) ** (k ** 1.5)
            x += lp((saw(ff, n) + saw(ff * 1.505, n) * .6 + saw(ff * .5, n) * .8) / 2.4, 7000) * .7
        chans.append(x * e)
    return norm(np.vstack(chans))

def glitch(d=.24):
    out = []
    for ch in range(2):
        n = ns(d); y = np.zeros(n); i = 0
        while i < n:
            g = ns(R.uniform(.01, .045)); typ = R.integers(5)
            if typ == 0: seg = sqr(R.uniform(180, 2400), g) * .6
            elif typ == 1: seg = np.round(wn(g) * 3) / 3 * .45
            elif typ == 2: seg = np.zeros(g)
            elif typ == 3: seg = np.repeat(saw(R.uniform(50, 300), g)[::10], 10)[:g] * .8
            else: seg = np.sin(phase(np.linspace(R.uniform(2000, 6000), 200, g))) * .7
            m = min(g, n - i); y[i:i + m] = fade(seg[:m], .001, .002); i += g
        out.append(hp(y, 90))
    x = np.vstack(out) * hump(ns(d), .08, 1, 1.4); return norm(x)

def blip(f=1200, d=.09, h2=.25, dec=None):
    n = ns(d); x = sine(f, n) + h2 * sine(2 * f, n)
    return norm(x * expdec(n, dec or d / 3, .002))

def tick(f=3500, d=.012, q=1.5):
    n = ns(d); return norm(bp(wn(n), f, q) * expdec(n, .0018, .0002))

def ping(f, dec=.08, d=None):
    n = ns(d or dec * 5); return sine(f, n) * expdec(n, dec, .0008)

def sparkle(d, dens=60, fmin=3000, fmax=10000, dec=.05):
    n = ns(d); x = np.zeros((2, n))
    for _ in range(int(d * dens)):
        p = ping(R.uniform(fmin, fmax), dec * R.uniform(.5, 1.5)); i = R.integers(0, max(1, n - p.size))
        a = R.uniform(-1, 1); m = min(p.size, n - i)
        x[:, i:i + m] += pan2(p[:m], a) * R.uniform(.3, 1)
    return norm(x)

def crowd(d):
    n = ns(d); x = np.zeros(n)
    for f in (320, 520, 760, 1100, 1600, 2300):
        x += bp(pink(n), f, 3) * (smooth_rand(n, R.uniform(3, 7), .1, 1) ** 2)
    return norm(hp(x, 150))

def squeal(d=.7, f=3100):
    n = ns(d); t = tax(n)
    ff = f + 120 * np.sin(2 * np.pi * 7.5 * t) + smooth_rand(n, 12, -60, 60)
    x = sine(ff, n) + .3 * sine(ff * 2.01, n) + .25 * bp(wn(n), f, 4)
    return norm(x * hump(n, .2, 1.5, 1.2))

def skid(d=.6):
    n = ns(d); x = bp(wn(n), 1800, 1.2) * (.6 + .4 * smooth_rand(n, 30, 0, 1)); return norm(x * hump(n, .15, 1, 1.5))

def servo(d, rate=.67, base=520, ph=0.0):
    n = ns(d); t = tax(n); v = np.abs(np.cos(2 * np.pi * rate * t + ph))
    f = base + 380 * v
    x = .5 * saw(f, n) + .3 * sine(f * 2.01, n) + .2 * saw(f * 3.02, n)
    x = bp(x, 1400, 1.0) * (.25 + .75 * v)
    return norm(x)

def weld(d):
    n = ns(d); imp = np.zeros(n)
    k = R.integers(0, n, int(d * 420)); imp[k] = R.uniform(-1, 1, k.size) ** 3
    crack = bp(imp, 4200, .8) * 3
    buzz = bp(np.tanh(4 * sine(100, n)), 1500, 1.2) * np.clip(smooth_rand(n, 35, -.2, 1), 0, 1)
    siz = hp(wn(n), 6000) * np.clip(smooth_rand(n, 25, -.5, 1), 0, 1) * .5
    return norm(crack + buzz + siz)

def clank(f0=520, dec=.22):
    n = ns(dec * 4); x = np.zeros(n)
    for r, a, dd in ((1, 1, 1), (2.59, .7, .6), (4.29, .5, .4), (5.96, .35, .3), (8.1, .2, .2)):
        x += a * sine(f0 * r, n) * expdec(n, dec * dd, .0005)
    x += .5 * bp(wn(n), 2500, 1) * expdec(n, .006, .0002)
    return norm(x)

def beep(f=880, d=.12, wave_='sqr'):
    n = ns(d); x = (sqr(f, n) if wave_ == 'sqr' else sine(f, n)); x = lp(x, 3500)
    return norm(fade(x, .004, .02))

def hum(d, f=60, harm=10, buzz=.3):
    n = ns(d); x = np.zeros(n)
    for h in range(1, harm + 1): x += sine(f * h, n) / h ** 1.1
    x += buzz * bp(np.tanh(5 * sine(f * 2, n)), 900, 1)
    return norm(x)

def zap(d=.4):
    n = ns(d); t = tax(n)
    g = (sine(120, n) > 0).astype(float)
    x = bp(wn(n) * g, 3500, .7) * 1.4
    imp = np.zeros(n); k = R.integers(0, n, int(d * 300)); imp[k] = R.uniform(-1, 1, k.size)
    x += bp(imp, 5000, .6) * 4
    x += .6 * np.sin(phase(2800 * np.exp(-t / .07) + 180))
    return norm(np.tanh(2 * x) * expdec(n, .12, .001))

def klaxon(d=.62, f1=640, f2=880, rate=6):
    n = ns(d); t = tax(n); f = np.where(np.floor(t * rate) % 2 == 0, f1, f2)
    x = lp(saw(f, n) + .6 * saw(f * 1.006, n), 2600)
    return norm(fade(x, .01, .06))

def power_down(d=1.3, f0=420, f1=22):
    n = ns(d); k = np.linspace(0, 1, n); f = f0 * (f1 / f0) ** (k ** .7)
    x = saw(f, n) + .5 * saw(f * 1.5, n) + .8 * sine(f * .5, n)
    x = sweep(x, expf(5000, 140, .6), 'lp', 1)
    return norm(x * (1 - k) ** .6)

def fan(d, r0=55, r1=3, f0=190, f1=18):
    n = ns(d); k = np.linspace(0, 1, n)
    rate = r0 * (r1 / r0) ** (k ** .8); ff = f0 * (f1 / f0) ** (k ** .8)
    trem = 1 + .55 * np.sin(phase(rate))
    x = bp(pink(n), 900, .6) * trem * (1 - k) ** .5 + .35 * sine(ff, n) * (1 - k) ** .4
    return norm(x)

def typing(d, rate=17):
    n = ns(d); x = np.zeros(n); t = R.exponential(1 / rate)
    while t < d - .02:
        m2 = ns(.02); c = bp(wn(m2), R.uniform(2500, 4800), 2) * expdec(m2, .0025, .0002)
        c += .6 * bp(wn(m2), 700, 1.5) * expdec(m2, .004, .0003)
        i = S(t); m = min(c.size, n - i); x[i:i + m] += c[:m] * R.uniform(.5, 1); t += R.exponential(1 / rate) + .02
    return norm(x)

def crt_off():
    n = ns(.35); t = tax(n)
    x = np.sin(phase(2600 * np.exp(-t / .05) + 90)) * expdec(n, .09, .001)
    x += lp(wn(n), 900) * expdec(n, .025, .0005) * .8
    return norm(x)

def wind(d, lo=250, hi=900, rate=.4):
    n = ns(d); out = []
    for _ in range(2):
        cut = smooth_rand(n + 300, rate, 0, 1)
        y = sweep(pink(n), lambda k, c=cut: lo + (hi - lo) * c[int(k * (n - 1))], 'bp', .8, blk=512)
        out.append(y * (.5 + .5 * smooth_rand(n, rate * 1.5, 0, 1)))
    return norm(np.vstack(out))

def thump(f0=110, f1=55, d=.3, nz=.3):
    n = ns(d); t = tax(n); f = f1 + (f0 - f1) * np.exp(-t / .03)
    x = np.sin(phase(f)) * expdec(n, d / 4, .001) + nz * lp(wn(n), 600) * expdec(n, .02, .0005)
    return norm(x)

def chirp(f0=3600, f1=5600, syl=3, sd=.06, gap=.035):
    n = ns(syl * (sd + gap) + .05); x = np.zeros(n)
    for s in range(syl):
        m = ns(sd); k = np.linspace(0, 1, m)
        f = f0 + (f1 - f0) * np.sin(k * np.pi) + 250 * np.sin(2 * np.pi * 38 * k * sd)
        y = np.sin(phase(f)) * np.sin(np.pi * k) ** 1.5; i = S(s * (sd + gap)); x[i:i + m] += y
    return norm(x)

def birds(d, dens=2.2, gain=1.0):
    n = ns(d); x = np.zeros((2, n))
    for _ in range(int(d * dens)):
        c = chirp(R.uniform(2600, 4200), R.uniform(4300, 7000), int(R.integers(2, 6)), R.uniform(.04, .08), R.uniform(.02, .06))
        i = R.integers(0, max(1, n - c.size)); m = min(c.size, n - i)
        x[:, i:i + m] += pan2(c[:m], R.uniform(-.9, .9)) * R.uniform(.25, 1)
    return x * gain

def clock_tick(hi=True):
    n = ns(.05); x = bp(wn(n), 2400 if hi else 1700, 7) * expdec(n, .006, .0003)
    x[:ns(.012)] += .3 * tick(5000)
    return norm(x)

def page_flip():
    n = ns(.09); t = tax(n)
    x = bp(wn(n), 3800, .9) * hump(n, .25, 1, 1.5) * (1 + .7 * np.sin(2 * np.pi * 70 * t))
    return norm(x)

def crumble(d=.8, dens=55):
    n = ns(d); x = np.zeros(n)
    for _ in range(int(d * dens)):
        u = R.uniform(0, 1) ** 1.8; g = lp(wn(ns(.02)), R.uniform(500, 2500)) * expdec(ns(.02), .004)
        i = S(u * d * .95); m = min(g.size, n - i); x[i:i + m] += g[:m] * (1 - u) * R.uniform(.3, 1)
    return norm(x)

def crack():
    n = ns(1.4); x = np.zeros(n)
    x[:ns(.04)] += hp(wn(ns(.04)), 1500) * expdec(ns(.04), .004, .0001)
    for f, a in ((180, .7), (410, .5), (760, .4), (1330, .3)):
        x += a * ping(f, .07, 1.4)[:n]
    x[:ns(.5)] += thump(90, 40, .5, .5) * .9
    cr = np.pad(crumble(1.1), (ns(.05), 0)); x[:min(n, cr.size)] += cr[:n] * .5
    return norm(np.tanh(1.5 * x))

def rustle(d, f=4800):
    n = ns(d); out = []
    for _ in range(2):
        out.append(bp(wn(n), f, 1.1) * np.clip(smooth_rand(n, 14, -.4, 1), 0, 1) ** 1.5)
    return norm(np.vstack(out))

def shimmer(d, base=440, up=2.0, chord=(1, 1.25, 1.5, 2, 2.5)):
    n = ns(d); k = np.linspace(0, 1, n); x = np.zeros((2, n))
    for i, r in enumerate(chord):
        f = base * r * up ** k
        y = sine(f, n) * (.6 + .4 * np.sin(2 * np.pi * (6 + i) * tax(n)))
        x += pan2(y, (i / (len(chord) - 1)) * 1.4 - .7)
    return norm(x * hump(n, .6, 1.5, 1.2))

def creak(d=.6, f0=520, rate0=35, rate1=70):
    n = ns(d); k = np.linspace(0, 1, n)
    rate = rate0 + (rate1 - rate0) * (np.sin(np.pi * k) + smooth_rand(n, 5, -.3, .3))
    ph = np.cumsum(rate) / SR; imp = np.diff(np.floor(ph), prepend=0) * R.uniform(.4, 1, n)
    x = bp(imp, f0, 14) + .7 * bp(imp, f0 * 2.31, 12) + .4 * bp(imp, f0 * 3.7, 10)
    return norm(x * hump(n, .5, 1, 1))

def metal_groan(d=1.2, f=58):
    n = ns(d); ff = f + 9 * smooth_rand(n, 2, -1, 1)
    src = saw(ff, n) + .4 * pink(n) * .2
    x = bp(src, 330, 18) + .8 * bp(src, 780, 20) + .5 * bp(src, 1420, 22) + .3 * lp(src, 200)
    return norm(x * hump(n, .45, 1.4, 1.2) * (.6 + .4 * smooth_rand(n, 3, 0, 1)))

def cable_snap():
    n = ns(1.0); t = tax(n); f = 160 + 1100 * np.exp(-t / .12)
    x = np.zeros(n)
    for h, a in ((1, 1), (2.02, .5), (3.05, .3), (4.1, .2)): x += a * np.sin(phase(f * h)) * expdec(n, .25 / h, .0005)
    x += hp(wn(n), 5000) * expdec(n, .02, .0002) * .7
    return norm(x)

def splash(d=1.6):
    n = ns(d); x = sweep(wn(n), expf(7000, 900), 'lp', 1) * expdec(n, .35, .008)
    x += lp(brown(n), 300) * expdec(n, .25, .005) * .8
    for _ in range(45):
        m = ns(R.uniform(.02, .06)); f0 = R.uniform(300, 700)
        b = np.sin(phase(np.linspace(f0, f0 * R.uniform(1.6, 2.6), m))) * np.hanning(m)
        i = S(R.uniform(.05, 1.3)); mm = min(m, n - i); x[i:i + mm] += b[:mm] * R.uniform(.1, .35)
    return norm(x)

def wingflaps(d=.7, rate=9):
    n = ns(d); x = np.zeros(n); t = 0
    while t < d - .05:
        g = ns(.05); y = lp(wn(g), 1400) * hump(g, .3, 1, 2); i = S(t); m = min(g, n - i); x[i:i + m] += y[:m]
        t += 1 / rate * R.uniform(.8, 1.2)
    return norm(x * np.linspace(1, .3, n))

def pad(d, freqs, bright=1400, att=.6, rel=.8, det=.005):
    n = ns(d); out = np.zeros((2, n))
    for ch, s in enumerate((1, -1)):
        y = np.zeros(n)
        for f in freqs: y += saw(f * (1 + s * det), n) + .7 * saw(f * (1 - s * det * .6), n)
        out[ch] = lp(y, bright, 2)
    e = np.minimum(1, tax(n) / att) * np.minimum(1, (d - tax(n)) / rel)
    return norm(out * np.clip(e, 0, 1))

def choir(d, freqs, att=.8, rel=.6):
    n = ns(d); out = np.zeros((2, n))
    for ch, s in enumerate((1, -1)):
        y = np.zeros(n)
        for f in freqs: y += saw(f * (1 + s * .004) * (1 + .003 * np.sin(2 * np.pi * 5 * tax(n))), n)
        out[ch] = bp(y, 750, 5) + .8 * bp(y, 1150, 6) + .4 * bp(y, 2700, 8)
    e = np.minimum(1, tax(n) / att) * np.minimum(1, (d - tax(n)) / rel)
    return norm(out * np.clip(e, 0, 1))

def drone(d, f=36.7, bright=320):
    n = ns(d); out = np.zeros((2, n))
    for ch, s in enumerate((1, -1)):
        y = sine(f, n) + .6 * sine(f * 1.5 * (1 + s * .003), n) + .5 * lp(saw(f * 2 * (1 + s * .004), n), bright)
        out[ch] = y * (.8 + .2 * np.sin(2 * np.pi * .3 * tax(n) + ch))
    return norm(out)

def ui_pop(f0=420, f1=1150, d=.11):
    n = ns(d); x = np.sin(phase(np.geomspace(f0, f1, n))) * expdec(n, .035, .001)
    x[:ns(.012)] += .35 * tick(4000)[:ns(.012)]
    return norm(x)

def bubble():
    n = ns(.07); x = np.sin(phase(np.geomspace(260, 1400, n))) * hump(n, .15, 1, 1.4); return norm(x)

def ding(f=1320, d=.9):
    n = ns(d); x = sine(f, n) * expdec(n, .35) + .5 * sine(f * 2, n) * expdec(n, .18) + .25 * sine(f * 3.01, n) * expdec(n, .09)
    return norm(x)

def scan(d=1.5, f0=500, f1=1800):
    n = ns(d); k = np.linspace(0, 1, n)
    f = f0 * (f1 / f0) ** k
    x = sine(f, n) * (.55 + .45 * np.sin(2 * np.pi * 16 * tax(n))) + .3 * bp(wn(n), 3000, 2) * (k * (1 - k) * 4)
    return norm(x * hump(n, .5, .6, .6))

def heartbeat_blip(): return blip(1000, .14, .1, .05)

def flatline(d=1.0):
    n = ns(d); x = sine(1000, n) + .08 * sine(2000, n); return norm(fade(x, .01, .25))

# ======================================================================
#                               TIMELINE
# ======================================================================
def ticks(t0, t1, r0, r1, f=4200, g=-22, pan_alt=.3, q=1.5):
    t = t0; k = 0
    while t < t1:
        u = (t - t0) / (t1 - t0); place(tick(f, .012, q), t, db(g), pan_alt * (1 if k % 2 else -1)); k += 1
        t += 1 / (r0 + (r1 - r0) * u)

def thumps(t0, t1, period, g=-22, f0=95, f1=42):
    t = t0
    while t < t1 - .01: place(thump(f0, f1, .32, .15), t, db(g)); t += period

# ---- transitions at every cut (types mirror the SHOTS table in the HTML)
CUTS = [(0, 'cut'), (1.25, 'flash'), (2.44, 'glitch'), (3.85, 'whip'), (4.75, 'glitch'), (5.54, 'flash'), (6.75, 'zoom'),
        (7.96, 'glitch'), (11.04, 'whip'), (12.75, 'dip'), (14.18, 'flash'), (15.65, 'whip'), (16.88, 'zoom'), (18.85, 'flash'),
        (20.04, 'whip'), (21.45, 'zoom'), (22.78, 'flash'), (25.0, 'glitch'), (26.75, 'whip'), (28.48, 'flash'), (30.4, 'zoom'),
        (33.28, 'glitch'), (36.18, 'flash'), (38.0, 'whip'), (40.1, 'zoom'), (42.22, 'flash')]
for i, (t, tr) in enumerate(CUTS):
    if tr == 'flash':
        place(rev_swell(.4), t - .4, db(-17), verb=.15); place(impact(.45, 150, 48, 6000), t, db(-11), verb=.25)
        place(sparkle(.3, 90, 5000, 12000, .03), t, db(-23))
    elif tr == 'glitch':
        place(glitch(.24), t - .02, db(-13))
    elif tr == 'whip':
        d = 1 if i % 2 else -1; place(whip(.36, .9 * d, -.9 * d), t - .16, db(-9), verb=.1)
    elif tr == 'zoom':
        place(whoosh(.5, 120, 1400, 1.3, .85, 0, 0), t - .42, db(-13)); place(impact(.6, 75, 30, 1500), t, db(-10), verb=.2)
    elif tr == 'dip':
        place(rev_swell(.6, 400, 3000), t - .6, db(-21), verb=.3)

# ---- tension pulse, ACT I/II (heartbeat of the dying city)
thumps(2.44, 7.96, .5, -21)
thumps(7.96, 10.6, .375, -19)

# ---- 1. HOOK: crowd vanishes, population counter crashes to zero
place(impact(1.0), 0.0, db(-4), verb=.3); place(glitch(.18), 0.0, db(-15))
place(drone(2.6, 36.7), 0.0, db(-17), bus='bed')
c = crowd(1.6); c *= 1 - np.clip((tax(c.size) - .25) / 1.05, 0, 1) ** .8; place(c, 0.0, db(-19), bus='bed')
sp = sparkle(1.3, 140, 4000, 11000, .04); sp *= np.linspace(.3, 1, sp.shape[1]); place(sp, .12, db(-17))
ticks(.12, 2.3, 10, 42, 4200, -21)
# 2. single figure disintegrates (Thanos-style ash drifting right)
place(whoosh(1.15, 900, 7000, 1.4, .35, 0, .85), 1.3, db(-14), verb=.2)
place(sparkle(1.1, 240, 2500, 9000, .025), 1.3, db(-16))
place(impact(1.15, 110, 30), 2.3, db(-3), verb=.4); place(glitch(.2), 2.3, db(-14))
place(beep(220, .28) + .6 * beep(110, .28), 2.32, db(-16))

# ---- 3. traffic stops
tr_ = lp(brown(ns(1.5)), 320) * np.linspace(1, .15, ns(1.5)); place(tr_, 2.44, db(-17), bus='bed')
en = lp(saw(np.geomspace(95, 38, ns(1.1)), ns(1.1)), 450) * np.linspace(1, 0, ns(1.1)); place(en, 2.46, db(-22), -.2)
for t, pn, g in ((2.52, -.45, -19), (2.68, .5, -21), (2.86, .1, -20)): place(squeal(.7, R.uniform(2900, 3400)), t, db(g), pn, verb=.15)
place(skid(.6), 2.5, db(-21), -.3); place(skid(.55), 2.72, db(-23), .4)
hn = ns(.85); horn = lp(saw(415, hn) + .8 * saw(523, hn), 1800) * hump(hn, .08, 1, .3); place(horn, 3.0, db(-25), -.6, verb=.25)
for k in range(4): place(tick(1700, .02, 4) + .6 * np.pad(tick(2600, .012, 4), (0, ns(.008))), 3.17 + k * .2, db(-18), .35)

# ---- 4. robots keep working, no operator
place(servo(.9, .67, 520, 0), 3.85, db(-19), -.5); place(servo(.9, .67, 650, 1), 3.85, db(-19), .5)
place(weld(.9), 3.85, db(-17), -.35); place(weld(.9), 3.85, db(-19), .35)
place(hum(.9, 50, 8), 3.85, db(-25), bus='bed')
for t, f, pn in ((3.95, 520, -.3), (4.25, 760, .4), (4.55, 600, -.1)): place(clank(f), t, db(-18), pn, verb=.2)
for t in (3.85, 4.183, 4.517): place(beep(1250, .15), t, db(-19))

# ---- 5. empty control room
place(hum(.8, 60, 6, .1), 4.75, db(-27), bus='bed'); place(bp(pink(ns(.8)), 1500, .5), 4.75, db(-31), bus='bed')
for _ in range(9): place(blip(R.uniform(1600, 3400), .05), R.uniform(4.78, 5.45), db(-25), R.uniform(-.7, .7))
for t in (4.75, 5.15): place(beep(660, .14, 'sine'), t, db(-17)); place(beep(990, .12, 'sine'), t + .1, db(-19))
place(creak(.6, 620, 30, 55), 4.86, db(-21), .1)

# ---- 6. the city goes silent
n6 = ns(1.4); city = lp(brown(n6), 260) + .5 * bp(pink(n6), 900, .7) + .35 * crowd(1.4)
city = sweep(city, expf(7000, 140), 'lp', 1) * (1 - np.linspace(0, 1, n6)) ** 1.4
place(np.vstack([city, np.roll(city, 900)]), 5.54, db(-14), bus='bed')
sn = ns(1.1); siren = lp(saw(800 + 300 * np.sin(2 * np.pi * 1.2 * tax(sn)), sn), 2500) * np.linspace(1, 0, sn) ** 2
place(siren, 5.6, db(-27), .7, verb=.4)
place(horn[:ns(.35)], 5.75, db(-26), -.7, verb=.4)

# ---- 7. flatline — absolute silence
place(sine(38, ns(1.25)) * hump(ns(1.25), .3, 1, 1), 6.75, db(-24), bus='bed')
place(heartbeat_blip(), 6.80, db(-17)); place(heartbeat_blip(), 6.86, db(-16))
place(flatline(1.0), 6.95, db(-19), verb=.2)
for t in (6.9, 7.02, 7.14, 7.26): place(tick(3000, .015, 3), t, db(-24))

# ---- 8. grid fails in stages
hn8 = ns(2.7); hb = hum(2.7, 60, 10, .4); lvl = np.ones(hn8); tt8 = tax(hn8) + 7.96
for k, s_ in enumerate((8.55, 9.3, 10.05, 10.6)): lvl[tt8 >= s_] = [0.7, .45, .22, 0][k]
lvl = np.convolve(lvl, np.ones(ns(.04)) / ns(.04), 'same'); place(hb * lvl, 7.96, db(-19), bus='bed')
for _ in range(14): place(blip(R.uniform(500, 900), .07, .1), R.uniform(8.0, 10.4), db(-29), R.uniform(-.8, .8))
for k, s_ in enumerate((8.55, 9.3, 10.05)):
    place(zap(.45), s_ - .12, db(-11), [-.4, .4, 0][k]); place(impact(.9 + .1 * k, 100, 30), s_, db(-5 + k), verb=.3)
    place(klaxon(.6), s_ + .04, db(-18), verb=.25); place(power_down(.7, 320, 45), s_, db(-15))
place(power_down(1.4, 520, 18), 10.6, db(-7)); place(impact(1.4, 70, 22), 10.6, db(-3), verb=.45); place(glitch(.2), 10.6, db(-15))

# ---- 9. unattended systems shut down
place(fan(1.75), 11.04, db(-15), bus='bed')
for _ in range(10): place(tick(1100, .02, 5), R.uniform(11.1, 12.2), db(-23), R.uniform(-.6, .6))
for _ in range(22): place(blip(R.uniform(2200, 4200), .04, 0), R.uniform(11.3, 12.4), db(-29), R.uniform(-.9, .9))
place(typing(1.25, 22), 11.06, db(-18), .1)
place(beep(440, .16), 12.13, db(-15)); place(beep(220, .3), 12.29, db(-15))
place(crt_off(), 12.49, db(-12), verb=.2)

# ---- 10. dogs wait by the door
place(wind(1.5, 180, 520, .5), 12.75, db(-24), bus='bed')
place(pad(1.6, [146.8, 174.6, 220.0, 293.7], 900, .35, .6), 12.75, db(-22), bus='bed', verb=.3)
for k in range(3): place(clock_tick(k % 2 == 0), 12.9 + k * .5, db(-25), .3)
for t in (13.42, 13.87): place(thump(150, 75, .2, .4), t, db(-18), .25, verb=.15)

# ---- 11. others adapt (pack running at sunrise)
place(wind(1.5, 300, 1000, .7), 14.18, db(-22), bus='bed'); place(birds(1.5, 3), 14.18, db(-19), bus='bed', verb=.2)
pt = 14.2
while pt < 15.6:
    g = lp(wn(ns(.03)), R.uniform(700, 1600)) * expdec(ns(.03), .006, .0005)
    place(g, pt, db(-15) * R.uniform(.5, 1), np.interp(pt, [14.2, 15.6], [-.6, .6])); pt += R.uniform(.03, .07)
place(whoosh(1.4, 200, 1200, 1.0, .5, -.7, .7), 14.2, db(-19))

# ---- 12. months pass (timelapse)
ticks(15.65, 16.85, 6, 22, 2400, -20, .5, 7)
for k in range(3): place(whoosh(.42, 250, 2200, 1.4, .5, -.7 + 1.4 * (k % 2), .7 - 1.4 * (k % 2)), 15.65 + k * .42, db(-16))
place(riser(1.2, 300, 3000, tone=False, curve=1.0), 15.68, db(-21))
for k in range(12): place(page_flip(), 15.65 + k * .1025, db(-19), .4 * (1 if k % 2 else -1))

# ---- 13. weeds crack the pavement open
rn = ns(.5); place(lp(brown(rn), 110) * np.linspace(0, 1, rn) ** 2, 16.88, db(-14), bus='bed')
for _ in range(10): place(tick(R.uniform(1200, 3000), .02, 3), R.uniform(16.95, 17.3), db(-25), R.uniform(-.5, .5))
place(crack(), 17.33, db(-3), verb=.25); place(impact(1.0, 85, 30), 17.33, db(-7))
place(crumble(1.0), 17.38, db(-14), .1)
place(shimmer(1.4, 330, 2.0), 17.5, db(-20), verb=.35); place(rustle(1.4, 5200), 17.5, db(-21))
for t in (17.8, 18.2): place(creak(.5, 900, 60, 120), t, db(-25), R.uniform(-.4, .4))
for k in range(16): place(bubble(), 17.5 + k * .027 + R.uniform(0, .02), db(-27), R.uniform(-.6, .6))

# ---- 14-16. years in: deer, suburb, sagging houses
place(pad(3.95, [146.8, 220.0, 293.7, 329.6, 370.0], 1800, .5, .7), 18.85, db(-21), bus='bed', verb=.3)
place(wind(3.95, 250, 800, .4), 18.85, db(-25), bus='bed'); place(birds(3.9, 2.6), 18.85, db(-19), bus='bed', verb=.25)
place(impact(.6, 120, 50), 18.88, db(-11), verb=.3); place(ding(880, .9), 18.9, db(-21), verb=.3)
place(rustle(1.0, 4200), 18.95, db(-25))
for dr, off, g in ((0, 0, -21), (1, .2, -24)):
    t = 20.06 + off
    while t < 21.4:
        cl = bp(wn(ns(.03)), 850, 4) * expdec(ns(.03), .008, .0005)
        cl[:ns(.03)] += .5 * thump(220, 120, .03, 0)[:ns(.03)]
        place(lp(cl, 2500), t, db(g), np.interp(t, [20.0, 21.4], [-.7, .4])); t += .22 + R.uniform(-.02, .02)
place(creak(.7, 380, 25, 60), 20.6, db(-25), .5, verb=.3)
place(creak(.5, 420, 30, 90), 21.5, db(-17), -.1, verb=.2); place(creak(.4, 300, 25, 70), 21.74, db(-15), .2, verb=.2)
place(crack(), 21.95, db(-6), verb=.3); place(impact(1.1, 90, 30), 21.95, db(-5), verb=.3)
place(crumble(1.2), 21.98, db(-12)); place(sparkle(.5, 200, 4000, 12000, .02), 21.97, db(-17))
place(lp(brown(ns(1.0)), 150) * expdec(ns(1.0), .4, .01), 21.95, db(-12))
place(wingflaps(.8), 21.97, db(-17), -.3, verb=.2); place(chirp(4200, 6200, 4, .04, .02), 22.02, db(-20), .4)
place(shimmer(1.2, 220, 1.5), 21.6, db(-23), verb=.3)

# ---- 17. speeding through decades (time warp)
place(riser(2.22, 150, 5200, curve=1.3), 22.78, db(-9), verb=.2)
tw_ = ns(2.2); ff_ = np.geomspace(60, 900, tw_); tape = lp(saw(ff_, tw_) + .7 * saw(ff_ * 1.5, tw_), 3000) * np.linspace(.3, 1, tw_) * (.7 + .3 * np.sin(2 * np.pi * 9 * tax(tw_)))
place(np.vstack([tape, np.roll(tape, 300)]), 22.8, db(-17))
ticks(22.8, 24.9, 6, 60, 2600, -18, .2, 6)
for k, t in enumerate((23.0, 23.5, 23.9, 24.3)): place(whoosh(.45, 300 + 200 * k, 3000 + 1000 * k, 1.5, .6, -.6, .6), t, db(-19))
place(ui_pop(400, 1200), 22.8, db(-16))
place(rev_swell(.5, 800, 8000), 24.5, db(-12))

# ---- 18. bridges weaken and fall
place(wind(1.75, 300, 1200, .8), 25.0, db(-21), bus='bed')
wn_ = ns(1.75); river = bp(pink(wn_), 700, .7) * (.7 + .3 * smooth_rand(wn_, 3, 0, 1)) + .4 * hp(wn(wn_), 5000) * smooth_rand(wn_, 20, 0, 1)
place(river, 25.0, db(-24), bus='bed')
place(metal_groan(1.0, 55), 25.12, db(-12), -.3, verb=.35); place(metal_groan(.8, 47), 25.72, db(-11), .3, verb=.35)
for t, pn, g in ((25.45, -.5, -15), (25.7, .4, -14), (25.95, -.2, -13), (26.12, .6, -12)): place(cable_snap(), t, db(g), pn, verb=.3)
place(impact(1.6, 60, 22), 26.22, db(-2), verb=.45); place(clank(180, .6), 26.22, db(-12), -.3, verb=.4)
place(clank(255, .5), 26.26, db(-13), .3, verb=.4); place(metal_groan(1.2, 40), 26.18, db(-10), verb=.3)
place(crumble(1.2), 26.25, db(-14)); place(splash(1.6), 26.48, db(-7), verb=.3)

# ---- 19-21. forests swallow streets, landmarks wrapped in green
place(pad(6.35, [146.8, 220.0, 277.2, 329.6, 370.0], 2200, .8, .4), 26.75, db(-19), bus='bed', verb=.3)
place(rustle(6.3, 4600), 26.75, db(-26), bus='bed'); place(wind(6.3, 280, 900, .35), 26.75, db(-25), bus='bed')
place(birds(6.3, 2.4), 26.75, db(-20), bus='bed', verb=.25)
place(whoosh(1.7, 200, 900, 1.0, .5, -.2, .2), 26.78, db(-22)); place(shimmer(1.7, 196, 2.0), 26.8, db(-21), verb=.3)
for _ in range(12):
    t = R.uniform(26.85, 28.2); place(thump(150, 90, .15, .3), t, db(-24), R.uniform(-.7, .7)); place(rustle(.25, 3500), t, db(-25))
place(wingflaps(1.2, 14), 28.6, db(-23), .5, verb=.3)
for t in (29.0, 29.6, 30.0): place(creak(.5, R.uniform(450, 800), 40, 90), t, db(-26), R.uniform(-.6, .6))
place(whoosh(2.6, 200, 4000, 1.2, .92, 0, 0), 30.45, db(-17), verb=.2); place(riser(2.4, 300, 2500, tone=False), 30.5, db(-20))
place(rustle(2.8, 6000), 30.4, db(-23))

# ---- 22. please read the description
place(whoosh(.35, 800, 5000, 1.5, .5, -.3, .3), 33.3, db(-15)); place(ui_pop(380, 1100), 33.45, db(-14))
place(ding(1760, .8), 33.62, db(-22), verb=.3)
place(pad(2.95, [146.8, 220.0, 329.6, 440.0], 1100, .4, .4), 33.28, db(-21), bus='bed', verb=.3)
place(rustle(2.9, 5000), 33.28, db(-28), bus='bed')
for c0 in np.arange(33.75, 36.0, .91):
    for j, f in enumerate((1318.5, 1174.7, 987.8)): place(blip(f, .08, .15), c0 + j * .114, db(-27))
place(ui_pop(300, 900), 34.58, db(-15)); place(thump(130, 70, .2, .2), 34.58, db(-19))

# ---- 23. roads leave their traces (satellite scan)
place(scan(1.5, 400, 1600), 36.22, db(-17), verb=.2)
place(hum(1.8, 55, 5, 0), 36.18, db(-29), bus='bed'); place(wind(1.8, 400, 900, .4), 36.18, db(-27), bus='bed')
for t in (36.53, 36.83, 37.13): place(ui_pop(500, 1300), t, db(-16)); place(blip(1600, .06), t + .04, db(-21))

# ---- 24. human world -> wilderness wipe
place(whoosh(1.7, 200, 5000, 1.2, .5, .9, -.9), 38.1, db(-13), verb=.2)
n24 = ns(2.1); xf = np.clip((tax(n24) - .2) / 1.6, 0, 1)
cty = lp(brown(n24), 260) + .5 * bp(pink(n24), 900, .7); place(cty * (1 - xf), 38.0, db(-20), bus='bed')
place(birds(2.1, 3.5) * xf, 38.0, db(-19), bus='bed'); place(wind(2.1, 300, 900) * xf, 38.0, db(-23), bus='bed')
place(rustle(1.7, 4800), 38.1, db(-20))
place(ui_pop(450, 1250), 39.05, db(-15)); place(ding(1568, .8), 39.07, db(-21), verb=.3)

# ---- 25. faster than you'd imagine (earth goes green)
place(drone(2.2, 41.2), 40.1, db(-17), bus='bed'); place(choir(2.2, [146.8, 220.0, 293.7, 370.0], 1.0, .3), 40.1, db(-19), bus='bed', verb=.4)
place(sparkle(1.65, 40, 1500, 4000, .06), 40.25, db(-22))
ticks(40.25, 41.9, 20, 8, 3200, -25, .2, 3)
place(riser(.9, 400, 6000), 41.32, db(-13))

# ---- 26. CTA: what's your wildest what-if?
place(impact(1.2), 42.22, db(-4), verb=.35)
place(pad(.9, [293.7, 370.0, 440.0, 587.3], 5000, .005, .7), 42.22, db(-14), verb=.4)
place(sparkle(.6, 120, 4000, 12000, .04), 42.22, db(-18))
for t, a, b, g in ((42.22, 500, 1300, -13), (42.37, 400, 1000, -12), (42.57, 600, 1600, -13)): place(ui_pop(a, b), t, db(g))
thumps(42.22, 46.3, .5, -15, 120, 45)
for t in np.arange(42.47, 46.3, .5): place(hp(wn(ns(.04)), 8000) * expdec(ns(.04), .01, .0005), t, db(-25), .2)
for i, f in enumerate((1318.5, 1568.0, 1760.0, 2093.0)):
    t = 42.22 + .75 + i * .4; sd = 1 if i % 2 else -1
    place(whoosh(.3, 600, 3000, 1.5, .5, sd * .8, sd * .2), t - .05, db(-21))
    place(bubble(), t + .12, db(-14), sd * .2); place(ding(f, .7), t + .14, db(-19), sd * .2, verb=.25)
    for j in range(3): place(ping(2600 + 400 * j, .03, .12), t + .5 + j * .09, db(-26), .5)
place(ui_pop(300, 800), 42.82, db(-15))
for t in (43.92, 45.82): place(tick(2500) + thump(300, 200, .06, 0)[:ns(.012)], t, db(-14), .3); place(ding(2093, .5), t + .02, db(-22), .3)
place(whoosh(.3, 500, 3500, 1.5, .5, 0, 0), 44.97, db(-18))
place(whoosh(.35, 300, 4000, 1.4, .6, -.2, .2), 45.08, db(-15)); place(impact(.5, 140, 50, 5000), 45.12, db(-11), verb=.3)
place(sparkle(.8, 80, 3000, 10000, .05), 45.15, db(-17)); place(shimmer(1.2, 440, 1.5), 45.15, db(-21), verb=.3)
place(impact(.55, 130, 45, 5000), 46.22, db(-10), verb=.3); place(ding(1760, .3), 46.24, db(-18))

# ======================================================================
#                               MASTER
# ======================================================================
# word onsets of the voice-over (same timing as the captions in the HTML);
# ambience beds dip ~5 dB while words are spoken so the VO stays on top.
VO_WORDS = [0, 0.3, 0.55, 0.95, 1.4, 1.85, 2.6, 3, 3.45, 3.85, 4.05, 4.25, 4.45, 4.7, 4.95, 5.1, 5.3, 6, 6.3, 6.5, 6.9, 7.25, 7.96, 8.35, 8.9, 9.1, 9.55, 10, 10.2, 11.04, 11.65, 12.1, 12.4, 12.9, 13.3, 13.6, 13.8, 14, 14.6, 15.05, 15.75, 16.25, 16.88, 17.3, 17.7, 17.9, 18.4, 18.95, 19.35, 19.6, 20.04, 20.5, 20.75, 21.3, 21.5, 21.95, 22.25, 22.45, 22.55, 22.9, 23.2, 23.5, 23.9, 24.3, 25.1, 25.64, 26.05, 26.25, 26.8, 27.25, 27.65, 27.95, 28.6, 29.2, 29.7, 30.4, 30.9, 31.1, 32.4, 32.8, 33.28, 33.75, 34.1, 34.3, 36.18, 36.6, 36.9, 37.15, 37.9, 38.05, 38.45, 38.85, 39.05, 39.45, 40.1, 40.5, 40.7, 41, 41.7, 42, 42.22, 42.7, 42.95, 43.2, 43.9, 44.15, 44.3, 44.45, 44.6, 45.1, 45.3, 45.55, 45.75, 45.95, 46.1, 46.25]

def duck_curve():
    sp = np.zeros(N)
    for i, t in enumerate(VO_WORDS):
        nxt = VO_WORDS[i + 1] if i + 1 < len(VO_WORDS) else t + .5
        sp[S(t):S(min(nxt, t + .55) + .08)] = 1
    a = np.exp(-1 / (.04 * SR)); r = np.exp(-1 / (.25 * SR))
    env = signal.lfilter([1 - r], [1, -r], sp)            # slow release
    env = np.maximum(env, signal.lfilter([1 - a], [1, -a], sp))
    return 1 - (1 - db(-5)) * np.clip(env, 0, 1)

def make_ir(dur=2.3, decay=.5):
    n = ns(dur); t = tax(n); ir = []
    for _ in range(2):
        x = lp(wn(n), 5500) * np.exp(-t / decay); pre = ns(.015); x[:pre] = 0
        x[pre:pre + ns(.01)] *= np.linspace(0, 1, ns(.01)); ir.append(x / np.sqrt(np.sum(x ** 2)))
    return ir

def tape_stop(x, t0, t1):
    i0, i1 = S(t0), S(t1); n = i1 - i0
    rate = np.linspace(1, 0, n) ** 1.3; pos = i0 + np.cumsum(rate)
    for ch in range(2):
        x[ch, i0:i1] = np.interp(pos, np.arange(x.shape[1]), x[ch]) * np.linspace(1, .2, n)
    return x

def limiter(x, ceil=db(-1.0), look=.004, rel=.08):
    a = np.max(np.abs(x), axis=0); w = ns(look) * 2 + 1
    g = np.minimum(1, ceil / np.maximum(maximum_filter1d(a, w), 1e-9))
    g = minimum_filter1d(g, ns(.01))
    c = np.exp(-1 / (rel * SR)); g = signal.filtfilt([1 - c], [1, -c], g)
    y = x * np.minimum(g, 1)
    return np.clip(y, -ceil, ceil)

def write_wav24(path, x):
    x = np.clip(x, -1, 1)
    iv = np.ascontiguousarray((x.T * 8388607).astype('<i4'))
    b = iv.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(3); w.setframerate(SR); w.writeframes(b)

OUT_GAIN = -6.0   # stem level: ~-21 LUFS / peaks ~-7 dBFS, sits under a -14 LUFS voice-over

def master():
    ir = make_ir()
    snd = np.vstack([hp(SEND[c], 220, 2) for c in range(2)])       # keep sub-bass out of the reverb
    wet = np.vstack([signal.fftconvolve(snd[c], ir[c])[:N] for c in range(2)]) * db(-3)
    mix = FX + BED * duck_curve() + wet
    mix = tape_stop(mix, 33.04, 33.27)                      # "but the..." -> record stops
    mix = np.vstack([hp(mix[c], 24, 4) for c in range(2)])
    mix = np.tanh(mix * 1.15) / np.tanh(1.15)               # gentle glue / saturation
    mix = mix / (np.max(np.abs(mix)) + 1e-9) * db(-0.5)
    mix = limiter(mix * db(3.0))
    mix = fade(mix, .004, .06)
    return mix * db(OUT_GAIN)

if __name__ == '__main__':
    out = master()
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'what-if-humans-disappeared-sfx.wav')
    write_wav24(path, out)
    rms = 20 * np.log10(np.sqrt(np.mean(out ** 2)) + 1e-12); pk = 20 * np.log10(np.max(np.abs(out)))
    print(f'wrote {path}  {out.shape[1] / SR:.2f}s  peak {pk:.1f} dBFS  rms {rms:.1f} dBFS')
