"""
Procedural sound design for "What If the Sun Disappeared for 24 Hours" (YouTube Short).
Every event is placed on the exact shot cuts / beat times used by the HTML animation.
Output: 48 kHz, 24-bit, stereo WAV, 43.9 s.
"""
import sys
import numpy as np
from scipy import signal

SR = 48000
DUR = 43.9
N = int(SR * DUR)
rng = np.random.default_rng(2024)

dryL = np.zeros(N); dryR = np.zeros(N)
sndL = np.zeros(N); sndR = np.zeros(N)

# ------------------------------------------------------------------ basics
def ns(d): return max(1, int(round(d * SR)))
def tt(d): return np.arange(ns(d)) / SR
def white(d): return rng.standard_normal(ns(d))
def _shaped(d, p):
    n = ns(d); X = np.fft.rfft(rng.standard_normal(n)); f = np.fft.rfftfreq(n, 1 / SR); f[0] = f[1]
    y = np.fft.irfft(X / f ** p, n); return y / (np.std(y) + 1e-12)
def pink(d): return _shaped(d, .5)
def brown(d): return _shaped(d, 1.0)
def sos(kind, f, o=2):
    return signal.butter(o, f, btype=kind, fs=SR, output='sos')
def lp(x, f, o=2): return signal.sosfilt(sos('lowpass', min(f, SR * .45), o), x)
def hp(x, f, o=2): return signal.sosfilt(sos('highpass', f, o), x)
def bp(x, f1, f2, o=2): return signal.sosfilt(sos('bandpass', [f1, min(f2, SR * .45)], o), x)
def norm(x, p=1.0):
    m = np.max(np.abs(x)); return x * (p / m) if m > 0 else x
def sat(x, k=2.0): return np.tanh(x * k) / np.tanh(k)
def fade(x, a=.004, r=.01):
    x = x.copy(); na, nr = min(len(x), ns(a)), min(len(x), ns(r))
    x[:na] *= np.linspace(0, 1, na); x[len(x) - nr:] *= np.linspace(1, 0, nr); return x
def curve(d, pts):
    """piecewise-linear control curve; pts=[(t,v),...] with t in seconds"""
    t = tt(d); ts, vs = zip(*pts); return np.interp(t, ts, vs)

def rbj(kind, fc, Q):
    w0 = 2 * np.pi * fc / SR; cw, sw = np.cos(w0), np.sin(w0); al = sw / (2 * Q)
    if kind == 'lp': b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
    elif kind == 'hp': b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
    else: b = [al, 0, -al]
    a = [1 + al, -2 * cw, 1 - al]
    return np.array(b) / a[0], np.array(a) / a[0]
def tvf(x, fc, kind='bp', Q=1.0, block=128):
    """time-varying biquad (block-wise coefficient updates)"""
    fc = np.broadcast_to(np.asarray(fc, float), x.shape)
    y = np.empty_like(x); zi = np.zeros(2)
    for i in range(0, len(x), block):
        b, a = rbj(kind, float(np.clip(fc[i], 20, SR * .45)), Q)
        y[i:i + block], zi = signal.lfilter(b, a, x[i:i + block], zi=zi)
    return y

def phase(f, n):
    f = np.broadcast_to(np.asarray(f, float), (n,)); return np.cumsum(f) / SR
def sine(f, d): n = ns(d); return np.sin(2 * np.pi * phase(f, n))
def saw(f, d):
    n = ns(d); f = np.broadcast_to(np.asarray(f, float), (n,)).astype(float)
    ph = phase(f, n) % 1.0; dt = np.maximum(f / SR, 1e-9); y = 2 * ph - 1
    m1 = ph < dt; t1 = ph[m1] / dt[m1]; y[m1] -= t1 + t1 - t1 * t1 - 1
    m2 = ph > 1 - dt; t2 = (ph[m2] - 1) / dt[m2]; y[m2] -= t2 * t2 + t2 + t2 + 1
    return y
def smooth_noise(d, rate, seed_scale=1.0):
    """smoothed random control signal in [-1,1] changing ~rate times per second"""
    n = ns(d); k = max(2, int(d * rate) + 2)
    pts = rng.uniform(-1, 1, k) * seed_scale
    return np.interp(np.linspace(0, k - 1, n), np.arange(k), pts)

# ------------------------------------------------------------------ mixer
def put(x, t0, g=1.0, pan=0.0, send=.2, width=0.0):
    """mono x at time t0 (s). pan in [-1,1] (scalar or array). width adds Haas-ish decorrelation."""
    i0 = int(round(t0 * SR))
    if i0 >= N: return
    p = np.asarray(pan, float)
    if i0 < 0:
        x = x[-i0:]
        if p.ndim: p = p[-i0:]
        i0 = 0
    x = x[:N - i0]
    if p.ndim:
        p = p[:len(x)]
        if len(p) < len(x): p = np.concatenate([p, np.full(len(x) - len(p), p[-1])])
    gl = np.cos((p + 1) * np.pi / 4); gr = np.sin((p + 1) * np.pi / 4)
    l = x * gl * g * 1.4142; r = x * gr * g * 1.4142
    if width > 0:
        dl = int(width * SR * .012)
        r = np.concatenate([np.zeros(dl), r])[:len(x)]
    dryL[i0:i0 + len(x)] += l; dryR[i0:i0 + len(x)] += r
    if send > 0:
        sndL[i0:i0 + len(x)] += l * send; sndR[i0:i0 + len(x)] += r * send
def put2(l, r, t0, g=1.0, send=.2):
    i0 = int(round(t0 * SR)); n = min(len(l), N - i0)
    if n <= 0: return
    dryL[i0:i0 + n] += l[:n] * g; dryR[i0:i0 + n] += r[:n] * g
    sndL[i0:i0 + n] += l[:n] * g * send; sndR[i0:i0 + n] += r[:n] * g * send

# ------------------------------------------------------------------ instruments
def impact(size=1.0, d=None, bright=1.0):
    d = d or 1.2 + 1.6 * size; t = tt(d)
    f = 30 + 105 * np.exp(-t / (.07 + .05 * size))
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (.35 + .5 * size))
    body = lp(white(d), 700) * np.exp(-t / .09) * 1.6
    crack = hp(white(d), 2500) * np.exp(-t / .02) * .5 * bright
    tail = lp(white(d), 260) * np.exp(-t / (.3 + .45 * size)) * .2
    x = sat(sub * 1.2 + body * .55 + crack + tail, 1.8)
    return norm(fade(x, .001, .05))
def sub_drop(d=1.6, f0=120, f1=28):
    t = tt(d); f = f1 + (f0 - f1) * np.exp(-t / (d * .25))
    return norm(fade(sat(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (d * .45)), 1.4), .002, .1))
def braam(d=2.3, root=36.71):
    t = tt(d); x = np.zeros(ns(d))
    for mult, g in [(1, 1), (2, .8), (3, .45), (4, .35), (6, .2)]:
        for det in (-.004, 0, .005):
            x += saw(root * mult * (1 + det), d) * g
    cut = 140 + 1700 * np.exp(-t / .25) * (1 - np.exp(-t / .015)) + 350 * np.exp(-t / 1.2)
    x = tvf(x, cut, 'lp', .9)
    amp = (1 - np.exp(-t / .02)) * (.55 + .45 * np.exp(-t / .5)) * np.clip((d - t) / .6, 0, 1)
    x = sat(x * amp * .6, 2.4)
    return norm(fade(x + impact(1.2, d) * .7, .001, .1))
def whoosh(d=.55, f0=250, f1=3200, f2=500, pk=.72, Q=1.3):
    t = tt(d); u = t / d
    fc = np.where(u < pk, f0 * (f1 / f0) ** (u / pk), f1 * (f2 / f1) ** ((u - pk) / (1 - pk)))
    amp = np.where(u < pk, (u / pk) ** 2.2, np.exp(-(u - pk) / (1 - pk) * 4))
    x = tvf(white(d), fc, 'bp', Q) * amp + lp(pink(d), 400) * amp * .35
    return norm(fade(x))
def riser(d=1.0, f0=180, f1=2400, noise=1.0):
    t = tt(d); u = t / d
    tone = saw(f0 * (f1 / f0) ** (u ** 1.5), d) * .25 + sine(f0 * 2 * (f1 / f0) ** (u ** 1.5), d) * .3
    tone = lp(tone, 4000)
    nz = tvf(white(d), 300 * (14000 / 300) ** u, 'bp', .9) * noise
    trem = .75 + .25 * np.sin(2 * np.pi * np.cumsum(4 + 26 * u ** 2) / SR)
    return norm(fade((tone + nz) * u ** 2.2 * trem, .01, .004))
def reverse_swell(d=.8, f=2200):
    t = tt(d); x = lp(white(d), f) * np.exp(-t / (d * .3)); x += hp(white(d), 4000) * np.exp(-t / (d * .15)) * .4
    taps = [(0, 1), (.03, .6), (.071, .45), (.113, .3)]
    y = np.zeros_like(x)
    for dt, g in taps:
        k = ns(dt) if dt > 0 else 0; y[k:] += x[:len(x) - k] * g
    return norm(fade(y[::-1], .02, .002))
def click(d=.006, f=3000):
    t = tt(d); return norm(hp(white(d), f) * np.exp(-t / (d * .25)))
def tick(f=1500, d=.05, body=.5):
    t = tt(d); x = sine(f, d) * np.exp(-t / .012) + np.pad(click(.004, 2500), (0, ns(d) - ns(.004))) * body
    return norm(fade(x, .0005, .005))
def beep(f=1000, d=.12, harm=.3, sq=False):
    t = tt(d); x = sine(f, d) + harm * sine(f * 2, d) + (harm * .5 * sine(f * 3, d))
    if sq: x = np.sign(x) * .5 + x * .5
    return norm(fade(x * np.exp(-t / (d * .8)), .003, .02))
def bell(f=880, d=1.6, bright=1.0):
    t = tt(d); x = np.zeros(ns(d))
    for r, g, tau in [(1, 1, .9), (2.0, .5, .55), (2.76, .35, .4), (5.4, .2 * bright, .2), (8.93, .12 * bright, .1)]:
        x += sine(f * r, d) * g * np.exp(-t / (tau * d / 1.6))
    return norm(fade(x, .001, .05))
def pop(f0=900, f1=240, d=.09):
    t = tt(d); f = f1 + (f0 - f1) * np.exp(-t / .015)
    return norm(fade(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / .03) + np.pad(click(.003, 1800), (0, ns(d) - ns(.003))) * .3, .0005, .01))
def keyclick():
    d = .045; t = tt(d); f = rng.uniform(1800, 3200)
    x = bp(white(d), f * .6, f * 1.6) * np.exp(-t / .006) + sine(rng.uniform(140, 210), d) * np.exp(-t / .01) * .5
    return norm(fade(x, .0003, .01))
def heartbeat(g2=.65):
    d = .5; t = tt(d)
    def thump(f, tau): return np.sin(2 * np.pi * np.cumsum(f * (1 + .6 * np.exp(-t / .02))) / SR) * np.exp(-t / tau)
    a = thump(52, .09); b = np.zeros(ns(d)); k = ns(.19); b[k:] = thump(46, .1)[:ns(d) - k] * g2
    return norm(fade(sat((a + b) * 1.4, 1.6) + lp(white(d), 200) * np.exp(-t / .04) * .2, .001, .05))
def shatter(d=1.6):
    t = tt(d); x = np.zeros(ns(d))
    for _ in range(90):
        f = rng.uniform(2200, 11000); st = rng.uniform(0, .22) ** 1.6; tau = rng.uniform(.03, .35)
        k = ns(st); seg = sine(f, d - st) * np.exp(-tt(d - st) / tau) * rng.uniform(.2, 1)
        m = min(len(seg), len(x) - k); x[k:k + m] += seg[:m]
    x += hp(white(d), 3000) * np.exp(-t / .18) * 3
    x += lp(white(d), 900) * np.exp(-t / .05) * 1.5
    return norm(fade(x, .001, .1))
def crackle(d, density, f_lo=1500, f_hi=9000, decay=.0015):
    """sparse random impulses; density = curve (events/sec) array or scalar"""
    n = ns(d); dens = np.broadcast_to(np.asarray(density, float), (n,))
    imp = (rng.random(n) < dens / SR) * rng.uniform(-1, 1, n) ** 3
    k = np.exp(-np.arange(ns(decay * 6)) / (decay * SR)) * rng.standard_normal(ns(decay * 6))
    return norm(bp(signal.fftconvolve(imp, k)[:n], f_lo, f_hi))
def hum(d, f=60, fdrop=None):
    f = np.broadcast_to(np.asarray(f, float), (ns(d),))
    x = saw(f, d) * .5 + saw(f * 2.003, d) * .25 + sine(f * 3, d) * .2 + sine(f, d) * .6
    x = lp(x, 1800) + bp(white(d), 120, 900) * .05 * (1 + np.sin(2 * np.pi * np.cumsum(f * 2) / SR))
    return norm(x)
def wind(d, base=500, spread=350, gust_rate=.6, whistle=.15):
    t = tt(d); g = .55 + .45 * smooth_noise(d, gust_rate)
    fc = base + spread * smooth_noise(d, gust_rate * .8)
    x = tvf(brown(d) * .6 + pink(d) * .4, fc, 'bp', .6) * g
    if whistle:
        wf = 900 + 500 * smooth_noise(d, .35); x += tvf(white(d), wf, 'bp', 14) * g ** 2 * whistle * 3
    return norm(x)
def pad(notes, d, cut=1400, att=.6, rel=.8, det=(-.006, 0, .007), sub=None):
    t = tt(d); x = np.zeros(ns(d))
    for f in notes:
        for dd in det: x += saw(f * (1 + dd), d)
    x = tvf(x, cut * (1 + .25 * np.sin(2 * np.pi * .18 * t)), 'lp', .7)
    if sub: x += sine(sub, d) * len(notes) * .6
    env = np.clip(t / att, 0, 1) ** 1.5 * np.clip((d - t) / rel, 0, 1)
    return norm(x * env)
def glitch(d=.32):
    n = ns(d); x = np.zeros(n); i = 0
    while i < n:
        seg = ns(rng.uniform(.012, .045)); kind = rng.integers(0, 4); tseg = np.arange(seg) / SR
        if kind == 0: s = np.sign(np.sin(2 * np.pi * rng.uniform(200, 2400) * tseg))
        elif kind == 1: s = np.repeat(rng.uniform(-1, 1, seg // 24 + 1), 24)[:seg]
        elif kind == 2: s = hp(rng.standard_normal(seg), 3000)
        else: s = np.zeros(seg)
        s = s * rng.uniform(.3, 1); x[i:i + seg] = s[:n - i]; i += seg
    x = np.round(x * 6) / 6
    return norm(fade(x * np.exp(-tt(d) / (d * .7)), .001, .01))
def chatter(d=.3, lo=900, hi=3200):
    n = ns(d); x = np.zeros(n); i = 0
    while i < n:
        seg = ns(rng.uniform(.012, .03)); f = rng.uniform(lo, hi); ts = np.arange(seg) / SR
        s = np.sign(np.sin(2 * np.pi * f * ts)) * .5 + np.sin(2 * np.pi * f * ts) * .5
        x[i:i + seg] = (s * np.hanning(seg))[:n - i]; i += seg + ns(rng.uniform(0, .012))
    return norm(lp(x, 6000))
def sonar(f=1150, d=2.2):
    t = tt(d); x = sine(f, d) * np.exp(-t / .28) + sine(f * 2.01, d) * np.exp(-t / .12) * .2
    y = x.copy()
    for k, g in [(.26, .45), (.52, .25), (.78, .12)]:
        o = ns(k)
        if o < len(x): y[o:] += lp(x, 2500)[:len(x) - o] * g
    return norm(fade(y, .002, .1))
def breath(d=1.1):
    t = tt(d); env = np.sin(np.pi * np.clip(t / d, 0, 1)) ** 1.5 * np.exp(-t / .6)
    return norm(fade(tvf(white(d), 900 + 700 * np.exp(-t / .3), 'bp', .8) * env))
def footsteps(d, rate=14):
    n = ns(d); x = np.zeros(n); k = ns(.09)
    step = lp(white(.09), 380) * np.exp(-tt(.09) / .025) + bp(white(.09), 1500, 5000) * np.exp(-tt(.09) / .012) * .25
    times = np.cumsum(rng.exponential(1 / rate, int(d * rate * 2)))
    for tm in times[times < d - .1]:
        i = ns(tm); x[i:i + k] += step * rng.uniform(.3, 1)
    return norm(x)
def walla(d, voices=7):
    x = np.zeros(ns(d))
    for _ in range(voices):
        f = 500 + 450 * smooth_noise(d, 9); g = np.clip(smooth_noise(d, 5) + .2, 0, 1)
        x += tvf(white(d), f, 'bp', 4) * g + tvf(white(d), f * 2.6, 'bp', 6) * g * .5
    return norm(lp(x, 2800))
def flutter(d, birds=40):
    x = np.zeros(ns(d)); t = tt(d)
    for _ in range(birds):
        rate = rng.uniform(7, 13); ph = rng.uniform(0, 1)
        g = np.clip(np.sin(2 * np.pi * (rate * t + ph)), 0, 1) ** 6 * rng.uniform(.3, 1)
        x += g
    x = bp(white(d), 300, 2600) * x
    return norm(x)
def chirp(d=.07):
    t = tt(d); f0 = rng.uniform(2600, 4200); f = f0 * (1 + .55 * np.sin(np.pi * t / d)) * (1 + .1 * np.sin(2 * np.pi * 70 * t))
    return norm(fade(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / d) ** 2, .002, .01))
def siren(d, period=1.3, lo=520, hi=1150):
    t = tt(d); f = lo + (hi - lo) * (.5 - .5 * np.cos(2 * np.pi * t / period))
    x = saw(f, d) * .4 + sine(f, d)
    return norm(lp(x, 2600))
def eas(d=.55):
    t = tt(d); x = sine(853, d) + sine(960, d)
    return norm(fade(sat(x, 1.3), .01, .03))
def power_down(d=1.0, f0=420, f1=40):
    t = tt(d); u = t / d; f = f1 + (f0 - f1) * (1 - u) ** 2.2
    x = saw(f, d) * .5 + sine(f, d)
    x = tvf(x, 300 + 4000 * (1 - u) ** 2, 'lp', .9) * (1 - u) ** .6
    return norm(fade(x, .005, .05))
def shimmer(d, density=30, lo=3000, hi=9000):
    x = np.zeros(ns(d))
    for tm in np.sort(rng.uniform(0, d, int(d * density))):
        f = rng.uniform(lo, hi); dd = rng.uniform(.08, .35); k = ns(tm)
        s = sine(f, dd) * np.exp(-tt(dd) / (dd * .3)) * rng.uniform(.3, 1)
        x[k:k + len(s)] += s[:len(x) - k]
    return norm(x)
def solar_roar(d):
    t = tt(d)
    rum = lp(brown(d), 380) * (.7 + .3 * smooth_noise(d, 3))
    mid = tvf(pink(d), 140 + 90 * smooth_noise(d, 2), 'bp', .9) * .5
    crk = crackle(d, 90, 900, 6000, .002) * .25
    drone = (sine(41.2, d) + .7 * sine(55.4, d) + .4 * sine(82.6, d)) * (.8 + .2 * np.sin(2 * np.pi * .7 * t)) * .5
    return norm(rum + mid + crk + drone)
def drone(d, freqs=(36.7, 55.0, 73.4), noise=.4, lpf=180):
    t = tt(d); x = sum(sine(f * (1 + .002 * np.sin(2 * np.pi * (.07 + i * .05) * t)), d) / (i + 1) for i, f in enumerate(freqs))
    return norm(x + lp(brown(d), lpf) * noise)

# ------------------------------------------------------------------ timeline (matches the HTML)
SHOTS = [0, 1.09, 2.00, 2.96, 4.34, 5.51, 6.02, 7.50, 9.51, 10.32, 11.20, 12.14, 13.85, 15.00, 16.09, 17.02, 18.00,
         19.01, 20.41, 21.73, 22.91, 24.34, 25.69, 26.49, 28.35, 29.78, 31.32, 32.62, 33.99, 35.22, 36.33, 38.50,
         40.21, 41.54, 43.50]
GLITCH = {1.09, 2.96, 5.51, 9.51, 11.20, 15.00, 18.00, 20.41, 21.73, 25.69, 28.35, 33.99, 38.50}
BEATS = [1.55, 2.59, 6.58, 9.87, 13.45, 14.17, 16.43, 18.54, 21.40, 23.48, 23.85, 25.69, 26.81, 29.10, 31.76, 35.82,
         39.04, 41.04, 43.11]
SPECIAL_BEATS = {1.55, 21.40, 23.48, 23.85, 25.69, 35.82, 31.76}

def section(name): print('  -', name, flush=True)

# --- transitions on every cut: a whoosh that peaks on the cut, plus a digital glitch on glitch cuts
section('cuts')
for i, c in enumerate(SHOTS[1:], 1):
    if c in (1.55,): continue
    d = rng.uniform(.42, .62); pk = .78
    w = whoosh(d, rng.uniform(180, 320), rng.uniform(2600, 5200), rng.uniform(400, 900), pk)
    pan = np.linspace(-.7, .7, ns(d)) * (1 if i % 2 else -1)
    quiet = c in (31.32, 32.62, 33.99, 35.22)
    put(w, c - d * pk, .16 if quiet else .34, pan, .18)
    if c in GLITCH:
        put(glitch(rng.uniform(.18, .3)), c, .2, rng.uniform(-.3, .3), .1)

# --- beat impacts
section('beats')
for b in BEATS:
    if b in SPECIAL_BEATS: continue
    put(impact(.55), b, .5, 0, .25)

# ================================================================ SCENES
section('sun')
# 0 - 1.55 : the living sun (hook boom at frame 0)
put(impact(1.3, 2.6), 0.0, .95, 0, .3)
put(sub_drop(1.8, 90, 30), 0.0, .55, 0, .1)
r = solar_roar(1.75); put2(r, np.roll(r, 900), 0.0, .3, .15)
put(whoosh(1.05, 1800, 500, 160, .15, .8), 0.02, .28, np.linspace(.2, -.2, ns(1.05)), .2)
put(crackle(1.5, np.linspace(60, 260, ns(1.5)), 1200, 8000), 0.0, .12, 0, .15, width=1)
# collapse (1.09 - 1.55)
put(riser(.48, 160, 2600), 1.08, .42, 0, .2)
for k, tb in enumerate([1.25, 1.35, 1.45]): put(beep(880 + 110 * k, .07, .4, True), tb, .14, 0, .2)
rs = reverse_swell(.42, 2600); put(rs, 1.55 - .42, .4, 0, .1)
# the implosion
put(impact(2.0, 3.2), 1.55, 1.0, 0, .45)
put(sub_drop(2.2, 130, 26), 1.55, .8, 0, .1)
put(whoosh(.9, 3000, 900, 120, .05, .9), 1.56, .38, np.linspace(-.9, .9, ns(.9)), .3)
put(shatter(1.2), 1.56, .14, .2, .5)
put(sine(7400, 1.3) * np.exp(-tt(1.3) / .5), 1.62, .025, .1, .3)  # after-image ring
put(drone(1.5, (41.2, 61.7), .3), 1.7, .16, 0, .2)

section('24h')
put(impact(.8), 2.00, .55, 0, .25)
put(chatter(.28, 1200, 3600), 2.00, .09, .3, .1)
for k in range(1, 25):
    tk = .75 * (1 - (1 - k / 24) ** (1 / 3))
    put(tick(1100 + k * 45, .05, .4), 2.0 + tk, .09, -.6 + 1.2 * k / 24, .15)

section('countdown')
# 2.96 - 6.02 : light still travelling - accelerating clock + photon shimmer + pulse
tcur = 2.98; k = 0
while tcur < 6.0:
    put(tick(2100 if k % 2 else 1600, .04, .6), tcur, .1 + .05 * (tcur - 3) / 3, .25 if k % 2 else -.25, .12)
    tcur += 1 / (5.5 + 4.5 * ((tcur - 2.96) / 3.06) ** 1.5); k += 1
put(shimmer(1.38, 28, 4000, 10000), 2.96, .1, np.linspace(-.8, .8, ns(1.38)), .4)
put(tvf(white(1.38), 6000, 'bp', 2) * np.linspace(.4, 1, ns(1.38)), 2.96, .05, np.linspace(-.6, .7, ns(1.38)), .3)
for tb in np.arange(3.0, 5.5, .62): put(heartbeat(.5), tb, .22, 0, .1)
put(pad([146.8, 220, 293.7, 370], 1.3, 1100, .4, .5), 4.30, .07, 0, .5)  # calm "nothing changes"
for tb in [5.55, 5.95]: put(heartbeat(.7), tb, .42, 0, .15)
put(reverse_swell(.5, 4000), 6.02 - .5, .32, 0, .2)

section('last light')
put(impact(.9), 6.02, .5, 0, .3)
put(shimmer(.7, 60, 5000, 12000), 6.02, .12, np.linspace(-.5, .5, ns(.7)), .5)
put(whoosh(1.3, 6000, 2500, 120, .12, .8), 6.04, .3, np.linspace(-.9, .9, ns(1.3)), .35)
put(power_down(1.4, 520, 45), 6.1, .26, 0, .3)
for tm in np.sort(rng.uniform(6.6, 7.4, 22)): put(click(.004, 3500), tm, .05, rng.uniform(-.7, .7), .3)

section('cold')
w = wind(11.6, 520, 320, .6, .2); w2 = wind(11.6, 560, 340, .55, .15)
env = curve(11.6, [(0, 0), (.6, .8), (2.0, .55), (9.5, .6), (10.6, .45), (11.6, 0)])
put2(w * env, w2 * env, 7.45, .26, .15)
put(drone(4.0, (36.7, 55.0), .25), 7.5, .17, 0, .2)
put(shimmer(2.0, 9, 2500, 6000), 7.6, .1, 0, .5)
for tb in [7.69, 9.23]: put(breath(1.0), tb, .05, -.5, .2)
# 9.51 hour one
put(impact(.8), 9.51, .5, 0, .25); put(chatter(.25), 9.51, .08, -.3, .1); put(tick(1500, .06), 9.62, .12, 0, .2)
# 10.32 temperature falling
t = tt(.95); f = 950 * (300 / 950) ** (t / .95)
put(fade(sine(f, .95) * (.6 + .4 * np.sin(2 * np.pi * 9 * t)) * np.exp(-t / .7), .01, .1), 10.33, .1, 0, .4)
put(fade(sine(f * 1.5, .95) * np.exp(-t / .5), .01, .1), 10.33, .04, .3, .4)
put(chatter(.55, 600, 1800), 10.38, .05, .4, .1)
# 11.20 frost crystals growing
put(crackle(.94, np.linspace(80, 900, ns(.94)), 2500, 12000, .0009), 11.2, .36, 0, .3, width=1)
put(whoosh(.9, 300, 1500, 250, .4, .8), 11.25, .12, np.linspace(.6, -.6, ns(.9)), .3)
put(shimmer(.94, 45, 5000, 11000), 11.2, .13, 0, .45)

section('solar')
dd = 1.71; t = tt(dd); f = 60 * np.where(t < 1.31, 1 - .55 * (t / 1.31) ** 2.5, .45)
h = hum(dd, f) * np.clip((1.31 - t) / .25, 0, 1); put(fade(h, .02, .02), 12.14, .2, 0, .1, width=1)
for tm in np.sort(.12 + 1.15 * rng.random(46) ** .8): put(click(.006, 1800), 12.14 + tm, .11, rng.uniform(-.8, .8), .2)
put(power_down(1.2, 260, 30), 12.3, .14, 0, .2)
put(impact(1.0), 13.45, .55, 0, .2); put(sub_drop(1.0, 80, 30), 13.45, .35, 0, .1)
put(chatter(1.2, 500, 1400), 12.2, .035, .5, .05)

section('city')
for tm in np.sort(.02 + .95 * rng.random(110) ** 1.3): put(click(.005, 2200), 13.85 + tm, .06, rng.uniform(-.9, .9), .25)
t = tt(2.24); put(fade(hum(2.24, 60 * (1 + .12 * t / 2.24)) * np.clip(t / 1.0, 0, 1), .05, .05), 13.85, .1, 0, .1, width=1)
for tm in [14.3, 14.75]: put(whoosh(.9, 200, 900, 150, .5, .7), tm, .06, np.linspace(-.8, .8, ns(.9)), .1)

section('grid overload')
put(riser(1.0, 90, 900, .5), 15.0, .3, 0, .1)
t = tt(1.09); put(fade(sat(hum(1.09, 60 * (1 + .15 * t)) * (.4 + .6 * t / 1.09), 2.5), .02, .05), 15.0, .22, 0, .1, width=1)
put(crackle(1.09, np.linspace(5, 70, ns(1.09)), 1500, 10000, .003), 15.0, .4, 0, .3, width=1)
for tm in [15.55, 15.72, 15.9]:
    put(impact(.25, .5, 2) * .5 + crackle(.5, 900, 2000, 12000, .001), tm, .3, rng.uniform(-.6, .6), .3)
al = np.concatenate([beep(1040, .11, .5, True), beep(780, .11, .5, True)] * 3); put(al, 15.6, .11, .2, .2)

section('hour 12')
put(impact(.8), 16.09, .5, 0, .25); put(reverse_swell(.6, 5000), 16.09 - .6, .2, 0, .2)
put(tick(1500, .06), 16.2, .12, 0, .2)
for k, tm in enumerate(np.arange(16.3, 17.0, .25)): put(tick(1900 if k % 2 else 1400, .04), tm, .06, .3 if k % 2 else -.3, .1)

section('crops')
d = 1.0; t = tt(d); g = .5 + .5 * smooth_noise(d, 3)
rust = bp(white(d), 1800, 7000) * np.clip(smooth_noise(d, 30) + .3, 0, 1.2) * g
put2(rust, bp(white(d), 1800, 7000) * np.clip(smooth_noise(d, 30) + .3, 0, 1.2) * g, 17.02, .32, .2)
put(fade(wind(1.0, 650, 300, 2.0, .2), .1, .2), 17.0, .26, 0, .2, width=1)
put(crackle(1.0, 120, 3000, 12000, .0008), 17.02, .11, 0, .3, width=1)
put(beep(660, .25, .2), 17.12, .07, 0, .3); put(beep(440, .3, .2), 17.32, .07, 0, .3)  # photosynthesis down

section('ecosystem')
put(flutter(1.01, 50), 18.0, .32, np.linspace(-.5, .5, ns(1.01)), .25, width=1)
for tm in np.sort(rng.uniform(18.0, 18.95, 16)): put(chirp(rng.uniform(.05, .09)), tm, .09, rng.uniform(-.8, .8), .4)
for tm in np.arange(18.05, 18.54, 1 / 1.8): put(beep(1000, .08, .1), tm, .1, .35, .2)
for tm in np.arange(18.6, 19.0, 1 / 3.2): put(beep(1000, .07, .1), tm, .07, .35, .2)
put(impact(.9), 18.54, .55, 0, .25)
for tm in [18.6, 18.88]: put(heartbeat(.7), tm, .3, 0, .1)

section('thermal crowd')
put(beep(1750, .09, .1), 19.05, .06, .6, .1)
put(footsteps(1.4, 26), 19.01, .2, 0, .2, width=1)
put(walla(1.4, 8) * np.clip(tt(1.4) / .2, 0, 1), 19.01, .07, 0, .5, width=1)
put(fade(sine(2200, 1.4) * .3 + sine(4400, 1.4) * .1, .05, .05), 19.01, .008, .7, 0)
put(pad([65.4, 98, 130.8, 196], 1.2, 600, .25, .4, sub=65.4), 19.6, .12, 0, .4)
put(whoosh(.7, 120, 800, 100, .6, .7), 20.0, .12, 0, .3)

section('emergency')
put(eas(.5), 20.45, .15, 0, .15); put(eas(.32), 20.98, .15, 0, .15)
put(siren(1.32, 1.3), 20.41, .08, np.sin(np.linspace(0, 2 * np.pi, ns(1.32))) * .7, .6)
put(chatter(.9, 1500, 4000), 20.6, .03, .7, .05)
for k in range(4): put(tick(2400, .03, .3), 20.6 + k * .12, .05, -.3, .1)
# PANIC
put(impact(1.6, 2.4), 21.40, .95, 0, .35); put(glitch(.32), 21.40, .3, .3, .1); put(glitch(.25), 21.42, .22, -.3, .1)
put(sat(impact(.8) * 2.5, 3), 21.40, .25, 0, .1)
put(reverse_swell(.35, 6000), 21.40 - .35, .25, 0, .1)

section('rationing')
for k in range(1, 5):
    tm = 21.73 + k / 3.4
    put(impact(.35, .6) * .7, tm, .25, .4 if k % 2 else -.4, .2)
    put(power_down(.25, 300, 60), tm, .1, .4 if k % 2 else -.4, .2)
for k, n in enumerate([1046.5, 880, 698.5, 587.3]): put(beep(n, .14, .25), 21.73 + (k + 1) / 4.2, .1, 0, .25)
for tm in np.arange(22.7, 22.9, .1): put(beep(440, .05, .3, True), tm, .07, 0, .2)

section('danger')
dr = drone(1.6, (36.7, 38.1, 55.0), .45, 160) * np.clip(tt(1.6) / .3, 0, 1)
put(fade(dr, .02, .2), 22.91, .32, 0, .2)
put(impact(.7), 22.91, .45, 0, .3)
put(reverse_swell(.5, 1500), 23.48 - .5, .3, 0, .2)
put(braam(2.0), 23.48, .85, 0, .35)
put(braam(1.6, 41.2), 23.85, .75, 0, .35)
put(impact(1.2), 23.85, .55, 0, .3)

section('gravity')
d = 1.4; t = tt(d)
gh = sine(42 * (1 + .06 * np.sin(2 * np.pi * .9 * t)), d) + .5 * sine(63 * (1 + .05 * np.sin(2 * np.pi * .7 * t)), d)
put(fade(gh, .05, .05), 24.34, .3, 0, .2)
put(pad([196, 277.2, 349.2, 440, 622.3], 5.5, 1600, .4, 1.0), 24.34, .08, 0, .6)
put(whoosh(.9, 300, 1800, 300, .5, 1.6), 24.75, .14, np.linspace(-.8, .8, ns(.9)), .3)
put(impact(.6), 24.34, .35, 0, .3)
# 25.69 gravity switches off
put(riser(.5, 300, 3000, .7), 25.69 - .5, .3, 0, .2)
put(impact(1.8, 3.0), 25.69, 1.0, 0, .45)
d = 2.0; t = tt(d); wob = sine(48 * (1 + .35 * np.exp(-t / .6) * np.sin(2 * np.pi * 5 * t)), d) * np.exp(-t / .8)
put(fade(sat(wob * 1.5, 1.5), .002, .1), 25.69, .55, 0, .2)
put(shimmer(.8, 80, 6000, 13000), 25.70, .1, 0, .6)
put(chatter(.3, 1500, 4000), 25.72, .08, .4, .1)

section('orbit')
put(shatter(1.6), 26.81, .32, -.3, .5); put(impact(.7), 26.81, .4, 0, .3)
d = 1.5; t = tt(d); dop = 900 * (1.2 - .5 * t / d)
put(fade(tvf(white(d), dop, 'bp', 2.5) * np.exp(-t / .9), .02, .1), 26.85, .14, np.linspace(-.6, .9, ns(d)), .4)
put(beep(1568, .1, .2), 26.86, .06, .6, .2); put(impact(.4), 27.06, .3, 0, .3)

section('drift')
d = 1.43; rush = tvf(pink(d), 400 + 300 * np.sin(np.linspace(0, 3, ns(d))), 'bp', .7)
put2(rush, tvf(pink(d), 450 + 300 * np.cos(np.linspace(0, 3, ns(d))), 'bp', .7), 28.35, .07, .3)
for tm in np.sort(rng.uniform(28.4, 29.6, 7)):
    put(whoosh(.35, 600, 5000, 900, .6, 2), tm, .06, np.linspace(-.9, .9, ns(.35)) * rng.choice([-1, 1]), .3)
put(drone(1.5, (32.7, 49.0), .2), 28.35, .14, 0, .3)

section('final hours')
w = wind(2.4, 480, 260, .7, .25); put(fade(w, .2, .8), 29.78, .14, 0, .2, width=1)
for k, tm in enumerate(np.arange(29.82, 31.32, .25)): put(tick(1700 if k % 2 else 1300, .05, .5), tm, .08, .3 if k % 2 else -.3, .15)
for tm in [30.39, 31.0]: put(bell(110, 2.2, .6), tm, .3, 0, .5); put(impact(.6), tm, .3, 0, .2)

section('darkness')
d = 1.0; t = tt(d); g = np.sin(np.pi * t / d) ** 1.2
put(hp(white(d), 3000) * g * (.5 + .5 * smooth_noise(d, 40)), 31.45, .06, np.linspace(-.8, .8, ns(d)), .4)
put(impact(1.5, 3.0, .3), 31.76, .7, 0, .5); put(sub_drop(2.2, 70, 24), 31.76, .5, 0, .1)

section('waiting')
w = wind(3.0, 380, 160, .3, .08); put(fade(w, .5, .6), 32.6, .06, 0, .3, width=1)
for tm in [32.95, 34.45]: put(breath(1.1), tm, .06, .45, .25)
put(shimmer(2.6, 3, 3000, 7000), 32.7, .05, 0, .8)

section('radar')
for tm in [34.05, 34.95]: put(sonar(1174.66, 2.0), tm, .18, 0, .35)
d = 1.2; st = bp(white(d), 900, 4000) * np.clip(smooth_noise(d, 18) + .2, 0, 1) * .6 + crackle(d, 140, 1000, 6000, .002) * .4
put(fade(st, .05, .1), 33.99, .045, -.4, .3)

section('return')
put(beep(1568, .08, .2), 35.24, .1, 0, .3); put(beep(2093, .1, .2), 35.33, .1, 0, .3)
put(crackle(.6, np.linspace(30, 500, ns(.6)), 800, 7000, .002), 35.25, .22, 0, .3, width=1)
put(riser(.6, 120, 1800), 35.22, .4, 0, .25)
put(reverse_swell(.5, 7000), 35.82 - .5, .35, 0, .2)
put(impact(2.0, 3.5), 35.82, 1.0, 0, .5); put(sub_drop(2.4, 110, 28), 35.82, .7, 0, .1)
r = solar_roar(2.7); put2(r * np.clip(tt(2.7) / .1, 0, 1), np.roll(r, 1200), 35.85, .22, .3)
put(shimmer(1.2, 90, 5000, 13000), 35.84, .12, 0, .6)

section('sunrise')
put(whoosh(.8, 200, 4000, 600, .5, 1), 36.2, .16, 0, .4)
d = 2.3; sr_pad = pad([73.4, 110, 146.8, 185, 220, 329.6, 370], d, 2200, .45, .5, sub=73.4)
put(sr_pad, 36.33, .2, 0, .5)
for k, f in enumerate([587.3, 740, 880, 1174.7, 1480, 1760]): put(bell(f, 1.8), 36.42 + k * .14, .07, -.6 + k * .24, .6)

section('cta')
put(impact(.9), 38.50, .6, 0, .25); put(impact(.5), 38.62, .4, 0, .2)
for i in range(5): put(pop(700 + i * 120, 240 + i * 30, .09), 38.5 + .1 + i * .06 + .12, .18, -.6 + i * .3, .2)
put(pad([196, 246.9, 293.7, 392, 493.9], 5.0, 1300, .8, 1.0, sub=98), 38.5, .1, 0, .5)
put(pop(500, 180, .12), 40.21, .2, 0, .2)
for i in range(29): put(keyclick(), 40.33 + i * (.85 / 29) + rng.uniform(-.006, .006), .1, rng.uniform(-.2, .2), .1)
for tm in np.arange(40.55, 41.35, .05): put(tick(3000, .02, .2), tm, .025, -.4, .05)
put(bell(1318.5, 1.4), 41.04, .16, 0, .5); put(bell(1760, 1.4), 41.12, .12, .2, .5)
put(whoosh(.45, 400, 6000, 900, .7, 1.8), 41.54 - .3, .25, np.linspace(.8, -.8, ns(.45)), .3)
put(shimmer(1.9, 22, 5000, 11000), 41.6, .11, 0, .45)
d = 1.5; put(fade(sine(440 * (2 ** (tt(d) / d)), d) * .3 * np.clip(tt(d) / d, 0, 1), .05, .05), 41.84, .03, .3, .4)
put(riser(1.0, 300, 1400, .6), 41.86, .16, 0, .3)
put(bell(1046.5, 1.2), 42.86, .14, 0, .5); put(bell(1568, 1.2), 42.96, .12, 0, .5)
for k in range(3): put(sonar(1568, .6), 42.86 + k * .21, .03, 0, .5)

section('loop')
put(reverse_swell(.45, 9000), 43.9 - .45, .5, 0, .1)
put(riser(.4, 400, 3200), 43.5, .3, 0, .1)
r = solar_roar(.4); put2(r, np.roll(r, 900), 43.5, .25, .1)

# ------------------------------------------------------------------ space + master
section('reverb + master')
def make_ir(d=2.8, tau=.75):
    n = ns(d); t = np.arange(n) / SR; out = []
    for _ in range(2):
        x = rng.standard_normal(n) * np.exp(-t / tau)
        xl = lp(x, 3200); w = np.clip(t / 1.2, 0, 1); x = x * (1 - w) * .6 + xl * (.4 + .6 * w)
        er = np.zeros(n)
        for tm, g in [(.011, .5), (.019, .4), (.027, .35), (.041, .3), (.053, .22)]:
            er[ns(tm)] += g * rng.choice([-1, 1])
        x = x * .9 + er; x = np.concatenate([np.zeros(ns(.018)), x])
        out.append(x / np.sqrt(np.sum(x ** 2)))
    return out
irL, irR = make_ir()
sndL = hp(sndL, 180, 2); sndR = hp(sndR, 180, 2)
wetL = signal.fftconvolve(sndL, irL)[:N]; wetR = signal.fftconvolve(sndR, irR)[:N]
L = dryL + wetL * 1.6; R = dryR + wetR * 1.6
L = hp(L, 24, 2); R = hp(R, 24, 2)

def limiter(L, R, ceil=.89, look=.003, rel=.12):
    pk = np.maximum(np.abs(L), np.abs(R)); from scipy.ndimage import maximum_filter1d
    pk = maximum_filter1d(pk, ns(look) * 2 + 1)
    g = np.minimum(1, ceil / np.maximum(pk, 1e-9))
    a = np.exp(-1 / (rel * SR))
    gs = signal.lfilter([1 - a], [1, -a], g - 1) + 1          # smooth release
    gs = np.minimum(np.minimum(gs, g), 1.0)                     # instant attack (window already looks ahead)
    return np.tanh(L * gs / ceil * 1.05) * ceil, np.tanh(R * gs / ceil * 1.05) * ceil

target_gain = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
pre = max(np.max(np.abs(L)), np.max(np.abs(R)))
L *= target_gain * .8 / pre; R *= target_gain * .8 / pre
L, R = limiter(L, R)
fo = ns(.004); L[-fo:] *= np.linspace(1, 0, fo); R[-fo:] *= np.linspace(1, 0, fo)

def write24(path, L, R):
    import wave
    st = np.empty(N * 2); st[0::2] = L; st[1::2] = R
    iv = np.clip(np.round(st * 8388607), -8388608, 8388607).astype('<i4')
    b = iv.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(3); w.setframerate(SR); w.writeframes(b)
out = sys.argv[1] if len(sys.argv) > 1 else 'sfx.wav'
write24(out, L, R)
print('wrote', out, f'{DUR}s', 'peak', round(float(max(np.max(np.abs(L)), np.max(np.abs(R)))), 3))
