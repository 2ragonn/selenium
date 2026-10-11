#!/usr/bin/env python3
"""
Sound-effects track for shorts/what-if-animals-were-intelligent.html

Every sound is synthesised from scratch (no samples, no licensing issues) and
placed on the same timeline as the animation, so sample 0 == animation time 0.
Cue times are taken from the SHOTS / kline / event timings in the HTML.

Output: 48 kHz, 16-bit, stereo WAV, 55.6 s, about -19 LUFS integrated with a
-1 dBFS peak ceiling so the voice-over sits on top.

Usage:  python3 generate_sfx.py [output.wav]
Needs:  numpy, scipy
"""
import sys
import wave

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

SR = 48000
DUR = 55.6
N = int(SR * DUR)
TAU = 2 * np.pi
rng = np.random.default_rng(20261011)


# ============================================================ helpers
def tt(d):
    return np.arange(int(round(d * SR))) / SR


def noise(d):
    return rng.standard_normal(int(round(d * SR)))


def norm(x, p=1.0):
    m = np.max(np.abs(x))
    return x * (p / m) if m > 0 else x


def filt(x, kind, f, order=2):
    return signal.sosfilt(signal.butter(order, f, kind, fs=SR, output='sos'), x)


def fade(x, a=0.002, r=0.02):
    x = np.array(x, dtype=float)
    na, nr = min(len(x), max(1, int(a * SR))), min(len(x), max(1, int(r * SR)))
    x[:na] *= np.linspace(0, 1, na)
    x[-nr:] *= np.linspace(1, 0, nr)
    return x


def phase(f):
    return TAU * np.cumsum(f) / SR


def svf(x, fc, q=1.0, mode='bp'):
    """TPT state-variable filter with per-sample cutoff (for sweeps)."""
    fc = np.broadcast_to(np.clip(fc, 20, SR * 0.45), x.shape)
    g = np.tan(np.pi * fc / SR)
    k = 1.0 / q
    a1 = 1 / (1 + g * (g + k))
    a2 = g * a1
    a3 = g * a2
    xs, A1, A2, A3 = x.tolist(), a1.tolist(), a2.tolist(), a3.tolist()
    y = [0.0] * len(xs)
    ic1 = ic2 = 0.0
    m = {'bp': 0, 'lp': 1, 'hp': 2}[mode]
    for n in range(len(xs)):
        v3 = xs[n] - ic2
        v1 = A1[n] * ic1 + A2[n] * v3
        v2 = ic2 + A2[n] * ic1 + A3[n] * v3
        ic1 = 2 * v1 - ic1
        ic2 = 2 * v2 - ic2
        y[n] = v1 if m == 0 else (v2 if m == 1 else xs[n] - k * v1 - v2)
    return np.array(y)


def env_pts(d, pts):
    """Piecewise-linear envelope from [(u, level), ...] with u in 0..1."""
    u = tt(d) / d
    return np.interp(u, [p[0] for p in pts], [p[1] for p in pts])


# ============================================================ mix bus
dry = np.zeros((2, N))
send = np.zeros((2, N))


def place(x, t, g=1.0, pan=0.0, verb=0.2):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        if np.ndim(pan) and len(pan) != len(x):
            pan = np.interp(np.linspace(0, 1, len(x)), np.linspace(0, 1, len(pan)), pan)
        p = np.clip(np.broadcast_to(pan, x.shape), -1, 1)
        a = (p + 1) * np.pi / 4
        st = np.vstack([x * np.cos(a), x * np.sin(a)]) * np.sqrt(2)
    else:
        st = x
    i0 = int(round(t * SR))
    if i0 < 0:
        st, i0 = st[:, -i0:], 0
    n = min(st.shape[1], N - i0)
    if n <= 0:
        return
    dry[:, i0:i0 + n] += st[:, :n] * g
    if verb:
        send[:, i0:i0 + n] += st[:, :n] * g * verb


# ============================================================ sound design
def hit(size='med', tone=1.0, dark=False):
    """Trailer impact: pitched sub drop + body + crack + metallic ring."""
    d = {'huge': 3.2, 'big': 2.2, 'med': 1.3, 'small': 0.6}[size]
    t = tt(d)
    f = 31 * tone + 140 * tone * np.exp(-t * 20)
    dec = {'huge': 1.4, 'big': 2.2, 'med': 3.4, 'small': 8.0}[size]
    sub = np.tanh(1.8 * np.sin(phase(f))) * (1 - np.exp(-t * 900)) * np.exp(-t * dec)
    body = norm(filt(noise(d), 'lowpass', 1400 if dark else 2600)) * np.exp(-t * 24)
    crack = norm(filt(noise(d), 'highpass', 3000)) * np.exp(-t * 80)
    x = sub + 0.5 * body + (0.15 if dark else 0.32) * crack
    if not dark:
        for fr, dc, a in [(1210, 14, 0.16), (2710, 22, 0.1), (4320, 30, 0.07), (6400, 40, 0.04)]:
            x += a * np.sin(TAU * fr * tone * t + rng.random() * TAU) * np.exp(-t * dc)
    return fade(norm(x), 0.0004, 0.05)


def text_pop(bright=1.0):
    """Light caption pop: thump + airy tick."""
    d = 0.18
    t = tt(d)
    th = np.sin(phase(80 + 120 * np.exp(-t * 45))) * np.exp(-t * 38)
    air = norm(filt(noise(d), 'bandpass', [2500, 9000])) * np.exp(-t * 55) * (1 - np.exp(-t * 3000))
    return fade(norm(th + 0.45 * bright * air), 0.0004, 0.02)


def whoosh(d=0.3, f0=500, f1=5000, q=1.3, peak=0.72, air=0.35):
    t = tt(d)
    u = t / d
    x = svf(noise(d), f0 * (f1 / f0) ** u, q, 'bp')
    x = norm(x) + air * norm(filt(noise(d), 'highpass', 5500))
    env = np.where(u < peak, (u / peak) ** 2.2, np.exp(-(u - peak) / (1 - peak) * 5))
    return fade(norm(x * env), 0.001, 0.01)


def riser(d=1.0, f0=120, f1=3500, tone=0.5):
    t = tt(d)
    u = t / d
    bp = svf(noise(d), f0 * (f1 / f0) ** (u ** 1.4), 3.0, 'bp')
    fr = 110 * 8 ** (u ** 1.6)
    tn = sum(np.sin(phase(fr * r)) for r in (1.0, 1.007, 1.5, 2.003))
    x = norm(bp) + tone * norm(tn)
    return fade(norm(x * u ** 2.4), 0.01, 0.004)


def reverse_swell(d=0.5):
    t = tt(d)
    u = t / d
    x = norm(filt(noise(d), 'highpass', 2500)) * np.exp((u - 1) * 7)
    return fade(norm(x), 0.01, 0.003)


def mclick(f=2400, d=0.12, low=0.0):
    t = tt(d)
    x = norm(filt(noise(d), 'highpass', 2000)) * np.exp(-t * 400)
    for r, dc, a in [(1, 60, 0.5), (2.32, 80, 0.35), (4.25, 110, 0.25), (6.63, 140, 0.15)]:
        x += a * np.sin(TAU * f * r * t) * np.exp(-t * dc)
    if low:
        x += low * np.sin(phase(60 + 160 * np.exp(-t * 60))) * np.exp(-t * 30)
    return fade(norm(x), 0.0003, 0.01)


def tick(f=3200, d=0.035):
    t = tt(d)
    x = np.sin(TAU * f * t) * np.exp(-t * 260) + 0.5 * norm(filt(noise(d), 'highpass', 4000)) * np.exp(-t * 900)
    return fade(norm(x), 0.0002, 0.005)


def beep(f, d=0.08, sq=0.3):
    t = tt(d)
    x = np.sin(TAU * f * t) + sq * (np.sin(3 * TAU * f * t) / 3 + np.sin(5 * TAU * f * t) / 5)
    return fade(norm(x), 0.003, 0.02)


def bell(f=880, d=1.6):
    t = tt(d)
    x = 0
    for r, dc, a in [(1, 2.4, 1), (2.0, 3.5, 0.5), (3.01, 5, 0.35), (4.17, 7, 0.2), (5.43, 9, 0.12)]:
        x = x + a * np.sin(TAU * f * r * t + rng.random() * TAU) * np.exp(-t * dc)
    return fade(norm(x) * (1 - np.exp(-t * 3000)), 0.0005, 0.05)


def clang(f=520, d=2.2):
    """Hammer on anvil (inharmonic modal partials + strike)."""
    t = tt(d)
    x = 0
    for r, dc, a in [(1, 1.6, 1.0), (1.006, 1.7, 0.5), (2.756, 2.4, 0.7), (5.404, 3.5, 0.5), (8.933, 5, 0.35), (13.34, 7, 0.22)]:
        x = x + a * np.sin(TAU * f * r * t + rng.random() * TAU) * np.exp(-t * dc)
    x = x + 0.9 * norm(filt(noise(d), 'highpass', 1500)) * np.exp(-t * 60)
    x = x + 0.7 * np.sin(TAU * 90 * t) * np.exp(-t * 12)
    return fade(norm(x), 0.0003, 0.05)


def sparks(d=0.8, density=420):
    n = int(d * SR)
    imp = np.zeros(n)
    k = rng.poisson(density * d)
    times = rng.random(k) ** 1.8 * d
    for s in times:
        imp[int(s * SR)] += rng.random() * np.exp(-s * 4) * rng.choice([-1, 1])
    kern = rng.standard_normal(int(0.002 * SR)) * np.exp(-np.arange(int(0.002 * SR)) / (0.0004 * SR))
    x = filt(np.convolve(imp, kern)[:n], 'highpass', 2500)
    x = norm(x) + 0.15 * norm(filt(noise(d), 'highpass', 6000)) * np.exp(-tt(d) * 6)
    return fade(norm(x), 0.0005, 0.05)


def flap(d=0.06, lo=300, hi=2200):
    t = tt(d)
    return filt(noise(d), 'bandpass', [lo, hi]) * np.sin(np.pi * t / d) ** 2


def flutter(t0, n, rate, g, pan, verb=0.15, lo=300, hi=2200):
    for k in range(n):
        place(flap(rng.uniform(0.035, 0.07), lo, hi), t0 + k / rate + rng.uniform(-0.006, 0.006),
              g * rng.uniform(0.6, 1.0) * (1 - 0.45 * k / max(1, n)), pan, verb)


def bloop(f0=700, up=True, d=0.12):
    t = tt(d)
    f = f0 * (1 + (0.8 if up else -0.4) * (1 - np.exp(-t * 60)))
    x = np.sin(phase(f)) * np.exp(-t * 35) * (1 - np.exp(-t * 2000))
    return fade(x, 0.0005, 0.01)


def keystroke():
    d = 0.045
    t = tt(d)
    x = 0.6 * np.sin(TAU * rng.uniform(230, 300) * t) * np.exp(-t * 120)
    x = x + norm(filt(noise(d), 'bandpass', [2000, 6500])) * np.exp(-t * 420)
    return fade(norm(x), 0.0002, 0.005)


def glitch(d=0.3):
    n = int(d * SR)
    x = np.zeros(n)
    i = 0
    while i < n:
        seg = int(rng.uniform(0.008, 0.05) * SR)
        typ = rng.integers(0, 4)
        ar = np.arange(seg) / SR
        if typ == 0:
            hold = int(rng.uniform(4, 40))
            s = np.repeat(rng.uniform(-1, 1, seg // hold + 1), hold)[:seg]
        elif typ == 1:
            s = np.sign(np.sin(TAU * rng.uniform(200, 3000) * ar)) * 0.6
        elif typ == 2:
            s = np.zeros(seg)
        else:
            s = np.round(np.sin(TAU * rng.uniform(80, 800) * ar) * 4) / 4
        m = min(seg, n - i)
        x[i:i + m] = s[:m] * rng.uniform(0.4, 1)
        i += seg
    return fade(norm(filt(x, 'lowpass', 9000)), 0.001, 0.005)


def static(d=0.5):
    t = tt(d)
    gate = np.repeat(rng.random(int(d * 40) + 1) > 0.35, SR // 40)[:len(t)]
    x = norm(filt(noise(d), 'highpass', 1200)) * (0.35 + 0.65 * gate)
    return fade(norm(x), 0.005, 0.02)


def zap(d=0.15):
    t = tt(d)
    f = 2400 * np.exp(-t * 25) + 180
    x = 0.5 * np.sign(np.sin(phase(f))) + norm(filt(noise(d), 'highpass', 3000))
    x = np.round(x * np.exp(-t * 20) * 6) / 6
    return fade(norm(filt(x, 'lowpass', 10000)), 0.0005, 0.01)


def crackle(d=0.6, rate=60):
    n = int(d * SR)
    x = np.zeros(n)
    for s in rng.random(int(rate * d)) * d:
        z = zap(rng.uniform(0.02, 0.06)) * rng.uniform(0.3, 1)
        i = int(s * SR)
        m = min(len(z), n - i)
        x[i:i + m] += z[:m]
    return fade(norm(x), 0.002, 0.02)


def dolphin_whistle(d=0.5, f0=3600, f1=7200):
    t = tt(d)
    u = t / d
    f = f0 + (f1 - f0) * np.sin(np.pi * u) ** 1.2 + 300 * np.sin(TAU * 7 * t)
    ph = phase(f)
    x = (np.sin(ph) + 0.15 * np.sin(2 * ph)) * np.sin(np.pi * u) ** 0.6
    return fade(norm(x), 0.005, 0.02)


def click_train(d=0.5, r0=15, r1=110):
    n = int(d * SR)
    x = np.zeros(n)
    s = 0.0
    while s < d:
        x[int(s * SR)] = rng.uniform(0.6, 1)
        s += 1 / (r0 + (r1 - r0) * (s / d))
    kern = rng.standard_normal(30) * np.exp(-np.arange(30) / 6)
    return fade(norm(filt(np.convolve(x, kern)[:n], 'highpass', 2500)), 0.001, 0.01)


def growl(d=0.8, f=72):
    t = tt(d)
    jit = 1 + 0.4 * norm(filt(noise(d), 'lowpass', 30))
    ph = phase(f * jit)
    saw = 2 * ((ph / TAU) % 1) - 1
    src = saw * (0.6 + 0.4 * np.sign(np.sin(TAU * 27 * t))) + 0.5 * norm(filt(noise(d), 'lowpass', 900))
    x = filt(src, 'bandpass', [170, 420]) + 0.6 * filt(src, 'bandpass', [600, 1100])
    return fade(norm(np.tanh(2 * norm(x)) * env_pts(d, [(0, 0), (0.15, 1), (0.8, 0.85), (1, 0)])), 0.01, 0.05)


def howl(d=1.6, base=1.0):
    t = tt(d)
    u = t / d
    f = np.interp(u, [0, 0.25, 0.7, 1], [330, 520, 500, 370]) * base * (1 + 0.012 * np.sin(TAU * 5.5 * t))
    ph = phase(f)
    x = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)
    x = norm(x) + 0.12 * norm(filt(noise(d), 'bandpass', [600, 2400]))
    return fade(norm(filt(x * env_pts(d, [(0, 0), (0.12, 1), (0.75, 0.85), (1, 0)]), 'lowpass', 3200)), 0.01, 0.05)


def caw(d=0.3, f0=560):
    t = tt(d)
    u = t / d
    f = f0 * (1 - 0.22 * u) * (1 + 0.25 * norm(filt(noise(d), 'lowpass', 60)) * 0.12)
    saw = 2 * ((phase(f) / TAU) % 1) - 1
    src = saw + 0.35 * norm(noise(d))
    x = filt(src, 'bandpass', [900, 1300]) + 0.7 * filt(src, 'bandpass', [1700, 2400]) + 0.3 * filt(src, 'bandpass', [2800, 3600])
    x = np.tanh(2 * norm(x) * env_pts(d, [(0, 0), (0.06, 1), (0.6, 0.8), (1, 0)]))
    return fade(norm(x), 0.003, 0.02)


def drum(f0=150, f1=55, d=0.6, snap=0.5):
    t = tt(d)
    x = np.sin(phase(f1 + (f0 - f1) * np.exp(-t * 30))) * np.exp(-t * 7)
    x = x + snap * norm(filt(noise(d), 'lowpass', 1200)) * np.exp(-t * 40)
    return fade(norm(np.tanh(1.4 * x)), 0.0004, 0.03)


def snare(d=0.25):
    t = tt(d)
    x = norm(filt(noise(d), 'bandpass', [1500, 7000])) * np.exp(-t * 22) + 0.5 * np.sin(TAU * 190 * t) * np.exp(-t * 30)
    return fade(norm(x), 0.0003, 0.02)


def thud(f=90, d=0.35):
    return drum(f * 1.8, f * 0.5, d, 0.4)


def knock(f=520, d=0.25):
    t = tt(d)
    x = 0
    for r, dc, a in [(1, 38, 1), (2.57, 60, 0.5), (4.1, 90, 0.3)]:
        x = x + a * np.sin(TAU * f * r * t) * np.exp(-t * dc)
    x = x + 0.6 * norm(filt(noise(d), 'bandpass', [800, 4000])) * np.exp(-t * 250)
    return fade(norm(x), 0.0005, 0.02)


def creak(d=0.5, f0=55, f1=120):
    t = tt(d)
    rate = (f0 + (f1 - f0) * t / d) * (1 + 0.35 * norm(filt(noise(d), 'lowpass', 20)))
    cyc = np.floor(np.cumsum(rate) / SR)
    imp = np.diff(cyc, prepend=0) * rng.uniform(0.5, 1, len(t))
    x = filt(imp, 'bandpass', [480, 720]) + 0.6 * filt(imp, 'bandpass', [1150, 1500]) + 0.3 * filt(imp, 'bandpass', [2300, 2800])
    return fade(norm(x * env_pts(d, [(0, 0), (0.15, 1), (0.85, 0.9), (1, 0)])), 0.005, 0.03)


def shatter(d=1.3):
    n = int(d * SR)
    x = np.zeros(n)
    g = tt(0.35)
    for _ in range(160):
        s0 = rng.random() ** 2.2 * d * 0.6
        gr = np.sin(TAU * rng.uniform(2200, 9500) * g) * np.exp(-g * rng.uniform(14, 60)) * rng.random() * np.exp(-s0 * 3)
        i = int(s0 * SR)
        m = min(len(gr), n - i)
        x[i:i + m] += gr[:m]
    t = tt(d)
    x = norm(x) + 0.6 * norm(filt(noise(d), 'highpass', 3000)) * np.exp(-t * 14) + 0.7 * np.sin(phase(50 + 120 * np.exp(-t * 30))) * np.exp(-t * 6)
    return fade(norm(x), 0.0003, 0.05)


def shimmer(d=1.5, n=40, base=880, scale=(0, 2, 4, 7, 9), octs=(0, 1, 2), dec=(2.5, 6)):
    """Stereo cascade of chime grains (pentatonic)."""
    out = np.zeros((2, int(d * SR) + int(1.6 * SR)))
    g = tt(1.6)
    for _ in range(n):
        s0 = rng.random() * d
        f = base * 2 ** ((rng.choice(scale) + 12 * rng.choice(octs)) / 12)
        gr = np.sin(TAU * f * g) * np.exp(-g * rng.uniform(*dec)) * (1 - np.exp(-g * 800)) * rng.uniform(0.4, 1) * (1 - 0.6 * s0 / d)
        a = (rng.uniform(-0.9, 0.9) + 1) * np.pi / 4
        i = int(s0 * SR)
        out[0, i:i + len(gr)] += gr * np.cos(a)
        out[1, i:i + len(gr)] += gr * np.sin(a)
    return out / np.max(np.abs(out))


def pad(d, notes, attack=0.4, release=0.6, bright=1200):
    t = tt(d)
    x = 0
    for f in notes:
        for det in (-0.004, 0.0, 0.005):
            ph = phase(np.full(len(t), f * (1 + det))) + rng.random() * TAU
            x = x + np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)
    x = filt(x, 'lowpass', bright) * (1 + 0.15 * np.sin(TAU * 0.8 * t))
    e = np.minimum(1, t / attack) * np.minimum(1, (d - t) / release)
    return fade(norm(x * np.clip(e, 0, 1)), 0.01, 0.05)


def neon_hum(d, gate):
    t = tt(d)
    x = sum(np.sin(TAU * 120 * k * t) / k for k in range(1, 9))
    x = norm(filt(x, 'lowpass', 2500)) + 0.1 * norm(filt(noise(d), 'highpass', 4000))
    return x * gate


def door_rumble(d=0.6):
    t = tt(d)
    rough = np.abs(filt(noise(d), 'lowpass', 15)) * 4
    x = norm(filt(noise(d), 'lowpass', 180)) + 0.5 * norm(filt(noise(d), 'bandpass', [200, 600])) * rough
    return fade(norm(x * env_pts(d, [(0, 0), (0.1, 1), (0.8, 0.8), (1, 0)])), 0.01, 0.05)


def error_buzz():
    out = np.zeros(int(0.36 * SR))
    for k, f in enumerate((420, 300)):
        t = tt(0.15)
        x = np.sign(np.sin(TAU * f * t)) * 0.6 + np.sin(TAU * f * 2 * t) * 0.3
        x = fade(filt(x, 'lowpass', 3000), 0.003, 0.02)
        i = int(k * 0.18 * SR)
        out[i:i + len(x)] += x
    return norm(out)


def trumpet(d=0.9, f0=420):
    t = tt(d)
    u = t / d
    f = f0 * np.interp(u, [0, 0.15, 0.7, 1], [0.7, 1.08, 1.0, 0.85]) * (1 + 0.01 * np.sin(TAU * 6 * t))
    saw = 2 * ((phase(f) / TAU) % 1) - 1
    x = filt(saw, 'bandpass', [500, 1800]) + 0.4 * filt(noise(d), 'bandpass', [1000, 3000])
    return fade(norm(np.tanh(2 * norm(x)) * env_pts(d, [(0, 0), (0.1, 1), (0.7, 0.8), (1, 0)])), 0.01, 0.05)


def stab(d=0.45, notes=(220, 261.6, 329.6)):
    t = tt(d)
    x = 0
    for f in notes:
        for det in (-0.006, 0.006):
            x = x + 2 * ((phase(np.full(len(t), f * (1 + det))) / TAU) % 1) - 1
    x = filt(x, 'lowpass', 2800) * np.exp(-t * 5) * (1 - np.exp(-t * 400))
    return fade(norm(x), 0.001, 0.05)


# ---------------- ambience beds (stereo, decorrelated) ----------------
def bed(d, kind):
    t = tt(d)
    out = np.zeros((2, len(t)))
    for ch in range(2):
        if kind == 'wind':
            gust = 0.55 + 0.45 * norm(filt(noise(d), 'lowpass', 0.8))
            x = norm(filt(noise(d), 'lowpass', 520)) * gust + 0.12 * norm(filt(noise(d), 'highpass', 3500))
        elif kind == 'water':
            x = norm(filt(noise(d), 'lowpass', 260)) + 0.25 * norm(filt(noise(d), 'bandpass', [80, 160]))
        elif kind == 'river':
            bab = np.abs(norm(filt(noise(d), 'lowpass', 9)))
            x = norm(filt(noise(d), 'bandpass', [300, 2600])) * (0.4 + bab) + 0.4 * norm(filt(noise(d), 'lowpass', 200))
        elif kind == 'room':
            x = 0.4 * np.sin(TAU * 55 * t) + 0.2 * np.sin(TAU * 110 * t) + norm(filt(noise(d), 'lowpass', 400)) * 0.6
        elif kind == 'crowd':
            x = 0
            for _ in range(7):
                syl = np.clip(np.sin(TAU * rng.uniform(3, 6) * t + rng.random() * TAU), 0, 1) ** 2
                x = x + filt(noise(d), 'bandpass', [rng.uniform(300, 500), rng.uniform(1200, 2600)]) * syl
            x = filt(x, 'lowpass', 2400)
        elif kind == 'rumble':
            x = norm(filt(noise(d), 'lowpass', 110))
        elif kind == 'hum':
            x = np.sin(TAU * 220 * t) * (0.6 + 0.4 * np.sin(TAU * 3 * t)) + 0.5 * np.sin(TAU * 330.5 * t) + 0.2 * norm(filt(noise(d), 'bandpass', [2000, 6000]))
        out[ch] = norm(x)
    return out * np.clip(np.minimum(t / 0.15, (d - t) / 0.2), 0, 1)


def crickets(d):
    t = tt(d)
    gate = ((t * 25) % 1 < 0.5) * ((t * 2.6) % 1 < 0.35)
    return norm(np.sin(TAU * 4600 * t) * gate * (0.7 + 0.3 * np.sin(TAU * 0.5 * t)))


def bird_chirp():
    d = rng.uniform(0.08, 0.16)
    t = tt(d)
    f = rng.uniform(2500, 3800) * (1 + 0.5 * np.sin(np.pi * t / d)) + 300 * np.sin(TAU * 40 * t)
    return fade(np.sin(phase(f)) * np.sin(np.pi * t / d), 0.002, 0.01)


# ============================================================ cue sheet
# shot cut times (from SHOTS in the HTML)
CUTS = [0.00, 1.10, 1.80, 2.98, 3.50, 4.95, 6.12, 7.75, 8.96, 9.50, 10.55, 12.50, 14.20, 14.96, 16.50, 19.20,
        20.94, 22.05, 23.45, 25.30, 27.34, 28.45, 30.36, 32.45, 33.60, 34.60, 36.28, 37.65, 38.65, 39.55,
        41.00, 43.00, 44.60, 46.40, 48.02, 50.00, 51.55, 53.92]
GLITCH_CUTS = {14.20, 20.94, 32.45}

# --- transitions: every cut gets a whoosh into it plus a crisp tick on the frame
for i, c in enumerate(CUTS[1:], 1):
    side = 1 if i % 2 else -1
    if c in GLITCH_CUTS:
        place(glitch(0.22), c - 0.02, 0.22, side * 0.3, 0.05)
    else:
        d = 0.28
        place(whoosh(d, 500, 6000 if i % 3 else 3000), c - d * 0.78, 0.30, np.linspace(-0.8 * side, 0.8 * side, int(d * SR)), 0.15)
    place(tick(rng.uniform(2600, 3800)), c, 0.10, side * 0.2, 0.05)

# --- caption hits (the visual "slams") and lighter caption pops
HITS = [(0.00, 'big'), (0.45, 'med'), (1.80, 'med'), (2.98, 'big'), (4.15, 'med'), (8.10, 'med'), (8.96, 'huge'),
        (9.72, 'small'), (11.98, 'big'), (14.20, 'big', True), (15.60, 'med', True), (18.78, 'big'),
        (19.68, 'med'), (21.30, 'med', True), (29.55, 'big'), (32.85, 'big', True), (33.60, 'med'),
        (35.20, 'med'), (37.95, 'med'), (38.65, 'med', True), (42.22, 'huge', True), (43.45, 'med'),
        (45.25, 'big'), (47.00, 'big'), (48.45, 'med'), (50.40, 'small'), (51.55, 'med'), (52.75, 'small'),
        (53.92, 'med'), (54.06, 'huge')]
HIT_GAIN = {'huge': 0.95, 'big': 0.8, 'med': 0.55, 'small': 0.4}
for h in HITS:
    t0, size = h[0], h[1]
    dark = len(h) > 2
    place(hit(size, tone=rng.uniform(0.92, 1.08), dark=dark), t0, HIT_GAIN[size], 0.0, 0.35 if size in ('big', 'huge') else 0.25)

POPS = [1.10, 2.60, 3.50, 4.95, 6.12, 6.35, 7.75, 8.60, 9.50, 10.55, 11.05, 12.50, 13.10, 14.96, 16.50, 17.35, 19.20,
        20.20, 20.94, 22.05, 22.50, 23.45, 24.14, 25.30, 25.95, 27.34, 27.75, 28.45, 30.36, 30.85, 31.45, 32.45,
        34.60, 36.28, 36.75, 37.65, 39.55, 39.90, 41.10, 43.00, 44.60, 46.40, 48.02, 50.00, 52.15]
for i, p in enumerate(POPS):
    place(text_pop(), p, 0.24, (-0.15, 0.15)[i % 2], 0.12)

# --- A 0.00 hook: eye snaps open, neural crackle, dive into the pupil
place(whoosh(0.35, 3000, 400, 1.0, 0.15), 0.0, 0.35, 0, 0.2)
place(shimmer(1.0, 18, 1320, octs=(0, 1)), 0.05, 0.16, 0, 0.3)
place(crackle(0.8, 30), 0.15, 0.08, 0.3, 0.1)
place(riser(0.32, 300, 6000, 0.6), 0.78, 0.45, 0, 0.1)

# --- B 1.10 silhouettes flicker through the neural net
for k, s in enumerate([1.10, 1.275, 1.45, 1.625]):
    place(zap(0.09), s, 0.22, (-0.5, 0.5, -0.3, 0.3)[k], 0.1)
    place(beep(880 * 2 ** (k * 3 / 12), 0.06), s, 0.08, 0, 0.2)


# --- C 1.80 IQ gauge: needle surges past HUMAN (ticks follow the needle)
def e_elastic(k):
    k = np.clip(k, 0, 1)
    return np.where(k <= 0, 0, np.where(k >= 1, 1, 2 ** (-10 * k) * np.sin((k * 10 - 0.75) * TAU / 3) + 1))


gt = np.arange(1.8, 2.98, 1 / SR)
v = np.clip(e_elastic((gt - 1.85) / 0.95), 0, 1.08)
pos = np.floor(v * 40)
for idx in np.nonzero(np.diff(pos) != 0)[0]:
    place(tick(2400 + 30 * pos[idx]), gt[idx], 0.09, float(np.clip(v[idx] * 1.6 - 0.8, -0.8, 0.8)), 0.05)
f_needle = 160 + 1100 * np.clip(v, 0, 1.1)
whine = np.sin(phase(f_needle)) * np.clip((gt - 1.82) * 8, 0, 1) * np.exp(-np.clip(gt - 2.35, 0, None) * 5)
place(fade(whine, 0.01, 0.1), 1.8, 0.12, 0, 0.2)
place(bell(1318.5, 1.2), 2.22, 0.18, 0.6, 0.35)

# --- D 2.98 HUMANS? neural link arcs between the brains
place(whoosh(0.5, 400, 3500, 2.0, 0.5), 2.98, 0.2, np.linspace(-0.7, 0.7, int(0.5 * SR)), 0.3)
place(crackle(0.45, 50), 3.0, 0.1, 0, 0.2)

# --- E 3.50 one crow: spotlight clunk, night wind, caw, eye flare
place(drum(110, 45, 0.5, 0.8), 3.5, 0.4, 0, 0.3)
place(bed(1.5, 'wind'), 3.5, 0.06, 0, 0)
place(caw(0.28, 560), 3.82, 0.22, 0.2, 0.35)
place(caw(0.3, 520), 4.18, 0.2, 0.2, 0.35)
place(shimmer(0.6, 14, 1760, octs=(0, 1)), 4.15, 0.15, 0.3, 0.3)

# --- F 4.95 watch it: HUD lock-on
place(whoosh(0.35, 1500, 400, 2.0, 0.6), 4.95, 0.18, 0, 0.1)
for k, s in enumerate([5.0, 5.12, 5.21, 5.27]):
    place(beep(1600, 0.05), s, 0.12, 0.2, 0.1)
place(beep(2093, 0.22, 0.2), 5.32, 0.14, 0.2, 0.15)
for k in range(14):
    place(beep(rng.choice([1200, 1500, 1800, 2400]), 0.025, 0.5), 5.45 + k * 0.045 + rng.uniform(0, 0.02), 0.05, rng.uniform(-0.6, 0.6), 0.05)
for s in (5.0, 5.625):
    place(whoosh(0.4, 800, 2500, 3.0, 0.5, 0.0), s, 0.06, np.linspace(-0.5, 0.5, int(0.4 * SR)), 0.1)

# --- G 6.12 three-part lock: dial ratchets decelerate into each lock
LOCKS = [6.5, 6.9, 7.3]
for i, tk in enumerate(LOCKS):
    ts = np.arange(6.12, tk, 1 / 2000)
    th = 7 * (tk - ts) ** 2 + 4 * (tk - ts)
    steps = np.floor(th / (TAU / 24))
    for idx in np.nonzero(np.diff(steps) != 0)[0]:
        place(tick((2600, 3200, 3900)[i], 0.02), ts[idx], 0.07, (-0.35, 0.0, 0.35)[i], 0.03)
    place(mclick(1700 + i * 300, 0.15, low=0.6), tk, 0.42, (-0.35, 0.0, 0.35)[i], 0.15)
    place(bell((659.3, 830.6, 987.8)[i], 0.8), tk + 0.02, 0.14, 0, 0.3)
place(mclick(900, 0.3, low=1.0), 7.42, 0.6, 0, 0.25)
for k, f in enumerate((1318.5, 1661.2, 1975.5, 2637.0)):
    place(bell(f, 0.9), 7.46 + k * 0.045, 0.1, (-0.3, -0.1, 0.1, 0.3)[k], 0.4)
place(whoosh(0.3, 600, 4000, 1.5), 7.4, 0.15, 0, 0.2)
for tk in LOCKS:
    place(whoosh(0.12, 1500, 500, 2.0, 0.5, 0.0), tk - 0.1, 0.05, -0.7, 0)

# --- H 7.75 the cage door creaks, the whole flock bursts out
place(creak(0.4, 50, 110), 7.78, 0.3, -0.3, 0.25)
place(mclick(700, 0.4, low=0.8), 8.15, 0.3, -0.3, 0.3)
place(whoosh(0.9, 300, 5000, 1.0, 0.55, 0.5), 7.85, 0.35, 0, 0.3)
for b in range(110):
    st = 7.87 + b * 0.007
    a = -np.pi / 2 + (rng.random() - 0.5) * 2.4
    flutter(st, rng.integers(4, 9), rng.uniform(10, 15), 0.06 * (1 - 0.5 * b / 110), float(np.cos(a)), 0.15)
place(reverse_swell(0.4), 8.56, 0.3, 0, 0.1)

# --- I 8.96 FREE: murmuration rush + bright air
for b in range(70):
    flutter(8.96 + rng.random() * 0.5, rng.integers(3, 7), rng.uniform(11, 16), 0.03, rng.uniform(-0.9, 0.9), 0.2, 500, 3000)
place(shimmer(0.5, 22, 1760, octs=(0, 1, 2)), 8.97, 0.18, 0, 0.45)

# --- J 9.50 now they talk: chat pops + typing dots
for k, (s, side) in enumerate([(9.58, -1), (9.8, 1), (10.02, -1), (10.24, 1)]):
    for j in range(3):
        place(tick(4200, 0.015), s - 0.17 + j * 0.05, 0.03, side * 0.4, 0.05)
    place(bloop(620 if side < 0 else 820, True), s, 0.35, side * 0.45, 0.15)
    place(tick(5200, 0.02), s + 0.01, 0.06, side * 0.45, 0.05)

# --- K 10.55 beavers engineer cities: blueprint scribbles, logs knock, power-on
place(bed(1.95, 'water'), 10.55, 0.07, 0, 0)
for t0 in (0.2, 0.42, 0.08, 0.5, 0.3, 0.6, 0.65):
    for j in range(3):
        place(knock(rng.uniform(380, 640)), 10.55 + t0 + j * 0.09, 0.22 * (1 - j * 0.25), rng.uniform(-0.7, 0.7), 0.15)
place(riser(1.3, 100, 2500, 0.35), 10.68, 0.25, 0, 0.15)
place(creak(0.6, 40, 70), 10.9, 0.12, 0.4, 0.2)
place(whoosh(0.5, 2000, 6000, 4.0, 0.5, 0.0), 10.6, 0.06, -0.4, 0.05)
place(drum(90, 40, 0.6, 0.2), 11.98, 0.4, 0, 0.2)
place(bed(0.5, 'hum'), 11.98, 0.05, 0, 0.1)
place(shimmer(0.4, 26, 1760, octs=(0, 1, 2), dec=(5, 10)), 11.98, 0.14, 0, 0.3)

# --- L 12.50 octopuses forge tools: underwater, hammer swings, anvil clangs + sparks
place(bed(1.7, 'water'), 12.5, 0.12, 0, 0)
for k in range(10):
    place(bloop(rng.uniform(300, 900), True, 0.08), 12.55 + rng.random() * 1.6, 0.07, rng.uniform(-0.8, 0.8), 0.2)
for k, ti in enumerate([13.05, 13.5, 13.95]):
    place(whoosh(0.2, 400, 1800, 1.5, 0.85, 0.1), ti - 0.19, 0.18, 0.4, 0.05)
    place(clang((560, 590, 530)[k]), ti, 0.6, 0.1, 0.3)
    place(sparks(0.7), ti, 0.22, 0.1, 0.25)

# --- M 14.20 wolves: eyes ignite, distant howl
for k in range(16):
    place(tick(rng.uniform(5000, 7000), 0.02), 14.24 + k * 0.03, 0.04, rng.uniform(-0.8, 0.8), 0.2)
place(howl(1.6), 14.28, 0.22, -0.2, 0.6)
place(howl(1.4, 0.82), 14.5, 0.12, 0.5, 0.7)

# --- N 14.96 organized like armies: war drums + marching
for k in range(7):
    b = 14.96 + k * 0.22
    accent = k in (0, 3, 6)
    place(drum(160 if accent else 130, 52, 0.6, 0.6), b, 0.55 if accent else 0.33, 0, 0.25)
    place(snare(), b + 0.11, 0.12, 0.1, 0.15)
for k in range(12):
    place(thud(rng.uniform(70, 110), 0.18), 14.98 + k * 0.125, 0.08, rng.uniform(-0.6, 0.6), 0.1)
place(bed(1.55, 'rumble'), 14.96, 0.12, 0, 0)

# --- O 16.50 species cards swipe in, stat bars fill, cards fan out
for i in range(6):
    s = 16.5 + i * 0.36
    place(whoosh(0.2, 800, 6000, 1.2, 0.5), s - 0.05, 0.22, np.linspace(0.8, -0.2, int(0.2 * SR)), 0.1)
    gl = np.sin(phase(np.linspace(600, 1500, int(0.2 * SR))))
    place(fade(gl, 0.01, 0.05), s + 0.08, 0.035, 0.2, 0.2)
    place(tick(3000 + i * 200, 0.02), s + 0.3, 0.06, 0.3, 0.1)
for k in range(6):
    place(whoosh(0.08, 2000, 7000, 1.5, 0.5, 0.5), 18.66 + k * 0.03, 0.08, -0.6 + k * 0.24, 0.05)
place(riser(0.5, 200, 4000, 0.5), 18.29, 0.3, 0, 0.1)
place(shimmer(1.0, 24, 1318.5), 18.8, 0.13, 0, 0.4)

# --- P 19.20 pyramid: camera tilts up, wolf / elephant / crow reach the top
place(bed(0.7, 'rumble'), 19.2, 0.15, 0, 0)
place(whoosh(0.35, 300, 2500, 1.2, 0.8), 19.6, 0.18, -0.6, 0.1)
place(thud(85, 0.3), 19.95, 0.35, -0.3, 0.2)
place(whoosh(0.4, 200, 1500, 1.0, 0.8), 19.76, 0.18, 0.6, 0.1)
place(thud(55, 0.5), 20.15, 0.5, 0.3, 0.25)
flutter(19.9, 5, 12, 0.08, 0.4)
place(tick(2800, 0.03), 20.3, 0.08, 0.3, 0.1)
place(shimmer(0.7, 18, 1318.5, octs=(0, 1)), 20.17, 0.12, 0, 0.4)

# --- Q 20.94 confusion: breaking-news stab, static, glitches, question pops
place(stab(0.5, (196.0, 233.1, 293.7)), 20.95, 0.32, 0, 0.3)
place(static(1.1), 20.94, 0.07, 0, 0)
for s in (21.05, 21.3, 21.55, 21.8):
    place(glitch(0.15), s, 0.16, rng.uniform(-0.6, 0.6), 0.05)
for k in range(7):
    place(bloop(rng.uniform(400, 900), rng.random() < 0.5, 0.1), 21.0 + rng.random() * 0.95, 0.12, rng.uniform(-0.8, 0.8), 0.15)

# --- R1 22.05 dolphin ambassador: council hall murmur, tank, whistles
place(bed(1.4, 'crowd'), 22.05, 0.06, 0, 0.2)
place(bed(1.4, 'water'), 22.05, 0.05, 0, 0)
place(dolphin_whistle(0.45, 3400, 6800), 22.25, 0.1, 0.2, 0.3)
place(dolphin_whistle(0.35, 4200, 7400), 22.95, 0.08, 0.2, 0.3)
place(bell(1046.5, 1.4), 22.52, 0.16, 0, 0.4)

# --- R2 23.45 waiting for a translator: sonar clicks, processing, NO MATCH
place(bed(1.85, 'water'), 23.45, 0.09, 0, 0)
place(bed(1.3, 'hum'), 23.5, 0.035, 0, 0.15)
for k in range(3):
    place(click_train(0.4, 20, 140), 23.5 + k * 0.714, 0.12, 0.3, 0.2)
for k in range(18):
    place(beep(rng.choice([900, 1200, 1500, 1800]), 0.03, 0.4), 23.6 + k * 0.065, 0.045, rng.uniform(-0.5, 0.5), 0.05)
for k in range(16):
    place(tick(2200, 0.02), 23.5 + k * 0.08, 0.03, 0.6, 0.05)
place(error_buzz(), 24.75, 0.2, 0, 0.15)
place(error_buzz(), 25.0, 0.12, 0, 0.15)

# --- S 25.30 someone offers a hand: warm swell into a glowing touch
place(pad(1.4, (146.8, 220.0, 293.7, 370.0), 1.0, 0.1, 1600), 25.15, 0.16, 0, 0.3)
place(riser(1.15, 200, 3000, 0.25), 25.35, 0.2, 0, 0.2)
place(reverse_swell(0.6), 25.9, 0.15, 0, 0.2)
place(hit('med', 1.0), 26.5, 0.4, 0, 0.45)
place(shimmer(1.2, 40, 1174.7, (0, 2, 4, 7, 9), (0, 1, 2)), 26.5, 0.25, 0, 0.5)
place(pad(1.0, (293.7, 370.0, 440.0, 587.3), 0.05, 0.6, 2400), 26.5, 0.12, 0, 0.4)
place(dolphin_whistle(0.35, 4500, 7800), 26.62, 0.07, -0.4, 0.4)

# --- T 27.34 years pass: accelerating clock + time-warp
tc, gap, k = 27.36, 0.13, 0
while tc < 28.42:
    place(tick(2200 if k % 2 else 1700, 0.03), tc, 0.12, (-0.3, 0.3)[k % 2], 0.1)
    tc += gap
    gap = max(0.028, gap * 0.9)
    k += 1
place(riser(1.05, 150, 5000, 0.6), 27.36, 0.28, np.linspace(-0.5, 0.5, int(1.05 * SR)), 0.2)
for k in range(3):
    place(whoosh(0.45, 200, 900, 1.0, 0.5, 0.0), 27.34 + k * 0.45, 0.1, 0, 0.1)

# --- U 28.45 treetop towns rise: night forest, houses pop, lanterns twinkle
place(bed(1.91, 'wind'), 28.45, 0.05, 0, 0)
place(crickets(1.9), 28.45, 0.03, 0.4, 0.1)
for k in range(5):
    place(bird_chirp(), 28.5 + rng.random() * 1.8, 0.05, rng.uniform(-0.8, 0.8), 0.4)
for k in range(10):
    s = 28.6 + rng.random() * 1.1
    place(knock(rng.uniform(300, 500), 0.2), s, 0.18, rng.uniform(-0.7, 0.7), 0.2)
    place(bloop(rng.uniform(500, 800), True, 0.08), s + 0.01, 0.08, 0, 0.2)
for k in range(14):
    place(bell(rng.choice([1760, 2093, 2349.3, 2637]), 0.5), 28.9 + rng.random() * 1.2, 0.04, rng.uniform(-0.8, 0.8), 0.4)
place(creak(0.5, 45, 80), 29.05, 0.08, 0.3, 0.2)
place(pad(0.85, (220.0, 277.2, 329.6, 440.0), 0.05, 0.6, 2200), 29.55, 0.12, 0, 0.4)

# --- V 30.36 otters ferry cargo down rivers
place(bed(2.09, 'river'), 30.36, 0.12, 0, 0.05)
for lane, pan in enumerate((-0.2, 0.2, 0.5)):
    s = 30.45 + lane * 0.13
    while s < 32.4:
        x = filt(noise(0.09), 'bandpass', [500, 3500]) * np.exp(-tt(0.09) * 40)
        place(fade(norm(x), 0.002, 0.01), s, 0.06 + 0.03 * lane, pan + rng.uniform(-0.15, 0.15), 0.15)
        s += rng.uniform(0.25, 0.4)
place(creak(0.6, 35, 60), 30.8, 0.08, -0.3, 0.2)
place(creak(0.5, 40, 65), 31.7, 0.07, 0.2, 0.2)
for k in range(4):
    place(knock(rng.uniform(250, 380), 0.2), 30.6 + rng.random() * 1.7, 0.08, rng.uniform(-0.5, 0.5), 0.2)

# --- W 32.45 then come the fights: growl, three claw slashes
place(growl(0.75, 70), 32.45, 0.35, -0.3, 0.15)
for k, s in enumerate([32.55, 32.61, 32.67]):
    place(whoosh(0.14, 9000, 1500, 0.9, 0.35, 0.8), s - 0.03, 0.4, np.linspace(0.7, -0.7, int(0.14 * SR)), 0.2)
    place(sparks(0.25, 900), s + 0.04, 0.15, (-0.2, 0, 0.2)[k], 0.15)
place(mclick(450, 0.35, low=0.8), 32.6, 0.25, 0.5, 0.3)

# --- X 33.60 land rights: paw stamps, flags drop, CONTESTED stamp
for k in range(9):
    place(thud(rng.uniform(80, 120), 0.15), 33.75 + k * 0.07, 0.15, -0.5 + rng.uniform(-0.2, 0.2), 0.1)
for k in range(6):
    s = 33.8 + k * 0.09
    place(whoosh(0.12, 3000, 800, 1.5, 0.8, 0.2), s, 0.08, 0.5, 0.05)
    place(knock(700, 0.12), s + 0.13, 0.12, 0.5, 0.1)
place(drum(120, 45, 0.5, 1.0), 34.25, 0.55, 0, 0.3)
x = norm(filt(noise(0.12), 'bandpass', [800, 3000])) * np.exp(-tt(0.12) * 50)
place(fade(x, 0.0005, 0.02), 34.25, 0.3, 0, 0.2)

# --- Y 34.60 who owns the forest: wind in pines, electric split, owl
place(bed(1.68, 'wind'), 34.6, 0.1, 0, 0)
place(crackle(1.6, 25), 34.62, 0.08, 0, 0.2)
place(whoosh(0.3, 200, 1500, 1.0, 0.7, 0.0), 34.62, 0.2, 0, 0.3)
for s, f in ((35.65, 410), (35.85, 380)):
    ot = tt(0.22)
    place(fade(np.sin(TAU * f * ot) * np.sin(np.pi * ot / 0.22), 0.02, 0.05), s, 0.05, -0.6, 0.6)

# --- Z 36.28 now you're in the room: heavy doors open, hologram hum
place(door_rumble(0.6), 36.28, 0.4, 0, 0.3)
place(whoosh(0.55, 150, 2500, 0.8, 0.6, 0.6), 36.3, 0.3, 0, 0.3)
place(bed(1.37, 'room'), 36.28, 0.06, 0, 0.1)
place(bed(1.0, 'hum'), 36.62, 0.04, 0, 0.2)

# --- AA 37.65 a human general: snare roll into the hit, medal glints
s, k = 37.65, 0
while s < 37.95:
    place(snare(0.12), s, 0.05 + 0.25 * (s - 37.65) / 0.3, 0.1, 0.2)
    s += 0.032
    k += 1
place(drum(140, 50, 0.9, 0.5), 37.95, 0.45, 0, 0.35)
for k in range(5):
    place(bell(rng.choice([2637, 3136, 3520]), 0.4), 38.0 + k * 0.11, 0.05, rng.uniform(-0.6, 0.6), 0.3)

# --- AB 38.65 wolf growl / 39.55 elephant elder rumble + trumpet
place(growl(0.9, 64), 38.66, 0.45, 0.2, 0.2)
t_el = tt(1.45)
rum = np.sin(phase(np.full(len(t_el), 38.0))) + 0.5 * np.sin(phase(np.full(len(t_el), 76.0))) + 0.25 * np.sin(phase(np.full(len(t_el), 114.0)))
place(fade(rum * (0.7 + 0.3 * np.sin(TAU * 1.3 * t_el)), 0.2, 0.3), 39.55, 0.3, 0, 0.1)
place(trumpet(0.9, 410), 39.7, 0.12, 0.3, 0.5)

# --- AC 41.00 cooperate (bright, linked) or compete (slam, crackle, tug)
place(pad(1.2, (261.6, 329.6, 392.0, 523.3), 0.2, 0.4, 2600), 41.0, 0.1, 0, 0.3)
for k in range(8):
    place(beep(1046.5 * 2 ** (rng.choice([0, 4, 7, 12]) / 12), 0.05, 0.1), 41.15 + k * 0.13, 0.04, rng.uniform(-0.5, 0.5), 0.3)
place(glitch(0.25), 42.22, 0.25, 0, 0.05)
place(crackle(0.75, 70), 42.25, 0.14, 0, 0.15)
t_w = tt(0.75)
wob = np.sin(TAU * 46 * t_w) * (0.5 + 0.5 * np.abs(np.sin(TAU * 1.43 * (t_w + 0.28))))
place(fade(np.tanh(2 * wob), 0.02, 0.1), 42.25, 0.22, 0, 0.05)

# --- AD 43.00 they choose to share: warm swell, orb ignites, rays burst
place(pad(1.6, (196.0, 293.7, 392.0, 493.9), 0.6, 0.5, 2000), 43.0, 0.16, 0, 0.35)
place(reverse_swell(0.45), 43.0, 0.15, 0, 0.2)
place(shimmer(1.1, 36, 987.8, (0, 2, 4, 7, 9), (0, 1, 2)), 43.45, 0.18, 0, 0.5)
place(whoosh(0.6, 300, 6000, 1.0, 0.6, 0.5), 43.88, 0.16, 0, 0.3)

# --- AE 44.60 intelligence changes who rules: crown shatters, shards fly
for k in range(6):
    place(bloop(523.3 * 2 ** ((0, 2, 4, 7, 9, 12)[k] / 12), True, 0.1), 44.65 + k * 0.05, 0.08, -0.6 + k * 0.24, 0.3)
place(shimmer(0.6, 12, 1568, octs=(0, 1)), 44.62, 0.08, 0, 0.4)
place(shatter(1.3), 45.25, 0.5, 0, 0.35)
for k in range(6):
    a = -np.pi / 2 + k / 6 * TAU
    place(whoosh(0.4, 3000, 800, 1.5, 0.4, 0.3), 45.3 + k * 0.02, 0.1, float(np.cos(a)), 0.2)

# --- AF 46.40 and how we live together: ring of species, uplifting
for k in range(8):
    place(bell(523.3 * 2 ** ((0, 2, 4, 7, 9, 12, 14, 16)[k] / 12), 0.9), 46.4 + k * 0.06, 0.09, -0.8 + k * 0.23, 0.4)
place(pad(1.62, (261.6, 329.6, 392.0, 493.9, 587.3), 0.4, 0.3, 2400), 46.4, 0.16, 0, 0.35)
place(shimmer(1.0, 40, 1046.5, (0, 2, 4, 7, 9), (0, 1, 2)), 47.0, 0.2, 0, 0.5)

# --- AG 48.02 what's your wildest what if: electric room, neon flickers on
place(bed(1.98, 'hum'), 48.02, 0.03, 0, 0.1)
place(crackle(1.0, 6), 48.1, 0.04, 0.3, 0.2)
nd = 1.0
tn = tt(nd)
step = np.floor(tn * 30)
flick = {int(s): (1.0 if s >= 7 or rng.random() > 0.4 else 0.1) for s in np.unique(step)}
fl = np.array([flick[int(s)] for s in step])
place(neon_hum(nd, filt(fl, 'lowpass', 400)) * np.minimum(1, (nd - tn) / 0.1), 49.0, 0.1, 0, 0.1)
place(crackle(0.26, 60), 49.0, 0.16, 0, 0.1)
place(mclick(600, 0.25, low=0.8), 49.0, 0.3, 0, 0.3)

# --- AH 50.00 drop it in the comments: card pops, typing, send, likes
place(bloop(520, True, 0.12), 50.0, 0.25, 0, 0.15)
for k in range(36):
    place(keystroke(), 50.25 + k * (0.7 / 36) + rng.uniform(-0.004, 0.004), 0.1, 0.1, 0.05)
place(whoosh(0.18, 800, 6000, 1.2, 0.6), 51.0, 0.22, np.linspace(-0.2, 0.7, int(0.18 * SR)), 0.1)
place(bloop(880, True, 0.12), 51.05, 0.3, 0.3, 0.2)
for k in range(6):
    place(bloop(1000 + k * 120, True, 0.06), 51.22 + k * 0.05, 0.06, 0.5, 0.2)
for k in range(3):
    place(bloop(300, False, 0.15), 50.6 + k * 0.45, 0.05, 0.6, 0.1)

# --- AI 51.55 your idea: bulb flickers, ding, morphs into a video player
for k in range(8):
    if rng.random() < 0.6:
        place(zap(0.04), 51.55 + k * 0.04, 0.1, 0, 0.1)
place(bell(1568, 1.6), 51.86, 0.2, 0, 0.4)
place(bed(0.6, 'hum'), 51.86, 0.03, 0, 0.1)
place(whoosh(0.5, 300, 3000, 1.0, 0.8), 52.45, 0.2, 0, 0.2)
place(bloop(660, True, 0.1), 52.95, 0.25, 0.3, 0.15)
place(mclick(2400, 0.08), 52.95, 0.1, 0.3, 0.1)
gl = np.sin(phase(np.linspace(300, 900, int(1.0 * SR)))) * np.linspace(0, 1, int(1.0 * SR))
place(fade(gl, 0.05, 0.1), 52.9, 0.035, 0, 0.2)

# --- AJ 53.92 COMMENT NOW: final slam with a long shimmering tail
place(reverse_swell(0.3), 53.76, 0.2, 0, 0.1)
for k in range(6):
    place(bloop(392.0 * 2 ** ((0, 4, 7, 12, 16, 19)[k] / 12), True, 0.1), 54.0 + k * 0.05, 0.08, -0.6 + k * 0.24, 0.3)
place(shimmer(1.4, 46, 783.99, (0, 2, 4, 7, 9), (0, 1, 2), dec=(1.5, 3.5)), 54.06, 0.2, 0, 0.6)
place(pad(1.5, (196.0, 293.7, 392.0, 493.9), 0.05, 1.2, 2200), 54.06, 0.14, 0, 0.5)


# ============================================================ master
def reverb_ir(d=2.3, pre=0.018):
    n = int(d * SR)
    t = np.arange(n) / SR
    ir = np.zeros((2, n))
    for ch in range(2):
        z = rng.standard_normal(n)
        ir[ch] = filt(z, 'lowpass', 600) * np.exp(-t * 2.4) + 0.8 * filt(z, 'bandpass', [600, 4000]) * np.exp(-t * 3.4) \
            + 0.5 * filt(z, 'highpass', 4000) * np.exp(-t * 7)
        for _ in range(10):
            ir[ch, int(rng.uniform(0.01, 0.08) * SR)] += rng.uniform(-0.6, 0.6) * 20
    ir[:, :int(pre * SR)] = 0
    return ir / np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))


def k_weighted_lufs(x):
    """Integrated loudness per ITU-R BS.1770 (48 kHz coefficients)."""
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]
    y = signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=1), axis=1)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    z = np.array([np.sum(np.mean(y[:, i:i + blk] ** 2, axis=1)) for i in range(0, y.shape[1] - blk, hop)])
    l = -0.691 + 10 * np.log10(z + 1e-12)
    z1 = z[l > -70]
    rel = -0.691 + 10 * np.log10(np.mean(z1)) - 10
    z2 = z1[(-0.691 + 10 * np.log10(z1)) > rel]
    return -0.691 + 10 * np.log10(np.mean(z2))


def limiter(x, ceil=0.891, look=0.003, rel=0.12):
    peak = np.max(np.abs(x), axis=0)
    need = np.minimum(1.0, ceil / np.maximum(peak, 1e-9))
    la = int(look * SR)
    g = minimum_filter1d(need, size=2 * la + 1)
    gl = g.tolist()
    c = np.exp(-1 / (rel * SR))
    out = [0.0] * len(gl)
    y = 1.0
    for i, v in enumerate(gl):
        y = v if v < y else v + (y - v) * c
        out[i] = y
    g = uniform_filter1d(np.array(out), size=la)
    return np.clip(x * g, -ceil, ceil)


def master(target=-19.0):
    wet = np.vstack([signal.fftconvolve(send[ch], IR[ch])[:N] for ch in range(2)])
    mix = dry + 0.85 * wet
    mix = signal.sosfilt(signal.butter(2, 28, 'highpass', fs=SR, output='sos'), mix, axis=1)
    mix *= 0.891 / np.max(np.abs(mix))
    for _ in range(4):
        out = limiter(mix)
        lufs = k_weighted_lufs(out)
        if abs(lufs - target) < 0.2:
            break
        mix *= 10 ** ((target - lufs) / 20)
    fl = int(1.2 * SR)
    out[:, -fl:] *= np.linspace(1, 0, fl) ** 2
    return out, k_weighted_lufs(out)


def write_wav(path, x):
    tpdf = (rng.random(x.shape) - rng.random(x.shape)) / 32768
    pcm = np.clip(np.round((x + tpdf) * 32767), -32768, 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.T.tobytes())


IR = reverb_ir()
if __name__ == '__main__':
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'what-if-animals-were-intelligent-sfx.wav'
    mixed, loud = master()
    write_wav(out_path, mixed)
    print(f'wrote {out_path}: {DUR:.1f}s, {SR} Hz stereo 16-bit, {loud:.1f} LUFS, peak {20 * np.log10(np.max(np.abs(mixed))):.1f} dBFS')
