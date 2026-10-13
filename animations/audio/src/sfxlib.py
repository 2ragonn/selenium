"""Sound-design library: real CC0 recordings + physically-modelled foley (noise-based, not beeps)."""
import numpy as np
from scipy import signal
from instruments import *

def T_(sec): return np.arange(int(sec * SR)) / SR
def mono2(y): return np.vstack([y, y]).astype(np.float32)
def smooth_rand(sec, rate, seed=None, lo=0.0, hi=1.0):
    """slowly varying random curve (for gusts, flicker, wobble)"""
    rg = np.random.default_rng(seed if seed is not None else RNG.integers(1 << 30))
    n = int(sec * SR); k = max(2, int(sec * rate) + 2)
    pts = rg.uniform(lo, hi, k)
    x = np.interp(np.linspace(0, k - 1, n), np.arange(k), pts)
    w = int(SR / max(rate, 0.05) * 0.5)
    if n > w * 2 and w > 1:
        c = np.cumsum(np.concatenate([np.full(w, x[0]), x, np.full(w, x[-1])]))
        x = (c[2 * w:2 * w + n] - c[:n]) / (2 * w)
    return x

def grains(sec, rate, dur=(0.002, 0.01), band=(1500, 8000), amp=(0.2, 1.0), seed=None, decay=True):
    """random noise grains (crackle, paper, debris, pencil)"""
    rg = np.random.default_rng(seed if seed is not None else RNG.integers(1 << 30))
    n = int(sec * SR); y = np.zeros((2, n), np.float32)
    cnt = rg.poisson(rate * sec)
    for _ in range(cnt):
        d = int(rg.uniform(*dur) * SR); i = int(rg.uniform(0, max(1, n - d)))
        g = rg.standard_normal(d).astype(np.float32) * rg.uniform(*amp)
        g *= np.exp(-np.linspace(0, 6, d)) if decay else np.hanning(d)
        r = min(d, 16); g[:r] *= np.linspace(0, 1, r)
        p = rg.uniform(-0.8, 0.8)
        y[0, i:i + d] += g * (1 - p) * 0.7; y[1, i:i + d] += g * (1 + p) * 0.7
    return bp(y, band[0], band[1], 2) if band else y

# ------------------------------------------------------------------ ambience beds
def wind(sec, strength=0.5, tone=500, gust=0.15, seed=None):
    rg = np.random.default_rng(seed)
    n = noise(sec, 'pink', seed)
    g = smooth_rand(sec, gust, seed, 0.25, 1.0) ** 1.6
    low = lp(n, 220, 2) * 1.2
    mid = bp(n, tone * 0.6, tone * 1.6, 2)
    whistle = bp(n, tone * 2.2, tone * 2.6, 2) * 0.35 * smooth_rand(sec, 0.3, (seed or 1) + 9, 0, 1) ** 3
    y = (low + mid * (0.4 + g) + whistle) * g * strength
    return y.astype(np.float32)

def room_tone(sec, hum=50, level=1.0, seed=None):
    n = noise(sec, 'brown', seed); y = lp(n, 180, 2) * 0.6
    t = T_(sec); h = (np.sin(2 * np.pi * hum * t) * 0.006 + np.sin(2 * np.pi * hum * 2 * t) * 0.004)
    return (y + mono2(h)) * level

def city_far(sec, level=1.0, seed=None):
    rg = np.random.default_rng(seed)
    y = lp(noise(sec, 'brown', seed), 350, 2) * 1.2
    # distant traffic swells
    for _ in range(int(sec / 3)):
        st = rg.uniform(0, sec - 3); d = rg.uniform(2, 6)
        s = lp(noise(d, 'pink'), rg.uniform(500, 1200), 2) * np.hanning(int(d * SR)) * rg.uniform(0.05, 0.15)
        i = int(st * SR); m = min(s.shape[1], y.shape[1] - i); y[:, i:i + m] += pan(s, rg.uniform(-0.8, 0.8))[:, :m]
    return y * level

def crickets(sec, n_cr=5, level=1.0, seed=None):
    rg = np.random.default_rng(seed); t = T_(sec); y = np.zeros((2, len(t)), np.float32)
    for k in range(n_cr):
        f = rg.uniform(3900, 5200); rate = rg.uniform(14, 22); chirps = rg.uniform(0.6, 1.4)
        car = np.sin(2 * np.pi * f * t + rg.uniform(0, 6))
        pulse = (np.sin(2 * np.pi * rate * t) > 0.3).astype(np.float32)
        gate = (np.sin(2 * np.pi * chirps * t + rg.uniform(0, 6)) > rg.uniform(-0.2, 0.4)).astype(np.float32)
        e = signal.lfilter([0.02], [1, -0.98], pulse * gate)
        s = (car * e).astype(np.float32) * rg.uniform(0.02, 0.05)
        y += pan(mono2(s), rg.uniform(-0.9, 0.9))
    return bp(y, 2500, 7000, 2) * level

def birds(sec, density=0.5, level=1.0, seed=None):
    rg = np.random.default_rng(seed); y = np.zeros((2, int(sec * SR)), np.float32)
    for _ in range(int(sec * density)):
        st = rg.uniform(0, sec - 1); nnotes = rg.integers(2, 7); base = rg.uniform(2400, 4800); p = rg.uniform(-0.8, 0.8); g = rg.uniform(0.01, 0.04)
        tt = st
        for j in range(nnotes):
            d = rg.uniform(0.04, 0.12); tn = np.arange(int(d * SR)) / SR
            f0 = base * rg.uniform(0.85, 1.25); f1 = f0 * rg.uniform(0.7, 1.5)
            fr = f0 + (f1 - f0) * (tn / d) + 300 * np.sin(2 * np.pi * rg.uniform(30, 70) * tn)
            s = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.sin(np.pi * tn / d) ** 2
            i = int(tt * SR); m = min(len(s), y.shape[1] - i)
            if m > 0: y[:, i:i + m] += pan(mono2(s[:m] * g), p)
            tt += d + rg.uniform(0.02, 0.1)
    return y * level

def fire(sec, level=1.0, seed=None):
    roar = lp(noise(sec, 'brown', seed), 500, 2) * 0.5 * smooth_rand(sec, 1.5, seed, 0.5, 1)
    cr = grains(sec, 25, (0.0008, 0.004), (1200, 9000), (0.1, 1.0), seed)
    return (roar + cr * 0.5) * level

def waves(sec, level=1.0, seed=None):
    rg = np.random.default_rng(seed); n = noise(sec, 'pink', seed)
    sw = smooth_rand(sec, 0.25, seed, 0, 1) ** 2
    y = lp(n, 900, 2) * (0.3 + sw) + hp(n, 2500) * 0.15 * sw ** 3
    return y * level

def drips(sec, rate=0.6, level=1.0, seed=None, verb=0.0):
    rg = np.random.default_rng(seed); y = np.zeros((2, int(sec * SR)), np.float32)
    for _ in range(rg.poisson(rate * sec)):
        st = rg.uniform(0, sec - 0.2); d = 0.12; tn = np.arange(int(d * SR)) / SR
        f = rg.uniform(900, 2200) * (1 + 0.8 * tn / d)
        s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tn * 40) * rg.uniform(0.05, 0.15)
        i = int(st * SR); m = min(len(s), y.shape[1] - i)
        if m > 0: y[:, i:i + m] += pan(mono2(s[:m]), rg.uniform(-0.8, 0.8))
    return y * level

def crowd_walla(sec, voices=14, level=1.0, seed=None):
    """distant crowd murmur: formant-filtered noise with syllabic modulation"""
    rg = np.random.default_rng(seed); n = int(sec * SR); y = np.zeros((2, n), np.float32)
    F = [(500, 1500), (700, 1100), (300, 2300), (400, 2000), (600, 1200)]
    for v in range(voices):
        src = noise(sec, 'pink') [0] * 0.5
        f1, f2 = F[rg.integers(len(F))]
        s = bp(mono2(src), f1 * 0.8, f1 * 1.25, 2)[0] + bp(mono2(src), f2 * 0.85, f2 * 1.2, 2)[0] * 0.6
        syl = (smooth_rand(sec, rg.uniform(3.5, 6), None, 0, 1) ** 2) * (smooth_rand(sec, 0.4, None, 0, 1) > 0.35)
        y += pan(mono2(s * syl * 0.4), rg.uniform(-0.9, 0.9))
    return lp(y, 2500, 2) * level

# ------------------------------------------------------------------ foley one-shots
def creak(dur=0.8, f0=25, f1=70, body=(380, 720, 1300), seed=None, level=1.0):
    """stick-slip creak (door, wood, rope)"""
    rg = np.random.default_rng(seed); n = int(dur * SR); y = np.zeros(n, np.float32)
    t = 0.0
    while t < dur:
        k = t / dur; rate = f0 + (f1 - f0) * (0.5 - 0.5 * np.cos(np.pi * k)) + rg.normal(0, 4)
        i = int(t * SR)
        if i < n: y[i] += rg.uniform(0.5, 1.0)
        t += 1.0 / max(5, rate)
    out = np.zeros(n, np.float32)
    for fb in body:
        b, a = signal.iirpeak(fb * rg.uniform(0.9, 1.1), 12, SR); out += signal.lfilter(b, a, y)
    out *= np.hanning(n) ** 0.3
    return mono2(out * 0.6 * level)

def knock_wood(hard=1.0, pitch=0, room=True):
    """knuckle/fist on a wooden door: real woodblock + low body thump"""
    x = shift(one('woodblock'), -9 + pitch)[:, :int(0.25 * SR)]
    body = lp(shift(one('bdrum_soft'), 2), 300)[:, :int(0.35 * SR)] * 0.7 * hard
    m = max(x.shape[1], body.shape[1]); y = np.zeros((2, m), np.float32)
    y[:, :x.shape[1]] += x * (0.5 + 0.4 * hard); y[:, :body.shape[1]] += body
    return fade(lp(y, 2500 + 2000 * hard), 0.001, 0.08)

def door_slam(hard=1.0):
    """heavy fist pounding on apartment steel/wood door"""
    a = shift(one('bdrum'), -4)[:, :int(0.6 * SR)] * 0.9
    b = shift(one('woodblock'), -14)[:, :int(0.3 * SR)] * 0.6
    c = shift(one('metal'), rnd(-3, 3))[:, :int(0.5 * SR)] * 0.08          # hinge/lock rattle
    d = grains(0.4, 120, (0.001, 0.004), (1500, 6000), (0.1, 0.6)) * 0.3     # frame rattle
    m = int(0.6 * SR); y = np.zeros((2, m), np.float32)
    for s in (a, b, c, d): y[:, :min(m, s.shape[1])] += s[:, :m]
    y = lp(y, 3500 + 2500 * hard) * hard
    return fade(y, 0.001, 0.15)

def footstep(surface='stone', level=1.0):
    th = shift(one('framedrum'), rnd(2, 6))[:, :int(0.18 * SR)] * 0.5
    sc = grains(0.08, 300, (0.002, 0.01), (1500, 6000) if surface != 'bare' else (300, 1500), (0.2, 0.6)) * (0.4 if surface != 'bare' else 0.15)
    m = max(th.shape[1], sc.shape[1]); y = np.zeros((2, m), np.float32); y[:, :th.shape[1]] += th; y[:, :sc.shape[1]] += sc
    if surface == 'bare': y = lp(y, 900)
    return fade(y * level, 0.001, 0.05)

def steps(sec, rate=1.8, surface='stone', level=1.0, run=False):
    out = np.zeros((2, int(sec * SR) + SR), np.float32); t = 0.0; k = 0
    while t < sec:
        s = footstep(surface, level * (1.0 if k % 2 else 0.8) * (1.3 if run else 1))
        i = int(t * SR); out[:, i:i + s.shape[1]] += pan(s, -0.15 if k % 2 else 0.15)
        t += 1.0 / rate * rnd(0.92, 1.08); k += 1
    return out[:, :int(sec * SR)]

def gunshot(dist=1.0, seed=None):
    """layered real snare + bass drum crack + noise, urban slap-back echoes; dist 0(close)..1(far)"""
    rg = np.random.default_rng(seed)
    cr = shift(one('snare'), rg.uniform(-5, -1))[:, :int(0.25 * SR)] * 1.0
    bd = shift(one('bdrum'), -2)[:, :int(0.5 * SR)] * 0.8
    nz = noise(0.06, 'white') * np.exp(-np.linspace(0, 8, int(0.06 * SR))) * 1.2
    m = int(0.6 * SR); y = np.zeros((2, m), np.float32)
    for s in (cr, bd, nz): y[:, :min(m, s.shape[1])] += s[:, :m]
    y = np.tanh(y * 2.5) * 0.6
    # building reflections
    out = np.zeros((2, int(2.0 * SR)), np.float32); out[:, :m] += y
    for k in range(6):
        d = rg.uniform(0.07, 0.6); g = rg.uniform(0.15, 0.45) * (1 - k * 0.1)
        i = int(d * SR); e = lp(y, 2500 - k * 250); out[:, i:i + m] += pan(e, rg.uniform(-0.9, 0.9))[:, :min(m, out.shape[1] - i)] * g
    out = lp(out, 9000 - 7000 * dist) * (1 - 0.6 * dist)
    return fade(out, 0.0005, 0.8)

def explosion_far(level=1.0):
    a = one('bd_vcsl'); b = lp(noise(3.0, 'brown'), 300) * np.exp(-np.linspace(0, 4, int(3.0 * SR))) * 2
    m = max(a.shape[1], b.shape[1]); y = np.zeros((2, m), np.float32); y[:, :a.shape[1]] += lp(a, 600); y[:, :b.shape[1]] += b
    return fade(y * level, 0.002, 1.0)

def whoosh(dur=0.5, f0=300, f1=3000, level=1.0, rev=False, kind='bp'):
    n = noise(dur, 'pink'); y = sweep_filter(n, f0, f1, kind, blocks=48)
    e = np.sin(np.linspace(0, np.pi, y.shape[1])) ** (1.5 if not rev else 0.7)
    if rev: e = np.linspace(0, 1, y.shape[1]) ** 3
    y = y * e * level
    # doppler-ish stereo movement
    p = np.linspace(-0.7, 0.7, y.shape[1]); a = (p + 1) * np.pi / 4
    return np.vstack([y[0] * np.cos(a) * 1.4, y[1] * np.sin(a) * 1.4]).astype(np.float32)

def static(dur, level=1.0, crackle=1.0, band=(800, 6000)):
    n = bp(noise(dur, 'white'), band[0], band[1], 2) * 0.35
    c = grains(dur, 60 * crackle, (0.0005, 0.003), (500, 9000), (0.3, 1.0)) * 0.6
    return (n + c) * level

def radio_tune(dur, level=1.0):
    t = T_(dur); f = 600 + 2500 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.35 * t + 1.0)) ** 2
    het = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.08 * smooth_rand(dur, 2, None, 0, 1) ** 2
    return (static(dur, 0.8) + mono2(het)) * level

def tone(freqs, dur, level=0.2, on=None, speaker=True):
    """electronic signal tones (phone network / alarms): authentic by nature"""
    t = T_(dur); y = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    if on: y *= (np.mod(t, on[0] + on[1]) < on[0]).astype(np.float32)
    y = mono2(y * level)
    y = fade(y, 0.005, 0.01)
    return bp(y, 300, 3400, 2) * 1.4 if speaker else y

def vibrate(dur=0.45, level=1.0):
    t = T_(dur); m = np.sign(np.sin(2 * np.pi * 165 * t)) * 0.3 + np.sin(2 * np.pi * 330 * t) * 0.2
    rattle = bp(noise(dur, 'white'), 900, 3000) * 0.15 * (np.sin(2 * np.pi * 165 * t) > 0.6)
    y = mono2(m) * 0.5 + rattle
    y = lp(y, 2500) * env_ar(len(t), 0.02, 0.05, 1)
    return y * level

def engine(dur, rpm0=800, rpm1=800, cyl=4, level=1.0, rough=0.08, diesel=False, seed=None):
    """combustion engine: jittered firing pulses -> exhaust resonances"""
    rg = np.random.default_rng(seed); n = int(dur * SR); y = np.zeros(n, np.float32)
    t = 0.0
    while t < dur:
        k = t / dur; rpm = rpm0 + (rpm1 - rpm0) * k
        f = rpm / 60 * cyl / 2
        i = int(t * SR)
        if i < n: y[i] += rg.uniform(0.7, 1.0)
        t += 1.0 / f * (1 + rg.normal(0, rough))
    pulse = signal.lfilter([1], [1, -0.93], y)                  # each firing = decaying pressure wave
    ex = np.zeros(n, np.float32)
    for fr, q, g in ([(60, 3, 1.0), (140, 4, 0.6), (320, 5, 0.3), (700, 6, 0.12)] if not diesel else [(45, 2, 1.0), (110, 3, 0.8), (260, 4, 0.5), (900, 5, 0.25)]):
        b, a = signal.iirpeak(fr, q, SR); ex += signal.lfilter(b, a, pulse) * g
    ex += lp(mono2(rg.standard_normal(n).astype(np.float32))[0], 400) * 0.03 * (1 + diesel)
    ex = np.tanh(ex * 3) * 0.4
    return mono2(ex) * level

def starter(dur=1.3, catch=False, level=1.0):
    """starter motor cranking: whine modulated by compression strokes"""
    t = T_(dur); crank = 4.5 + 1.5 * (t / dur)
    whine = np.sin(2 * np.pi * np.cumsum(220 + 30 * np.sin(2 * np.pi * crank * t)) / SR) * 0.25
    comp = (0.55 + 0.45 * np.sin(2 * np.pi * crank * t)) ** 3
    gr = bp(noise(dur, 'white'), 400, 2500, 2)[0] * 0.2
    y = mono2((whine + gr) * comp)
    y = lp(y, 3000) * env_ar(len(t), 0.03, 0.15)
    return y * level

def horn(dur=0.4, f=(410, 515), level=1.0):
    t = T_(dur); y = sum(signal.square(2 * np.pi * fr * t, 0.4) for fr in f) / len(f)
    y = bp(mono2(y.astype(np.float32)), 300, 3000, 2) * 0.25
    return fade(np.tanh(y * 2) * level, 0.01, 0.05)

def helicopter(dur, level=1.0):
    t = T_(dur); rate = 5.3
    chop = (0.4 + 0.6 * np.abs(np.sin(np.pi * rate * t)) ** 6)
    body = lp(noise(dur, 'pink'), 600, 2) * chop
    whine = mono2(np.sin(2 * np.pi * 2300 * t) * 0.01)
    return (body + whine) * level

def jet(dur, level=1.0):
    t = T_(dur); n = noise(dur, 'pink')
    y = lp(n, 2500, 2) * 0.6 + hp(n, 3000) * 0.2
    wh = mono2(np.sin(2 * np.pi * np.cumsum(1800 - 300 * t / dur) / SR) * 0.03 + np.sin(2 * np.pi * np.cumsum(3600 - 600 * t / dur) / SR) * 0.015)
    return (y + wh) * env_ar(len(t), dur * 0.5, dur * 0.4) * level

def breath(dur=1.5, level=1.0, rate=0.9, panting=False):
    t = T_(dur); r = rate if not panting else 3.2
    e = np.abs(np.sin(np.pi * r * t)) ** (2 if not panting else 1.2)
    n = bp(noise(dur, 'pink'), 500, 2600 if not panting else 3500, 2)
    return n * e * 0.3 * level

def growl(dur=1.5, pitch=-14, level=1.0, dog=False):
    src = one('tbn_buzz_pool') if False else inst('tbn_buzz').render(int(rnd(41, 50)), dur, 0.8)
    y = shift(src, pitch)[:, :int(dur * SR)]
    t = T_(y.shape[1] / SR)
    trem = 0.6 + 0.4 * np.sin(2 * np.pi * (24 if dog else 9) * t + np.sin(2 * np.pi * 3 * t))
    y = lp(y * trem, 900 if dog else 1600, 2)
    y = y + bp(noise(y.shape[1] / SR, 'pink'), 300, 1500)[:, :y.shape[1]] * 0.15 * trem
    y = np.tanh(y * 2.5) * 0.5
    return fade(y * level, 0.08, 0.3)

def zgroan(dur=None, pitch=None, level=1.0):
    """zombie groan: real trombone fall / didgeridoo, pitched down, formant-filtered, saturated"""
    c = RNG.integers(3)
    if c == 0: x = inst('tbn_fall').render(int(rnd(55, 66)), 1.0, 0.8)
    elif c == 1: x = inst('tbn_buzz').render(int(rnd(43, 55)), rnd(0.8, 1.6), 0.7)
    else: x = one('didge')
    x = shift(x, pitch if pitch is not None else rnd(-11, -6))
    if dur: x = x[:, :int(dur * SR)]
    y = bp(x, 180, 1800, 2) + peak(x, 650, 8, 2)[:, :x.shape[1]] * 0.3
    y = np.tanh(y * 3) * 0.4
    return fade(y * level, 0.05, 0.3)

def horde(dur, n=10, level=1.0):
    out = np.zeros((2, int(dur * SR) + 3 * SR), np.float32)
    for k in range(n):
        g = zgroan(); t = rnd(0, max(0.01, dur - 0.5)); i = int(t * SR)
        out[:, i:i + g.shape[1]] += pan(g, rnd(-0.9, 0.9))[:, :out.shape[1] - i] * rnd(0.3, 0.8)
    return out[:, :int(dur * SR)] * level

def glass(level=1.0, n=1):
    out = np.zeros((2, int(1.5 * SR)), np.float32)
    for k in range(n):
        g = one('glass')[:, :int(1.2 * SR)] * rnd(0.6, 1.0)
        i = int(k * rnd(0.01, 0.08) * SR); out[:, i:i + g.shape[1]] += pan(g, rnd(-0.6, 0.6))
    return out * level

def metal_hit(level=1.0, pitch=0, dur=1.2):
    return fade(shift(one('metal'), pitch)[:, :int(dur * SR)] * level, 0.001, 0.3)

def paper(dur=0.4, level=1.0):
    return grains(dur, 400, (0.003, 0.02), (1200, 9000), (0.1, 0.8)) * np.hanning(int(dur * SR)) * level

def pencil(dur=1.0, level=1.0):
    y = grains(dur, 900, (0.002, 0.006), (2500, 9000), (0.1, 0.35))
    strokes = (smooth_rand(dur, 12, None, 0, 1) > 0.35).astype(np.float32)
    return y * signal.lfilter([0.01], [1, -0.99], strokes) * 3 * level

def click(level=1.0, f=3500):
    return fade(bp(noise(0.012, 'white'), f * 0.6, f * 1.4) * np.exp(-np.linspace(0, 9, int(0.012 * SR))) * level, 0.0005, 0.004)

def chimes(notes, level=1.0, name='marimba', gap=0.09):
    out = np.zeros((2, int((len(notes) * gap + 1.6) * SR)), np.float32)
    for k, n in enumerate(notes):
        y = inst(name).render(n, 1.2, 0.6); i = int(k * gap * SR); out[:, i:i + y.shape[1]] += y[:, :out.shape[1] - i]
    return out * level

def bubbles_boil(dur, level=1.0):
    y = grains(dur, 70, (0.004, 0.02), (300, 2000), (0.2, 1.0), decay=False)
    b = one('bubbles')[:, :int(dur * SR)]
    out = np.zeros((2, int(dur * SR)), np.float32); out += y; out[:, :b.shape[1]] += b * 0.3
    return out * level

def spark(dur=0.4, level=1.0):
    return grains(dur, 300, (0.0004, 0.002), (2000, 12000), (0.3, 1.0)) * np.exp(-np.linspace(0, 5, int(dur * SR))) * level

def thunk(level=1.0, f=90):
    t = T_(0.4); y = np.sin(2 * np.pi * f * t * (1 - 0.3 * t)) * np.exp(-t * 14)
    return mono2(y * level) + click(level * 0.5, 1200)[:, :0].sum() * 0

def distant(x, amount=0.7):
    return lp(x, 6000 - 4500 * amount) * (1 - 0.5 * amount)

def phone_speaker(x):
    return np.tanh(bp(x, 350, 3400, 2) * 2) * 0.6
