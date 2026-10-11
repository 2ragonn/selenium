#!/usr/bin/env python3
"""
Procedural sound-design track for "What if Earth's gravity disappeared?"
(shorts/what-if-gravity-disappeared.html).

Every sound is synthesised from scratch (no samples) and placed on the exact
timeline of the animation, so t=0 in the WAV == t=0 in the animation.

Outputs (48 kHz / 24-bit / stereo, 39.62 s):
  what-if-gravity-sfx.wav                 full SFX mix (FX + ambience)
  stems/what-if-gravity-sfx_fx-only.wav   hits, whooshes, UI, foley
  stems/what-if-gravity-sfx_ambience.wav  beds / drones / pads

The mix is designed to sit UNDER a voice-over: sustained beds avoid the
1-4 kHz speech band, transients are short, master is limited to -1 dBFS.

Requires: numpy, scipy, (ffmpeg on PATH for the loudness check — optional).
Usage:    python3 generate_sfx.py [output_dir]
"""
import math, os, re, subprocess, sys, wave
import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d

SR = 48000
DUR = 39.62
N = int(round(DUR * SR))
TAIL = 4 * SR
TARGET_LUFS = -19.0
AMB_GAIN_DB = 7.0
CEIL_DB = -1.6
TAU = 2 * np.pi


# ----------------------------------------------------------------------------
# basic helpers
# ----------------------------------------------------------------------------
def R(seed):
    return np.random.default_rng(seed)


def ns(d):
    return int(round(d * SR))


def T(d):
    return np.arange(ns(d)) / SR


def H1(n):  # same hash as the animation, so cues line up with on-screen events
    s = math.sin(n * 127.1 + 311.7) * 43758.5453123
    return s - math.floor(s)


def norm(x, peak=1.0):
    m = np.max(np.abs(x)) if len(x) else 0
    return x * (peak / m) if m > 0 else x


def fades(x, a=0.002, r=0.01):
    n = x.shape[-1]
    na, nr = min(n, int(a * SR)), min(n, int(r * SR))
    e = np.ones(n)
    if na > 0:
        e[:na] = np.linspace(0, 1, na)
    if nr > 0:
        e[n - nr:] *= np.linspace(1, 0, nr)
    return x * e


def sos(kind, fc, order=2):
    return signal.butter(order, fc, kind, fs=SR, output='sos')


def lp(x, fc, o=2):
    return signal.sosfilt(sos('lowpass', min(fc, SR / 2 - 100), o), x)


def hp(x, fc, o=2):
    return signal.sosfilt(sos('highpass', fc, o), x)


def bp(x, lo, hi, o=2):
    return signal.sosfilt(sos('bandpass', [lo, min(hi, SR / 2 - 100)], o), x)


def phase(f):
    return TAU * np.cumsum(f) / SR


def noise(d, seed):
    return R(seed).standard_normal(ns(d))


def pink(d, seed):
    n = ns(d)
    F = np.fft.rfft(R(seed).standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    x = np.fft.irfft(F / np.sqrt(f), n)
    return x / (np.std(x) + 1e-12)


def brown(d, seed):
    x = np.cumsum(R(seed).standard_normal(ns(d)))
    x = hp(x, 18)
    return x / (np.std(x) + 1e-12)


def smooth_noise(n, rate, seed):
    """band-limited random wander in [-1, 1] (cosine-interpolated)."""
    k = int(n / SR * rate) + 3
    pts = R(seed).uniform(-1, 1, k)
    t = np.arange(n) / SR * rate
    i = np.floor(t).astype(int)
    fr = t - i
    w = (1 - np.cos(np.pi * fr)) / 2
    return pts[i] * (1 - w) + pts[i + 1] * w


def bandsweep(x, fc, bw=1.0):
    """time-varying gaussian band-pass (in octaves) via STFT masking.
    fc / bw: callables of time (s) or scalars."""
    nper, hop = 1024, 256
    f, tt, Z = signal.stft(x, fs=SR, nperseg=nper, noverlap=nper - hop)
    fcv = fc(tt) if callable(fc) else np.full_like(tt, fc)
    bwv = bw(tt) if callable(bw) else np.full_like(tt, bw)
    lf = np.log2(np.maximum(f, 1.0))[:, None]
    lc = np.log2(np.maximum(fcv, 1.0))[None, :]
    M = np.exp(-0.5 * ((lf - lc) / (bwv[None, :] / 2)) ** 2)
    _, y = signal.istft(Z * M, fs=SR, nperseg=nper, noverlap=nper - hop)
    y = y[:len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y


def pan2(x, pan):
    th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    return np.vstack([x * np.cos(th), x * np.sin(th)]) * math.sqrt(2)


# ----------------------------------------------------------------------------
# buses
# ----------------------------------------------------------------------------
class Bus:
    def __init__(self):
        self.x = np.zeros((2, N + TAIL))

    def add(self, sig, t0, db=0.0, pan=0.0, send=None, verb=None):
        """sig: mono (n,) or stereo (2,n). send=dB to reverb bus `verb`."""
        g = 10 ** (db / 20)
        if np.ndim(pan) > 0 and sig.ndim == 1 and len(pan) != len(sig):
            pan = np.interp(np.linspace(0, 1, len(sig)), np.linspace(0, 1, len(pan)), pan)
        st = pan2(sig, pan) if sig.ndim == 1 else sig
        st = st * g
        i0 = int(round(t0 * SR))
        if i0 < 0:
            st, i0 = st[:, -i0:], 0
        n = min(st.shape[1], self.x.shape[1] - i0)
        if n <= 0:
            return
        self.x[:, i0:i0 + n] += st[:, :n]
        if send is not None and verb is not None:
            verb.x[:, i0:i0 + n] += st[:, :n] * 10 ** (send / 20)


def make_ir(d=2.6, seed=5):
    t = T(d)
    out = []
    for ch in range(2):
        n = R(seed + ch).standard_normal(len(t))
        ir = 0.55 * hp(n, 1500) * np.exp(-t / 0.28) + lp(n, 3000) * np.exp(-t / 0.75)
        ir[:int(0.014 * SR)] = 0
        out.append(ir / np.sqrt(np.sum(ir ** 2)))
    return np.array(out)


# ----------------------------------------------------------------------------
# sound design library  (all mono unless noted, peak ~1.0)
# ----------------------------------------------------------------------------
def whoosh(d=0.45, f0=250, f1=3200, peak=0.6, bw=1.4, seed=0, curve=1.0):
    x = pink(d, seed)
    y = bandsweep(x, lambda tt: f0 * (f1 / f0) ** (np.clip(tt / d, 0, 1) ** curve), bw)
    t = T(d)
    k = peak * d
    e = np.where(t < k, (t / k) ** 2.2, np.exp(-(t - k) / (0.22 * (d - k) + 1e-3)))
    return fades(norm(y * e), 0.002, 0.03)


def doppler_whoosh(d=0.9, fpk=2600, flo=180, seed=0):
    t = T(d)
    pk = 0.5 * d
    fc = lambda tt: flo + (fpk - flo) * np.exp(-((tt - pk) / (0.22 * d)) ** 2)
    y = bandsweep(pink(d, seed), fc, 1.6)
    e = np.exp(-((t - pk) / (0.28 * d)) ** 2)
    return fades(norm(y * e), 0.005, 0.05)


def impact(d=1.8, f0=110, f1=34, sub_tau=0.45, body=1.0, crack=0.6, seed=0):
    # phone-speaker friendly: the sub is saturated so its harmonics carry the hit,
    # and a 90-320 Hz "punch" layer does the work small speakers can reproduce
    t = T(d)
    f1 = max(f1, 42)
    f = f1 + (f0 - f1) * np.exp(-t / 0.07)
    sub = np.sin(phase(f)) * np.exp(-t / (sub_tau * 0.7)) * (1 - np.exp(-t / 0.002))
    sub = np.tanh(3.0 * sub) / np.tanh(3.0)
    n = noise(d, seed)
    punch = norm(bp(n, 90, 320)) * np.exp(-t / 0.09) * (1 - np.exp(-t / 0.0015))
    bod = norm(lp(n, 1400)) * np.exp(-t / 0.14) * (1 - np.exp(-t / 0.001))
    cr = norm(hp(n, 2500)) * np.exp(-t / 0.014)
    return fades(norm(0.7 * sub + 0.6 * punch + 0.45 * body * bod + 0.35 * crack * cr), 0.0005, 0.05)


def riser(d=1.0, f0=300, f1=6000, seed=0, tone=0.35):
    t = T(d)
    y = norm(bandsweep(pink(d, seed), lambda tt: f0 * (f1 / f0) ** (np.clip(tt / d, 0, 1) ** 1.5), 0.9))
    tf = 70 * 8 ** ((t / d) ** 1.6)
    ph = phase(tf)
    tn = np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)
    trem = 1 + 0.45 * np.sin(phase(3 + 22 * (t / d) ** 2))
    return fades(norm((y + tone * norm(tn)) * (t / d) ** 2.4 * trem), 0.01, 0.004)


def rev_swell(d=0.7, seed=0, fc=4000):
    t = T(d)
    x = lp(hp(noise(d, seed), fc), 13000) * np.exp(-t / (d * 0.33))
    return fades(norm(x[::-1]), 0.01, 0.003)


def pop(f0=420, f1=1300, d=0.09):
    t = T(d)
    y = np.sin(phase(f0 * (f1 / f0) ** (t / d))) * np.exp(-t / 0.028) * (1 - np.exp(-t / 0.0008))
    return fades(norm(y), 0.0002, 0.01)


def click(f=2400, d=0.03, seed=0, tau=0.004):
    t = T(d)
    y = np.sin(TAU * f * t) * np.exp(-t / tau) + 0.35 * norm(hp(noise(d, seed), 3000)) * np.exp(-t / 0.0015)
    return fades(norm(y), 0.0001, 0.005)


def switch_click(seed=0):
    t = T(0.12)
    thump = np.sin(phase(160 * (1 + 2 * np.exp(-t / 0.004)))) * np.exp(-t / 0.018)
    y = 0.8 * thump
    y[: int(0.03 * SR)] += click(1150, 0.03, seed, 0.005)
    i = int(0.034 * SR)
    y[i:i + int(0.03 * SR)] += 0.8 * click(1750, 0.03, seed + 1, 0.004)
    return norm(y)


def glitch(d=0.3, seed=0, fps=30):
    r = R(seed)
    n = ns(d)
    y = np.zeros(n)
    seg = int(SR / fps)
    prev = None
    for i0 in range(0, n, seg):
        L = min(seg, n - i0)
        t = np.arange(L) / SR
        c = r.random()
        if c < 0.22:
            s = np.zeros(L)
        elif c < 0.55:
            f = r.choice([90, 120, 180, 240, 360, 480, 720, 960, 1440])
            s = np.sign(np.sin(TAU * f * t)) * 0.55
        elif c < 0.75:
            s = np.round(r.standard_normal(L) * 2) / 4
        elif prev is not None:
            s = prev[:L].copy()
        else:
            s = np.zeros(L)
        hold = int(r.choice([1, 3, 6, 12]))
        s = np.repeat(s[::hold], hold)[:L]
        if len(s) < L:
            s = np.pad(s, (0, L - len(s)))
        y[i0:i0 + L] = fades(s, 0.0005, 0.0005)
        prev = s
    return norm(lp(y, 7000))


def alarm(d=0.75, rate=4.0, f_hi=330, f_lo=262):
    t = T(d)
    f = np.where((np.floor(t * rate * 2) % 2) == 0, f_hi, f_lo)
    ph = phase(f)
    y = lp(signal.sawtooth(ph) * 0.6 + np.sin(ph) * 0.4, 1500)
    am = 0.55 + 0.45 * (0.5 + 0.5 * np.cos(TAU * rate * t))
    return fades(norm(y * am), 0.01, 0.06)


def metal_crash(d=1.3, seed=0, n=30, fmin=170, fmax=5200):
    r = R(seed)
    t = T(d)
    y = np.zeros(len(t))
    for _ in range(n):
        f = fmin * (fmax / fmin) ** r.random()
        y += (r.random() ** 1.5) * np.sin(TAU * f * t + r.random() * TAU) * np.exp(-t / (0.03 + r.random() ** 2 * 0.55))
    burst = norm(lp(noise(d, seed + 1), 5000)) * np.exp(-t / 0.05)
    return fades(norm(0.7 * norm(y) + 0.6 * burst), 0.0003, 0.05)


def grains(d=0.8, seed=0, n=140, fmin=2500, fmax=10000, tmean=0.12, tau=(0.004, 0.03), amp=1.0):
    """stereo cloud of tiny sine pings (glass, sparkle, droplets)."""
    r = R(seed)
    out = np.zeros((2, ns(d)))
    for _ in range(n):
        t0 = min(r.exponential(tmean), d - 0.06)
        f = fmin * (fmax / fmin) ** r.random()
        ta = tau[0] + r.random() * (tau[1] - tau[0])
        L = int(min(ta * 6, d - t0) * SR)
        tt = np.arange(L) / SR
        g = np.sin(TAU * f * tt) * np.exp(-tt / ta) * (0.25 + r.random()) * amp
        i0 = int(t0 * SR)
        L = min(L, out.shape[1] - i0)
        out[:, i0:i0 + L] += pan2(g[:L], r.uniform(-0.9, 0.9))
    return out / (np.max(np.abs(out)) + 1e-9)


def glass(d=0.8, seed=0):
    g = grains(d, seed, 150, 2600, 10000, 0.11)
    t = T(d)
    burst = norm(hp(noise(d, seed + 3), 4000)) * np.exp(-t / 0.03)
    return g * 0.8 + pan2(burst * 0.5, 0)


def groan(d=1.0, f=52, seed=0):
    t = T(d)
    n = len(t)
    fw = f * (1 + 0.12 * smooth_noise(n, 2.5, seed)) * (1 - 0.15 * t / d)
    x = signal.sawtooth(phase(fw))
    y = bandsweep(x, lambda tt: 380 + 220 * np.sin(TAU * 0.9 * tt + seed), 1.1)
    e = np.clip(np.sin(np.pi * t / d), 0, 1) ** 0.6 * (0.7 + 0.3 * smooth_noise(n, 9, seed + 1))
    return fades(norm(y * e), 0.02, 0.05)


def bubbles(d=1.0, n=60, seed=0, fmin=300, fmax=1400, tau=(0.01, 0.05)):
    r = R(seed)
    out = np.zeros((2, ns(d)))
    for _ in range(n):
        f0 = fmin * (fmax / fmin) ** r.random()
        ta = tau[0] + r.random() * (tau[1] - tau[0])
        L = int(ta * 6 * SR)
        tt = np.arange(L) / SR
        b = np.sin(phase(f0 * (1 + 0.7 * tt / (ta * 6)))) * np.exp(-tt / ta) * (1 - np.exp(-tt / 0.0008))
        i0 = int(r.uniform(0, max(1e-3, d - ta * 6)) * SR)
        L = min(L, out.shape[1] - i0)
        out[:, i0:i0 + L] += pan2(b[:L] * (0.3 + r.random()), r.uniform(-0.8, 0.8))
    return out / (np.max(np.abs(out)) + 1e-9)


def bloop(f0=160, d=0.45, seed=0):
    t = T(d)
    f = f0 * (1 + 1.6 * (1 - np.exp(-t / 0.12))) * (1 + 0.05 * np.sin(TAU * 9 * t))
    y = np.sin(phase(f)) * np.exp(-t / 0.13) * (1 - np.exp(-t / 0.004))
    return fades(norm(y), 0.001, 0.03)


def water_rush(d=1.4, seed=0, f0=250, f1=900):
    n = ns(d)
    st = []
    for ch in range(2):
        x = pink(d, seed + ch)
        wob = smooth_noise(n, 4, seed + 7 + ch)
        y = bandsweep(x, lambda tt: f0 * (f1 / f0) ** np.clip(tt / d, 0, 1) * (1 + 0.25 * np.interp(tt, np.arange(n) / SR, wob)), 1.6)
        st.append(y)
    st = np.array(st)
    low = norm(lp(brown(d, seed + 3), 220))
    st = st / np.max(np.abs(st)) + 0.6 * low
    return st / np.max(np.abs(st))


def splash(d=1.1, seed=0):
    t = T(d)
    y = bandsweep(pink(d, seed), lambda tt: 3200 * np.exp(-tt / 0.3) + 600, 2.0)
    e = (1 - np.exp(-t / 0.004)) * np.exp(-t / 0.24)
    body = norm(norm(y) * e + 0.7 * impact(d, 90, 40, 0.18, 0.6, 0.2, seed + 1))
    st = pan2(body, 0) * 0.8
    drops = bubbles(d, 50, seed + 2, 700, 2600, (0.006, 0.025))
    drops *= np.clip(np.linspace(-0.3, 1.2, drops.shape[1]), 0, 1) ** 0.5
    return st + 0.45 * drops


def wind(d=1.5, f0=300, f1=2400, seed=0, bw=1.1):
    n = ns(d)
    st = []
    for ch in range(2):
        x = pink(d, seed + ch * 11)
        y = bandsweep(x, lambda tt: f0 * (f1 / f0) ** np.clip(tt / d, 0, 1), bw)
        gust = 0.65 + 0.35 * smooth_noise(n, 5, seed + 3 + ch)
        st.append(y * gust)
    st = np.array(st)
    return st / np.max(np.abs(st))


def twang(f0=600, d=0.35, seed=0, metallic=False, tau=0.09):
    t = T(d)
    ph = phase(f0 * (0.55 + 0.45 * np.exp(-t / 0.05)))
    y = np.sin(ph)
    if metallic:
        y = y + 0.5 * np.sin(2.01 * ph) + 0.3 * np.sin(3.03 * ph) + 0.15 * np.sin(4.97 * ph)
    y = y * np.exp(-t / tau) + 0.5 * norm(hp(noise(d, seed), 2000)) * np.exp(-t / 0.003)
    return fades(norm(y), 0.0002, 0.03)


def rumble(d=1.0, seed=0, fc=140, crackle=0, crack_amp=0.5):
    t = T(d)
    n = len(t)
    b = norm(lp(brown(d, seed), fc)) * (0.65 + 0.35 * smooth_noise(n, 7, seed + 1))
    if crackle:
        r = R(seed + 2)
        c = np.zeros(n)
        for _ in range(int(crackle * d)):
            L = int(r.uniform(0.002, 0.012) * SR)
            i0 = r.integers(0, max(1, n - L))
            tt = np.arange(L) / SR
            c[i0:i0 + L] += r.standard_normal(L) * np.exp(-tt / (L / SR / 3)) * r.random()
        b = b + crack_amp * norm(bp(c, 700, 6000))
    return fades(norm(b), 0.005, 0.08)


def hum(d=0.7, f=60):
    t = T(d)
    y = signal.sawtooth(TAU * f * t) * 0.5 + np.sin(TAU * 2 * f * t) * 0.6 + 0.3 * signal.square(TAU * 3 * f * t)
    y = lp(y, 900) * (1 + 0.15 * np.sin(TAU * 7 * t))
    return fades(norm(np.tanh(1.5 * y)), 0.02, 0.05)


def zap(d=0.35, seed=0):
    r = R(seed)
    t = T(d)
    n = len(t)
    imp = np.zeros(n)
    idx = r.integers(0, n, int(d * 1200))
    imp[idx] = r.standard_normal(len(idx))
    cr = hp(signal.lfilter([1], [1, -0.65], imp), 1500)
    gate = np.repeat(r.random(int(d * 45) + 2) > 0.35, int(SR / 45) + 1)[:n]
    buzz = signal.square(TAU * (100 + 50 * r.random()) * t) * gate
    sizzle = norm(hp(noise(d, seed + 1), 5000))
    e = np.exp(-t / (d * 0.45)) * (1 - np.exp(-t / 0.001))
    y = np.tanh(3 * (norm(cr) + 0.35 * buzz + 0.3 * sizzle)) * e
    return fades(norm(lp(y, 12000)), 0.0002, 0.02)


def beep(f, d, a=0.004, r_=0.025):
    t = T(d)
    y = np.sin(TAU * f * t) + 0.12 * np.sin(2 * TAU * f * t)
    e = np.clip(np.minimum(t / a, (d - t) / r_), 0, 1)
    return y * e


def thump(f0=55, d=0.2, seed=0):
    t = T(d)
    y = np.sin(phase(f0 * (1 + 1.4 * np.exp(-t / 0.015)))) * np.exp(-t / 0.06) * (1 - np.exp(-t / 0.003))
    y = y + 0.35 * norm(lp(noise(d, seed), 300)) * np.exp(-t / 0.03)
    return fades(norm(np.tanh(1.6 * y)), 0.0005, 0.02)


def heartbeat(seed=0):
    out = np.zeros(int(0.45 * SR))
    out[: int(0.2 * SR)] += thump(55, 0.2, seed)
    i = int(0.16 * SR)
    out[i:i + int(0.18 * SR)] += 0.7 * thump(68, 0.18, seed + 1)
    return norm(out)


def pad(freqs, d, attack=0.8, release=1.0, seed=0, cutoff=1300, detune=0.006):
    t = T(d)
    r = R(seed)
    st = np.zeros((2, len(t)))
    for f in freqs:
        for det, p in ((-detune, -0.7), (0.0, 0.0), (detune, 0.7)):
            s = signal.sawtooth(TAU * f * (1 + det) * t + r.random() * TAU)
            st += pan2(s, p)
    st = np.array([lp(st[0], cutoff), lp(st[1], cutoff)])
    e = np.clip(np.minimum(t / attack, (d - t) / release), 0, 1) ** 1.5
    return st * e / (np.max(np.abs(st)) + 1e-9)


def bell(f=1000, d=2.0, seed=0):
    t = T(d)
    y = np.zeros(len(t))
    for ratio, dec, a in ((1, 1.4, 1), (2.0, 0.9, 0.35), (2.76, 0.6, 0.5), (4.07, 0.35, 0.25), (5.4, 0.2, 0.2)):
        y += a * np.sin(TAU * f * ratio * t + R(seed).random() * TAU) * np.exp(-t / dec)
    y *= 1 - np.exp(-t / 0.001)
    return fades(norm(y), 0.0002, 0.1)


def power_down(d=0.7, f0=700, f1=40):
    t = T(d)
    f = f0 * (f1 / f0) ** ((t / d) ** 0.6)
    y = signal.sawtooth(phase(f)) * 0.6 + np.sin(phase(f * 0.5)) * 0.6
    y = lp(y, 2200) * np.clip(1.15 - t / d, 0, 1)
    return fades(norm(np.tanh(1.4 * y)), 0.002, 0.05)


def power_up(d=0.8, f0=55, f1=880):
    t = T(d)
    f = f0 * (f1 / f0) ** ((t / d) ** 1.4)
    y = signal.sawtooth(phase(f)) * 0.5 + np.sin(phase(f)) * 0.7
    y = lp(y, 3000) * (0.3 + 0.7 * (t / d))
    return fades(norm(np.tanh(1.3 * y)), 0.02, 0.01)


def drone(d, seed=0, base=55.0, pulse=None, air=0.35):
    t = T(d)
    n = len(t)
    st = np.zeros((2, n))
    for ch, det in ((0, 1.0), (1, 1.004)):
        s = 0.3 * np.sin(TAU * base * det * t) + 0.3 * np.sin(TAU * base * 1.003 * t + ch) \
            + 0.35 * np.sin(TAU * base * 1.5 * det * t) + 0.32 * np.sin(TAU * base * 2.0 * t + 0.6 * np.sin(TAU * 0.13 * t)) \
            + 0.16 * np.sin(TAU * base * 3.0 * det * t)
        a = bandsweep(pink(d, seed + ch), lambda tt: 260 + 120 * np.sin(TAU * 0.11 * tt + ch), 1.4)
        st[ch] = norm(s) + air * norm(a)
    if pulse:
        st *= 0.72 + 0.28 * np.cos(TAU * pulse * t)
    return st / np.max(np.abs(st))


def unwind(d=0.95):
    t = T(d)
    f = 900 * 0.17 ** (t / d)
    f = f * (1 + 0.09 * (t / d) * np.sin(phase(5 + 12 * t / d)))
    ph = phase(f)
    y = (np.sin(ph) + 0.4 * np.sin(2.01 * ph)) * np.exp(-t / (d * 0.55)) * (1 - np.exp(-t / 0.01))
    return fades(norm(y), 0.002, 0.05)


def wobble(d=0.6):
    t = T(d)
    lfo = np.sin(TAU * 7 * t) * 0.035 * (0.3 + t / d)
    y = np.sin(phase(196 * (1 + lfo))) + np.sin(phase(207.6 * (1 - lfo))) + 0.3 * signal.sawtooth(phase(98 * (1 + lfo)))
    y = lp(y, 2000) * np.exp(-t / (d * 0.6)) * (1 - np.exp(-t / 0.005))
    return fades(norm(y), 0.002, 0.05)


def question(d=0.32, f0=230, f1=560):
    t = T(d)
    f = f0 * (f1 / f0) ** ((t / d) ** 2) * (1 + 0.025 * np.sin(TAU * 11 * t))
    y = np.sin(phase(f)) + 0.25 * np.sin(2 * phase(f))
    e = np.clip(np.minimum(t / 0.01, (d - t) / 0.08), 0, 1)
    return norm(y * e)


def stone_clack(seed=0, f=520):
    d = 0.08
    t = T(d)
    y = norm(bp(noise(d, seed), 700, 3200)) * np.exp(-t / 0.006) + 0.8 * np.sin(TAU * f * t) * np.exp(-t / 0.02)
    return fades(norm(y), 0.0002, 0.01)


def grind(d=1.0, seed=0):
    t = T(d)
    n = len(t)
    y = norm(bandsweep(pink(d, seed), lambda tt: 320 + 120 * np.sin(TAU * 1.3 * tt), 1.6))
    rough = 0.4 + 0.6 * np.abs(smooth_noise(n, 32, seed + 1))
    return fades(norm(y * rough), 0.01, 0.08)


def boing(f0=300, d=0.5):
    t = T(d)
    f = f0 * (1 + 0.35 * np.exp(-t / 0.22) * np.sin(TAU * 17 * t))
    return fades(norm(np.sin(phase(f)) * np.exp(-t / 0.2) * (1 - np.exp(-t / 0.002))), 0.0005, 0.03)


def static_burst(d=0.25, seed=0):
    r = R(seed)
    n = ns(d)
    gate = np.repeat(r.random(int(d * 60) + 2) > 0.4, int(SR / 60) + 1)[:n]
    y = bp(noise(d, seed), 900, 4500) * gate
    return fades(norm(y), 0.002, 0.03)


def signal_lost(seed=0):
    d = 0.32
    t = T(d)
    f = 1300 * (0.25 ** (t / d))
    y = lp(signal.square(phase(f)), 2500) * np.exp(-t / 0.18)
    out = norm(y)
    out[: int(0.25 * SR)] += 0.35 * static_burst(0.25, seed)
    return norm(out)


def solar_roar(d=1.7, seed=0):
    t = T(d)
    n = len(t)
    st = []
    for ch in range(2):
        a = norm(bandsweep(pink(d, seed + ch), lambda tt: 160 + 70 * np.sin(TAU * 0.6 * tt + ch), 1.3))
        b = norm(lp(brown(d, seed + 5 + ch), 90))
        st.append((0.7 * a + 0.8 * b) * (0.75 + 0.25 * smooth_noise(n, 3, seed + 9 + ch)))
    st = np.array(st)
    e = np.clip(np.minimum(t / 0.15, (d - t) / 0.3), 0, 1)
    return st * e / np.max(np.abs(st))


def popper(seed=0):
    d = 0.7
    t = T(d)
    b = norm(bp(noise(d, seed), 900, 6000)) * np.exp(-t / 0.02)
    return pan2(b, 0) * 0.8 + 0.6 * grains(d, seed + 1, 90, 1800, 6000, 0.18, (0.002, 0.008))


# ----------------------------------------------------------------------------
# timeline
# ----------------------------------------------------------------------------
def build():
    fx, amb, verb = Bus(), Bus(), Bus()
    V = dict(verb=verb)

    def F(sig, t0, db, pan=0.0, send=None):
        fx.add(sig, t0, db, pan, send, verb)

    def A(sig, t0, db, pan=0.0, send=None):
        amb.add(sig, t0, db + AMB_GAIN_DB, pan, send, verb)

    # transition whooshes on every hard cut that has no bigger hit of its own
    cut_whoosh = {1.15: 1, 3.95: 2, 5.5: 3, 6.32: 4, 7.7: 5, 10.85: 6, 13.0: 7, 18.9: 8, 21.2: 9,
                  22.55: 10, 26.2: 11, 28.0: 12, 31.4: 13, 34.5: 14, 37.75: 15}
    for tc, s in cut_whoosh.items():
        d = 0.38 + 0.12 * H1(s)
        f0, f1 = (220, 2600) if s % 2 else (2800, 260)
        w = whoosh(d, f0, f1, 0.72, 1.3, seed=100 + s)
        F(w, tc - 0.72 * d, -13, np.linspace(-0.6, 0.6, len(w)) * (1 if s % 3 else -1), send=-18)

    # ---------------- 0.00 HOOK ----------------
    A(drone(3.5, 1, 55.0, pulse=1.05), 0.0, -21)
    F(impact(2.4, 95, 30, 0.6, 0.8, 0.4, seed=1), 0.0, -2, send=-10)
    F(grains(1.2, 2, 50, 2500, 7000, 0.3, (0.01, 0.06)), 0.02, -22)
    for i, tm in enumerate((0.14, 0.38, 0.6, 0.83, 1.0)):  # meteors streaking in
        w = whoosh(0.28, 1400, 5200, 0.8, 1.0, seed=20 + i)
        F(w, tm, -22, -0.8 + 0.4 * i)
    # gravity field hum (1.15 - 2.2)
    t = T(1.1)
    fh = np.sin(TAU * 110 * t) + 0.5 * np.sin(TAU * 165.4 * t) + 0.3 * np.sin(TAU * 220.8 * t)
    F(fades(norm(fh) * (0.7 + 0.3 * np.sin(TAU * 3.6 * t)), 0.15, 0.1), 1.12, -25)
    F(glitch(0.35, 3), 1.85, -15, 0.2)
    F(click(3200, 0.03, 4), 1.86, -18)

    # ---------------- 2.2 SWITCH ----------------
    F(whoosh(0.28, 700, 4200, 0.8, 1.0, seed=30), 2.0, -17)
    F(pop(380, 1100), 2.26, -15, send=-16)
    F(riser(0.5, 400, 2400, seed=31, tone=0.2), 2.46, -21)
    F(switch_click(32), 2.95, -7, 0.3, send=-14)
    F(whoosh(0.16, 2500, 900, 0.5, 0.9, seed=33), 2.96, -20, 0.1)
    F(power_down(0.75, 650, 38), 3.0, -10, send=-16)
    F(beep(520, 0.09) * 0.8, 3.08, -19)
    F(rev_swell(0.62, 34), 2.6, -12)

    # ---------------- 3.2 GRAVITY OFF ----------------
    F(impact(2.8, 125, 26, 0.75, 1.0, 0.8, seed=40), 3.2, 0, send=-8)
    F(doppler_whoosh(0.9, 2200, 120, seed=41), 3.2 - 0.25, -9)
    F(glitch(0.13, 42), 3.2, -13)
    for k in range(12):  # 9.81 -> 0.00 readout (fast then slow, like eo3)
        u = 1 - (1 - (k + 0.5) / 12) ** (1 / 3)
        F(click(2600 - 120 * k, 0.025, 43 + k, 0.003), 3.22 + 0.4 * u, -24, 0.0)
    F(alarm(0.78), 3.2, -21)

    # ---------------- 3.95 CARS ----------------
    A(rumble(2.4, 50, 180), 3.95, -28)
    A(wind(2.4, 200, 700, seed=51), 3.95, -30)
    F(groan(1.0, 48, 52), 4.0, -17, -0.4)
    F(groan(0.9, 61, 53), 4.1, -19, 0.4)
    for i in range(6):  # cones / debris clatter as they lift
        F(stone_clack(54 + i, 900 + 140 * i), 4.15 + 0.13 * i, -26, -0.7 + 0.28 * i)
    F(riser(0.55, 300, 3000, 55, 0.25), 4.45, -18)
    F(impact(1.6, 105, 38, 0.4, 1.0, 1.0, seed=56), 5.0, -2, send=-10)
    F(metal_crash(1.3, 57), 5.0, -5, 0.0, send=-12)
    F(glass(0.9, 58), 5.0, -11)

    # ---------------- 5.5 PEOPLE LIFT ----------------
    F(wind(0.85, 250, 1400, seed=60), 5.5, -16)
    F(doppler_whoosh(0.7, 1800, 200, seed=61), 5.55, -12, np.linspace(0.6, -0.2, int(0.7 * SR)))

    # ---------------- 6.32 WATER ----------------
    F(splash(0.9, 70), 6.32, -11, -0.3, send=-14)
    A(water_rush(1.45, 71, 220, 950), 6.3, -15)
    F(bubbles(1.3, 70, 72, 280, 1300), 6.4, -17)
    F(bubbles(1.0, 45, 73, 900, 2600, (0.005, 0.02)), 6.7, -21)

    # ---------------- 7.7 AIR ESCAPES ----------------
    F(wind(1.7, 280, 2600, seed=80, bw=1.0), 7.66, -10)
    F(whoosh(0.9, 400, 6000, 0.95, 1.2, seed=81, curve=1.6), 7.6, -15)
    F(whoosh(0.5, 300, 5000, 0.5, 1.4, seed=82), 8.45 - 0.25, -9, send=-14)
    F(impact(0.7, 160, 70, 0.15, 0.6, 0.2, seed=83), 8.45, -14)

    # ---------------- 9.34 BOUND ----------------
    A(drone(6.2, 90, 55.0), 9.3, -25)
    F(glitch(0.08, 91), 9.34, -15)
    F(pop(350, 1000), 9.37, -16, send=-16)
    for i in range(34):  # tethers snapping one by one
        sn = 0.12 + H1(i * 3.7) * 0.75
        a = i / 34 * TAU + H1(i * 1.9) * 0.12 + sn * 0.1
        F(twang(560 + 520 * H1(i * 5.1), 0.3, 900 + i), 9.34 + sn, -25, math.cos(a) * 0.8)
    for k in range(14):  # escape-velocity counter
        u = 1 - (1 - (k + 0.5) / 14) ** (1 / 3)
        F(click(2200 - 50 * k, 0.02, 95 + k, 0.003), 9.46 + 0.73 * u, -28)

    # ---------------- 10.85 CLOUDS ----------------
    F(wind(1.4, 600, 260, seed=100, bw=1.4), 10.85, -15)
    for i in range(6):
        w = whoosh(0.7, 300, 1500, 0.5, 1.6, seed=101 + i)
        F(w, 10.9 + 0.17 * i, -21, -0.8 + 0.32 * i)
    F(riser(0.62, 500, 5000, 107, 0.3), 11.62, -15)
    F(rev_swell(0.4, 108), 11.86, -14)

    # ---------------- 12.22 SCATTER ----------------
    F(impact(1.6, 140, 45, 0.35, 0.8, 0.9, seed=120), 12.24, -3, send=-10)
    F(grains(1.1, 121, 160, 1500, 6500, 0.14, (0.01, 0.07)), 12.24, -9)
    F(doppler_whoosh(0.6, 3500, 400, seed=122), 12.1, -14)

    # ---------------- 13.0 OCEANS ----------------
    A(rumble(2.3, 130, 160), 13.0, -21)
    for i in range(9):
        st = 13.0 + 0.05 + H1(i * 3.3) * 0.45
        a = -math.pi / 2 + (i - 4) * 0.62
        F(bloop(110 + 140 * H1(i * 4.4), 0.5, 131 + i), st, -13, math.cos(a) * 0.8, send=-16)
    F(bubbles(1.2, 40, 140, 200, 700, (0.03, 0.09)), 13.2, -19)
    F(riser(0.5, 250, 3500, 141, 0.3), 13.85, -15)
    F(splash(1.2, 142), 14.35, -3, 0, send=-10)
    F(impact(1.4, 95, 34, 0.4, 0.7, 0.6, seed=143), 14.35, -5)
    F(bubbles(1.1, 90, 144, 900, 3200, (0.004, 0.02)), 14.42, -15)

    # ---------------- 15.28 BRIDGE ----------------
    A(rumble(2.8, 150, 200, crackle=40, crack_amp=0.3), 15.28, -17)
    A(wind(2.8, 300, 900, seed=151), 15.28, -27)
    F(impact(1.2, 120, 40, 0.3, 1.0, 1.0, seed=152), 15.29, -5, send=-12)
    F(groan(0.6, 44, 153), 15.28, -11)
    for k, tm in enumerate((15.31, 15.36, 15.42, 15.5)):
        F(twang(170 + 60 * k, 0.45, 154 + k, metallic=True, tau=0.16), tm, -12, -0.6 + 0.4 * k, send=-14)

    # ---------------- 15.85 BUILDINGS ----------------
    F(impact(1.1, 90, 32, 0.35, 1.0, 0.6, seed=160), 15.86, -4, send=-12)
    F(rumble(0.7, 161, 260, crackle=220, crack_amp=0.6), 15.86, -8)
    for i in range(6):
        tm = 15.85 + H1(i * 2.7) * 0.15
        F(stone_clack(162 + i, 300 + 60 * i), tm + 0.03, -20, -0.8 + 0.32 * i)

    # ---------------- 16.45 POWER LINES ----------------
    F(hum(0.7, 60), 16.45, -18)
    for i in range(3):
        for k in (-1, 0, 1):
            sn = 16.45 + 0.2 + H1(i * 3 + k + 5) * 0.16
            x = (160 + 700 * i + 350 - 1080) / 1080
            F(twang(240 + 40 * k, 0.3, 170 + i * 3 + k, metallic=True, tau=0.08), sn, -17, x)
            F(zap(0.28, 180 + i * 3 + k), sn, -14, x)
    F(zap(0.45, 190), 16.75, -7, -0.05, send=-12)
    F(impact(0.8, 140, 55, 0.18, 0.5, 1.0, seed=191), 16.75, -9)

    # ---------------- 17.1 GROUND TEARS ----------------
    F(impact(2.0, 80, 24, 0.5, 1.0, 0.8, seed=200), 17.1, 0, send=-10)
    F(rumble(1.0, 201, 220, crackle=300, crack_amp=0.7), 17.1, -5)
    F(grind(0.95, 202), 17.12, -13)
    for i in range(5):
        F(stone_clack(203 + i, 260 + 50 * i), 17.2 + 0.14 * i, -18, -0.6 + 0.3 * i)

    # ---------------- 18.06 ZOOM OUT ----------------
    F(doppler_whoosh(0.95, 3000, 110, seed=210), 18.0, -4, send=-12)
    F(impact(1.4, 70, 28, 0.5, 0.2, 0.0, seed=211), 18.48, -12)
    A(drone(7.2, 212, 55.0, air=0.25), 18.3, -25)

    # ---------------- 18.9 SATELLITES ----------------
    F(glitch(0.08, 220), 18.9, -17)
    for k in range(4):
        F(beep(1150 if k % 2 else 1320, 0.05), 18.98 + 0.15 * k, -30, 0.3 if k % 2 else -0.3)
    SATS = [(640, 220, -0.32, 0.4, 1.05), (790, 300, 0.22, 2.1, 0.85), (940, 350, -0.08, 4.0, 0.7),
            (560, 470, 0.95, 1.1, 1.2), (1060, 250, 0.5, 3.2, 0.6)]
    for i, (rx, ry, tl, ph, sp) in enumerate(SATS):
        rel = 0.62 + i * 0.09
        th = ph + sp * rel
        x = rx * math.cos(th) * math.cos(tl) - ry * math.sin(th) * math.sin(tl)
        F(signal_lost(221 + i), 18.9 + rel, -19, np.clip(x / 1100, -0.9, 0.9))
    F(static_burst(0.35, 230), 19.6, -26)

    # ---------------- 21.2 MOON ----------------
    t = T(0.6)
    F(fades(np.sin(TAU * 220 * t) * (0.6 + 0.4 * np.sin(TAU * 6 * t)), 0.1, 0.1), 21.2, -31)
    F(unwind(0.95), 21.7, -11, np.linspace(-0.3, 0.7, int(0.95 * SR)), send=-12)
    F(whoosh(0.6, 2400, 300, 0.4, 1.4, seed=240), 21.7, -18, 0.5)
    F(pop(500, 1300), 21.82, -20)

    # ---------------- 22.55 SUN ----------------
    A(solar_roar(1.75, 250), 22.55, -15)
    F(rumble(1.7, 251, 300, crackle=60, crack_amp=0.5), 22.55, -21)
    F(wobble(0.6), 23.9, -10, send=-14)
    F(impact(0.9, 120, 50, 0.2, 0.5, 0.4, seed=252), 23.92, -11)
    for k, tm in enumerate((23.9, 23.97, 24.04)):
        F(question(0.26, 260 + 40 * k, 620 + 60 * k), tm, -21, (-0.5, 0.5, 0.0)[k])

    # ---------------- 24.26 QUESTION ----------------
    F(glitch(0.24, 260), 24.26, -13)
    F(impact(1.2, 130, 45, 0.3, 0.6, 0.5, seed=261), 24.26, -5, send=-10)
    F(question(0.4, 200, 520), 24.3, -12, send=-12)
    A(drone(1.0, 262, 55.0, air=0.2), 24.2, -27)

    # ---------------- 25.15 NO AIR (vacuum) ----------------
    F(rev_swell(0.35, 270, 2500), 24.85, -14)
    F(doppler_whoosh(0.45, 1800, 200, seed=271), 25.1, -14)
    for k in range(8):  # O2 gauge 21% -> 0%
        F(click(1500 - 90 * k, 0.02, 272 + k, 0.004), 25.27 + 0.075 * k, -27)
    for k in range(3):  # warning, heard "through" the vacuum
        F(beep(440, 0.12), 25.86 + 0.13 * k, -22)
    t = T(1.0)
    F(fades(np.sin(TAU * 2900 * t) * 0.5, 0.2, 0.2), 25.2, -46)  # thin vacuum whine

    # ---------------- 26.2 NO WATER ----------------
    F(bloop(700, 0.18, 280), 26.22, -15)
    F(pop(450, 1200), 26.26, -21)
    F(rumble(0.2, 281, 2500, crackle=400, crack_amp=1.0), 26.3, -22)
    F(glass(0.7, 282), 26.5, -8)
    F(splash(0.6, 283), 26.5, -13)
    A(wind(0.8, 250, 500, seed=284), 26.2, -27)

    # ---------------- 26.96 NO SOLID GROUND ----------------
    F(impact(1.2, 100, 36, 0.35, 0.8, 0.5, seed=290), 26.96, -6, send=-12)
    F(grind(1.05, 291), 26.97, -12)
    r = R(292)
    for k in range(18):
        F(stone_clack(300 + k, 380 + r.integers(0, 400)), 27.0 + r.random() * 0.6, -24, r.uniform(-0.8, 0.8))
    A(drone(1.1, 293, 49.0, air=0.4), 26.95, -24)

    # ---------------- 28.0 LIFE CAN'T SURVIVE ----------------
    t = T(1.95)
    room = 0.5 * np.sin(TAU * 60 * t) + 0.25 * np.sin(TAU * 120 * t)
    A(fades(norm(room), 0.1, 0.1), 27.98, -36)
    for tb, a in ((28.0, 1.0), (28.42, 0.72), (28.88, 0.42)):
        F(heartbeat(310), tb - 0.02, -6 + 20 * math.log10(a), send=-18)
        F(beep(880, 0.11), tb, -19 + 20 * math.log10(a))
    flat = beep(880, 0.8, 0.01, 0.12) * np.clip(np.linspace(1.15, 0.0, ns(0.8)), 0, 1)
    F(flat, 29.13, -19)
    F(glitch(0.14, 311), 29.18, -18)

    # ---------------- 29.92 GRAVITY REVEALED (calm) ----------------
    # D-major-ish warm pad, low register (out of the voice band)
    A(pad([73.42, 110.0, 146.83, 185.0, 220.0], 3.1, 0.6, 0.6, 320, 1100), 29.9, -18)
    F(grains(1.5, 321, 28, 1800, 4200, 0.5, (0.04, 0.15)), 29.95, -27)
    F(whoosh(0.85, 250, 2200, 0.8, 1.6, seed=322), 30.42, -19)
    F(bell(587.3, 2.0, 323), 30.6, -24, -0.2, send=-10)
    for k in range(4):  # gentle gravity pulses under the person
        F(thump(48, 0.2, 324 + k), 31.45 + 0.37 * k, -24)
    F(pop(300, 800), 31.92, -24)

    # ---------------- 32.92 HOLD YOUR WORLD TOGETHER ----------------
    A(pad([73.42, 110.0, 146.83, 185.0, 220.0, 293.66], 1.75, 0.3, 0.4, 330, 1600), 32.85, -18)
    F(rev_swell(0.78, 331, 3000), 32.92, -10)
    F(power_up(0.78, 55, 880), 32.98, -14)
    r = R(332)
    for k in range(14):  # debris snapping back onto the planet
        F(pop(300 + r.integers(0, 600), 900 + r.integers(0, 900), 0.06), 33.2 + r.random() * 0.47, -27, r.uniform(-0.8, 0.8))
    F(impact(1.8, 110, 34, 0.5, 0.7, 0.4, seed=333), 33.7, -2, send=-9)
    for f in (587.3, 739.99, 880.0, 1174.66):
        F(bell(f, 2.2, 334), 33.7, -16, (f - 800) / 800, send=-10)
    F(grains(1.2, 335, 70, 2500, 8000, 0.25, (0.01, 0.08)), 33.72, -17)

    # ---------------- 34.5 WHAT IF ----------------
    A(pad([110.0, 164.81, 220.0, 277.18], 5.2, 0.5, 1.0, 340, 1200), 34.45, -24)
    F(pop(320, 900, 0.1), 34.58, -13, send=-16)
    F(thump(70, 0.2, 341), 34.6, -17)
    for k in range(1, 6):  # teaser slot-roll
        tm = 34.5 + 0.34 * k
        F(whoosh(0.12, 3000, 900, 0.6, 1.0, seed=342 + k), tm - 0.06, -25)
        F(click(2500, 0.025, 350 + k, 0.003), tm, -22)

    # ---------------- 36.5 COMMENTS ----------------
    F(pop(260, 700, 0.12), 36.5, -10, send=-14)
    F(boing(300, 0.55), 36.52, -15)
    for k, tm in enumerate((36.58, 36.72, 36.86, 37.0)):
        F(whoosh(0.2, 900, 5000, 0.75, 1.0, seed=360 + k), tm - 0.1, -18, np.linspace(0.8, 0.0, int(0.2 * SR)))
        F(pop(420 + 60 * k, 1200 + 100 * k, 0.07), tm + 0.12, -18)

    # ---------------- 37.75 YOUR IDEA -> NEXT VIDEO ----------------
    F(bell(1174.66, 2.0, 370), 37.77, -14, send=-10)
    F(grains(0.9, 371, 60, 2500, 8000, 0.2, (0.01, 0.06)), 37.78, -19)
    F(pop(500, 1400), 37.76, -17)
    F(whoosh(0.5, 3000, 400, 0.6, 1.3, seed=372), 38.25, -17)
    F(whoosh(0.5, 300, 3500, 0.8, 1.3, seed=373), 38.3, -15)
    F(riser(0.62, 400, 6000, 374, 0.3), 38.4, -14)
    F(rev_swell(0.3, 375), 38.72, -14)
    F(impact(1.4, 120, 36, 0.45, 0.7, 0.6, seed=376), 39.02, -1, send=-9)
    F(pop(380, 1300, 0.1), 39.03, -9)
    F(popper(377), 39.04, -9)
    for f in (587.3, 880.0, 1174.66):
        F(bell(f, 1.8, 378), 39.02, -17, (f - 880) / 900, send=-10)

    return fx, amb, verb


# ----------------------------------------------------------------------------
# mastering
# ----------------------------------------------------------------------------
def muffle(x, t0, t1, fc=650, xf=0.06):
    """vacuum effect: low-pass the bus between t0..t1 with short cross-fades."""
    y = np.array([lp(x[0], fc, 4), lp(x[1], fc, 4)])
    n = x.shape[1]
    t = np.arange(n) / SR
    w = np.clip(np.minimum((t - t0) / xf, (t1 - t) / xf), 0, 1)
    return x * (1 - w) + y * w


def limiter(x, ceil_db=CEIL_DB, look=0.004, release=0.15):
    c = 10 ** (ceil_db / 20)
    peak = np.max(np.abs(x), axis=0)
    need = np.minimum(1.0, c / np.maximum(peak, 1e-9))
    w = int(look * SR)
    g = minimum_filter1d(need, size=2 * w + 1)
    g = uniform_filter1d(g, size=w)
    a = 1 - math.exp(-1 / (release * SR))
    gs = 1 - signal.lfilter([a], [1, -(1 - a)], 1 - g)
    g = np.minimum(gs, g)
    return g


def write_wav24(path, st):
    x = np.clip(st.T, -1.0, 1.0 - 1.0 / 8388608)
    i = (x * 8388607).astype('<i4').reshape(-1)
    b = i.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(SR)
        w.writeframes(b)


def lufs(path):
    try:
        out = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', path, '-af', 'ebur128=peak=true', '-f', 'null', '-'],
                             capture_output=True, text=True).stderr
        i = float(re.findall(r'I:\s+(-?[\d.]+) LUFS', out)[-1])
        tp = float(re.findall(r'Peak:\s+(-?[\d.]+) dBFS', out)[-1])
        return i, tp
    except Exception:
        return None, None


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(out_dir, 'stems'), exist_ok=True)
    fx, amb, verb = build()
    ir = make_ir()
    wet = np.array([signal.oaconvolve(verb.x[ch], ir[ch])[: verb.x.shape[1]] for ch in range(2)])
    fxb = fx.x + 0.8 * wet
    ambb = amb.x.copy()
    fxb, ambb = muffle(fxb, 25.22, 26.18), muffle(ambb, 25.22, 26.18, 400)
    hpf = sos('highpass', 30, 4)
    fxb = signal.sosfilt(hpf, fxb, axis=1)[:, :N]
    ambb = signal.sosfilt(hpf, ambb, axis=1)[:, :N]

    tmp = os.path.join(out_dir, '_tmp_measure.wav')
    gain = 1.0
    for _ in range(3):  # loudness-match, then limit
        mix = (fxb + ambb) * gain
        g = limiter(mix)
        write_wav24(tmp, mix * g)
        I, tp = lufs(tmp)
        if I is None:
            break
        delta = TARGET_LUFS - I
        if abs(delta) < 0.3:
            break
        gain *= 10 ** (delta / 20)
    mix = (fxb + ambb) * gain
    g = limiter(mix)
    end = np.ones(N)
    end[-int(0.04 * SR):] = np.linspace(1, 0, int(0.04 * SR))
    full, fxo, ambo = mix * g * end, fxb * gain * g * end, ambb * gain * g * end
    if os.path.exists(tmp):
        os.remove(tmp)
    p_full = os.path.join(out_dir, 'what-if-gravity-sfx.wav')
    write_wav24(p_full, full)
    write_wav24(os.path.join(out_dir, 'stems', 'what-if-gravity-sfx_fx-only.wav'), fxo)
    write_wav24(os.path.join(out_dir, 'stems', 'what-if-gravity-sfx_ambience.wav'), ambo)
    I, tp = lufs(p_full)
    print(f'wrote {p_full}  {N / SR:.2f}s  {SR} Hz / 24-bit stereo  integrated={I} LUFS  true-peak={tp} dBFS')


if __name__ == '__main__':
    main()
