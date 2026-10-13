"""Orchestral background score, composed against the animation timeline. Real CC0 orchestral samples."""
import sys, time
import numpy as np
from instruments import *
from tl import T, TOTAL, shots

M = Bus(TOTAL)      # dry
R = Bus(TOTAL)      # hall reverb send
RS = Bus(TOTAL)     # short room send (piano / intimate)

# ------------------------------------------------------------------ primitives
def N(name, n, t, dur, vel=0.6, g=-14, p=0.0, rev=0.35, atk=0.0, rel=None, sw=None, room=0.0):
    if t >= TOTAL or dur <= 0: return
    y = inst(name).render(n, dur, vel, atk, rel)
    if sw is not None:
        k = max(1, int(dur * SR)); e = np.full(y.shape[1], db(sw[1]), np.float32)
        e[:min(k, len(e))] = db(np.linspace(sw[0], sw[1], k))[:len(e)]
        y = y * e
    G = db(g)
    M.add(y, t, G, p); R.add(y, t, G * rev, p)
    if room: RS.add(y, t, G * room, p)

def X(sig, t, g=-10, p=0.0, rev=0.4, room=0.0):
    if sig is None or t >= TOTAL: return
    G = db(g); M.add(sig, t, G, p); R.add(sig, t, G * rev, p)
    if room: RS.add(sig, t, G * room, p)

def O(pool_name, t, g=-10, p=0.0, rev=0.4, pitch=0.0, i=None, cut=None, fadein=None):
    x = one(pool_name, i)
    if pitch: x = shift(x, pitch)
    if cut: x = fade(x[:, :int(cut * SR)], 0.002, min(cut * 0.4, 0.6))
    if fadein: x = x * env_ar(x.shape[1], fadein, 0.0, 2.0)
    X(x, t, g, p, rev)

def ending_at(pool_name, t_end, g=-10, rev=0.5, i=None, length=None):
    """place a crescendo sample (cymbal swell / roll) so its peak lands on t_end"""
    x = one(pool_name, i)
    a = np.abs(x).mean(0); k = int(np.argmax(np.convolve(a, np.ones(2400) / 2400, 'same')))
    x = x[:, :k + int(0.05 * SR)]
    if length: x = x[:, -int(length * SR):]
    x = x * env_ar(x.shape[1], min(1.0, x.shape[1] / SR * 0.3), 0.06, 1.5)
    X(x, t_end - x.shape[1] / SR, g, 0, rev)

def sub_drop(t, g=-10, f0=55, f1=28, dur=2.2):
    n = int(dur * SR); tt = np.arange(n) / SR
    f = f0 * (f1 / f0) ** (tt / dur); ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * np.exp(-tt * 2.2) * np.minimum(1, tt / 0.004)
    X(np.vstack([y, y]).astype(np.float32), t, g, 0, 0.1)

CH = {'m': [0, 3, 7], 'M': [0, 4, 7], 'sus2': [0, 2, 7], 'sus4': [0, 5, 7], 'dim': [0, 3, 6], 'aug': [0, 4, 8],
      'm7': [0, 3, 7, 10], '7': [0, 4, 7, 10], 'M7': [0, 4, 7, 11], 'madd9': [0, 3, 7, 14], 'add9': [0, 4, 7, 14],
      'hic': [0, 1, 4, 7], 'cl': [0, 1, 6], '5': [0, 7]}
PC = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}

def parse(c):
    m = re.match(r'([A-G][#b]?)(.*)', c); return PC[m.group(1)], (m.group(2) or 'M')

def fit(pc, lo, hi):
    """lowest note of pitch class pc >= lo, folded below hi"""
    n = lo + ((pc - lo) % 12)
    while n > hi: n -= 12
    return n

def voicing(c, top=74):
    pc, q = parse(c); iv = CH[q]
    cb = fit(pc, 31, 42)
    vc = [fit(pc, 43, 54), fit((pc + 7) % 12, 47, 57)]
    vla = [fit((pc + iv[1]) % 12, 55, 64), fit((pc + iv[2 % len(iv)]) % 12, 57, 67)]
    vln = sorted({fit((pc + i) % 12, top - 9, top + 2) for i in iv})
    return dict(cb=cb, vc=vc, vla=vla, vln=vln, pc=pc, iv=iv)

def prog_times(t0, t1, prog, bpm, bpb=4):
    """prog: list of (chord, bars) cycled until t1 -> [(t, dur, chord)]"""
    out = []; t = t0; i = 0; beat = 60.0 / bpm
    while t < t1 - 0.05:
        c, bars = prog[i % len(prog)]; d = bars * bpb * beat
        out.append((t, min(d, t1 - t), c)); t += d; i += 1
    return out

def pad(t0, t1, prog, bpm=60, g=-20, vel=0.4, layers=('cb', 'vc', 'vla', 'vln'), atk=0.6, rel=1.4, trem=False, top=74, rev=0.45, sw=None):
    for t, d, c in prog_times(t0, t1, prog, bpm):
        v = voicing(c, top)
        for L in layers:
            nm = L + ('_trem' if trem and L != 'cb' else '') if L in ('vln', 'vla', 'vc') else ('cb_trem' if trem and L == 'cb' else L)
            ns = v[L] if isinstance(v[L], list) else [v[L]]
            gg = g + {'cb': -1, 'vc': -2, 'vla': -4, 'vln': -4}.get(L, -3) - 2 * (len(ns) - 1)
            pn = {'cb': -0.15, 'vc': -0.35, 'vla': 0.25, 'vln': -0.1}.get(L, 0)
            for n in ns:
                N(nm, n, t + rnd(0, 0.04), d + 0.15, vel, gg, pn, rev, atk, rel, sw)

def drone(t0, t1, note, name='cb', g=-18, vel=0.4, atk=2.0, rel=3.0, rev=0.5):
    N(name, midi(note), t0, t1 - t0, vel, g, 0, rev, atk, rel)

def melody(t0, notes, bpm, name, g=-12, vel=0.6, p=0.1, rev=0.4, legato=1.02, transpose=0, t1=None, atk=0.03, rel=None):
    b = 60.0 / bpm; t = t0
    for item in notes:
        if item[0] is None:  # rest
            t += item[1] * b; continue
        n, d = item[0], item[1]; v = item[2] if len(item) > 2 else vel
        if t1 and t >= t1: break
        N(name, midi(n) + transpose, t, d * b * legato, v, g, p, rev, atk, rel)
        t += d * b
    return t

def arp(t0, t1, prog, bpm, name='piano', g=-16, vel=0.4, pattern=(0, 1, 2, 1), sub=2, octave=60, rev=0.3, room=0.25, p=0.0, ring=1.6):
    beat = 60.0 / bpm
    for t, d, c in prog_times(t0, t1, prog, bpm):
        pc, q = parse(c); iv = CH[q]
        tones = [fit((pc + i) % 12, octave, octave + 11) for i in iv] + [fit(pc, octave + 12, octave + 23)]
        k = 0; tt = t
        while tt < t + d - 0.01 and tt < t1:
            n = tones[pattern[k % len(pattern)] % len(tones)]
            N(name, n, tt, ring, vel * (1.0 if k % (sub * 2) == 0 else 0.85), g, p + rnd(-0.1, 0.1), rev, 0, 0.6, room=room)
            k += 1; tt += beat / sub

def ostinato(t0, t1, bpm, prog, name='vc_spic', g=-16, vel=0.55, steps=(0, 0, 12, 0, 1, 0, 12, 0), base=38, sub=2, accent=(0, 4), p=-0.2, rev=0.3, dur=None):
    beat = 60.0 / bpm; step = beat / sub
    for t, d, c in prog_times(t0, t1, prog, bpm):
        pc, q = parse(c); root = fit(pc, base, base + 11)
        k = 0; tt = t
        while tt < t + d - 0.005 and tt < t1:
            s = steps[k % len(steps)]
            if s is not None:
                v = vel * (1.15 if (k % len(steps)) in accent else 0.85)
                N(name, root + s, tt + rnd(0, 0.008), dur or step * 0.9, min(1, v), g, p, rev)
            k += 1; tt += step

def drums(t0, t1, bpm, pat, bars_len=4, g=0, rev=0.35):
    """pat: list of (beat, pool, gain, pitch)"""
    beat = 60.0 / bpm; L = bars_len * beat; t = t0
    while t < t1 - 0.01:
        for b, pl, gg, pt in pat:
            tt = t + b * beat
            if tt < t1: O(pl, tt + rnd(0, 0.006), g + gg + rnd(-1.5, 0.5), rnd(-0.2, 0.2), rev, pt)
        t += L

def heartbeat(t0, t1, bpm=62, g=-14):
    beat = 60.0 / bpm; t = t0
    while t < t1:
        O('bdrum_soft', t, g, 0, 0.2, -3, cut=0.6); O('bdrum_soft', t + 0.2, g - 5, 0, 0.2, -2, cut=0.5); t += beat

def HIT(t, size=2, chord='Dm', g=-6, brass=True, riser=None):
    """cinematic orchestral impact (real bass drums, timpani, gong, cymbals, low brass)"""
    O('bdrum', t, g, 0, 0.5, -2); O('bd_vcsl', t, g - 2, 0, 0.6, 0, i=1)
    O('timp_hit', t, g - 3, -0.1, 0.5, -3 + int(rnd(-1, 1)))
    if size >= 2:
        O('cymbal', t + 0.01, g - 9, 0.2, 0.6); O('cineboom', t, g - 4, 0, 0.3); sub_drop(t, g - 3)
    if size >= 3:
        O('gong', t, g - 6, -0.1, 0.6, 0, i=1); O('gong_big', t, g - 8, 0.2, 0.5, -2, i=2)
    if brass:
        pc, q = parse(chord)
        for n, nm, gg in [(fit(pc, 34, 45), 'tuba', -2), (fit(pc, 46, 57), 'tbn', -4), (fit((pc + 7) % 12, 50, 61), 'tbn', -5),
                          (fit(pc, 53, 64), 'horn', -5), (fit((pc + CH[q][1]) % 12, 57, 68), 'horn', -6)]:
            N(nm, n, t, 0.9 if size < 3 else 1.8, 0.95, g + gg - 4, rnd(-0.3, 0.3), 0.5, 0.0, 1.2)
    if riser:
        ending_at('cresc', t, g - 10, 0.6, i=4, length=riser)

def soft_hit(t, g=-12, pitch=-3):
    O('timp_hit', t, g, 0, 0.6, pitch); O('bdrum_soft', t, g - 3, 0, 0.5); O('gong', t, g - 12, 0, 0.6, 0, i=3)

def stinger(t, g=-8):
    """horror stinger: dissonant string sfz + brass cluster + metal"""
    for n, nm in [(74, 'vln_trem'), (75, 'vln_trem'), (80, 'vln_trem'), (62, 'vla_trem'), (63, 'vla_trem'), (50, 'vc_trem'), (38, 'cb_trem')]:
        N(nm, n, t, 1.6, 0.95, g - 8, rnd(-0.4, 0.4), 0.5, 0.0, 1.0, sw=(0, -14))
    for n in (45, 46, 51): N('tbn', n, t, 1.2, 0.95, g - 9, rnd(-0.3, 0.3), 0.5, 0.05, 0.8, sw=(-6, 0))
    O('bdrum', t, g - 2, 0, 0.5, -3); O('gong', t, g - 10, 0, 0.6, 2, i=0); sub_drop(t, g - 4)

def tension_cluster(t0, t1, g=-20, base=62):
    for n, nm, pn in [(base, 'vln_trem', -0.3), (base + 1, 'vln_trem', 0.3), (base - 12, 'vla_trem', 0), (base - 24, 'vc_trem', -0.2)]:
        N(nm, n, t0, t1 - t0, 0.5, g, pn, 0.5, (t1 - t0) * 0.6, 1.2, sw=(-18, 0))

# ------------------------------------------------------------------ themes
MAIN = [('D4', 2), ('A4', 1), ('G4', 1), ('F4', 1.5), ('E4', .5), ('D4', 1), ('E4', 1), ('F4', 2), ('G4', 1), ('A4', 1), ('E4', 4),
        ('D4', 2), ('A4', 1), ('Bb4', 1), ('A4', 1.5), ('G4', .5), ('F4', 1), ('E4', 1), ('D4', 1), ('F4', 1), ('E4', 1), ('C#4', 1), ('D4', 4)]
MAIN_PROG = [('Dm', 1), ('Bb', 1), ('F', 1), ('A', 1), ('Dm', 1), ('Gm', 1), ('Bb', .5), ('A', .5), ('Dm', 1)]
KEMAL = [('A4', 1), ('D5', 1), ('C5', 1), ('A4', 1), ('Bb4', 1.5), ('A4', .5), ('G4', 2), ('G4', 1), ('A4', 1), ('F4', 1), ('E4', 1), ('D4', 4),
         ('F4', 1), ('A4', 1), ('C5', 1), ('F5', 1), ('E5', 1.5), ('D5', .5), ('C5', 2), ('Bb4', 1), ('A4', 1), ('G4', 1), ('C#5', 1), ('D5', 4)]
KEMAL_PROG = [('Dm', 1), ('Gm', 1), ('C', .5), ('A7', .5), ('Dm', 1), ('F', 1), ('C', 1), ('Gm', .5), ('A7', .5), ('Dm', 1)]
HICAZ = [('A4', 1), ('Bb4', .5), ('A4', .5), ('G4', 1), ('F#4', 1), ('Eb4', 1), ('F#4', 1), ('G4', 2), ('A4', .5), ('Bb4', .5), ('C5', 1), ('Bb4', 1), ('A4', 1),
         ('G4', 1), ('F#4', 1), ('Eb4', 1), ('D4', 1), ('D4', 4)]
KILER = [('D5', .5), ('Eb5', .5), ('F#5', .5), ('G5', .5), ('A5', 1), ('G5', .5), ('F#5', .5), ('Eb5', 1), ('D5', 2)]
HOPE = [('F#4', 2), ('A4', 1), ('D5', 1), ('C#5', 1.5), ('B4', .5), ('A4', 2), ('G4', 1), ('F#4', 1), ('E4', 1), ('A4', 1), ('F#4', 4),
        ('B4', 2), ('A4', 1), ('G4', 1), ('F#4', 1.5), ('E4', .5), ('D4', 2), ('E4', 1), ('G4', 1), ('F#4', 1), ('E4', 1), ('D4', 4)]
HOPE_PROG = [('D', 1), ('A', 1), ('G', .5), ('A', .5), ('D', 1), ('G', 1), ('D', 1), ('Em', .5), ('A', .5), ('D', 1)]
COMIC = [('F3', .5), (None, .5), ('C3', .5), (None, .5), ('F3', .5), ('A3', .5), ('G3', .5), (None, .5),
         ('E3', .5), (None, .5), ('C3', .5), (None, .5), ('E3', .5), ('G3', .5), ('F3', 1)]

def hicaz_groove(t0, t1, bpm8=300, g=-14):
    """aksak 9/8 (2+2+2+3): real darbuka + frame drum"""
    e = 60.0 / bpm8; L = 9 * e; t = t0
    dm = pool('darbuka');
    while t < t1:
        for k, (b, kind, gg) in enumerate([(0, 'dum', 0), (2, 'tek', -5), (4, 'tek', -6), (6, 'dum', -2), (7, 'tek', -8), (8, 'tek', -6)]):
            tt = t + b * e
            if tt >= t1: break
            if kind == 'dum': O('framedrum', tt, g + gg - 1, -0.1, 0.3, -2, i=int(rnd(0, 3)))
            O('darbuka', tt + rnd(0, 0.005), g + gg + (0 if kind == 'tek' else 1), 0.15, 0.3, 0 if kind == 'tek' else -5)
        t += L

# ------------------------------------------------------------------ the score
def compose():
    # ===== HOOK
    HIT(0.02, 3, 'Dm', -4)
    tension_cluster(0.3, 2.0, -16, 74)
    t_ne = T('olsaydı ne')
    HIT(t_ne, 2, 'Bb', -6)
    N('cb', 26 + 12, t_ne, 2.4, 0.8, -12, 0, 0.5, 0, 2.0)
    # waking up: fragile
    t_eye = T('gözlerini açıyorsun')
    drone(t_eye, T('seni uyandıran kapıdan') + 1.0, 'D2', 'cb', -24, 0.3, 2.5, 2.0)
    N('vln', 81, t_eye + 0.5, 9.0, 0.2, -30, 0.3, 0.6, 3.0, 2.0)
    N('vla', 69, t_eye + 3.0, 7.0, 0.2, -30, -0.3, 0.6, 3.0, 2.0)
    arp(T('telefonunda onlarca'), T('ama seni uyandıran'), [('Dm', 1)], 112, 'glock', -27, 0.3, (4, 2, 3, 1), 2, 72, 0.5, 0.0)
    t_door = T('seni uyandıran kapıdan')
    tension_cluster(t_door, T('güm güm güm'), -17, 62)
    ending_at('cresc', T('güm güm güm'), -18, 0.5, i=5, length=3.0)
    t_g = T('güm güm güm')
    sub_drop(t_g, -8); O('timp_hit', t_g, -10, 0, 0.5, -5)
    # ===== KEMAL SUSPENSE
    t_pound = T('biri kapını yumrukluyor')
    t_fb = T('hani her sabah')
    ostinato(t_pound, t_fb, 104, [('Dm', 2), ('Eb', 1), ('Dm', 1)], 'vc_spic', -17, 0.6, (0, 0, 12, 0, 1, 0, 12, 0), 38)
    ostinato(t_pound + 60 / 104 * 8, t_fb, 104, [('Dm', 1)], 'cb_pizz', -17, 0.6, (0, None, None, None, 0, None, None, None), 26)
    N('vln_trem', 74, t_pound, t_fb - t_pound, 0.4, -24, 0.3, 0.5, 4, 0.8, sw=(-12, 0))
    N('vln_trem', 75, t_pound + 4, t_fb - t_pound - 4, 0.4, -26, -0.3, 0.5, 3, 0.8, sw=(-12, 0))
    # flashback: warm Kemal theme (piano)
    t_blood = T('ama bu sabah tek')
    melody(t_fb + 0.2, KEMAL[:12], 92, 'piano', -14, 0.5, 0.05, 0.3, 1.6, t1=t_blood, rel=1.0)
    arp(t_fb + 0.2, t_blood, [('F', 1), ('C', 1), ('Dm', 1), ('Bb', 1)], 92, 'piano', -22, 0.32, (0, 2, 1, 2), 2, 48)
    pad(t_fb + 0.2, t_blood, [('F', 1), ('C', 1), ('Dm', 1), ('Bb', 1)], 92, -27, 0.3, ('vc', 'vla', 'vln'), 1.0, 1.5)
    # horror return
    stinger(t_blood, -10)
    t_kan = T('gömleği kan içinde'); t_hir = T('ağzından sadece')
    tension_cluster(t_blood + 0.8, t_kan, -20, 63)
    HIT(t_kan, 1, 'Dm', -10, brass=False)
    pad(t_kan, t_hir, [('Dm', 1)], 60, -22, 0.5, ('cb', 'vc'), 0.5, 1.0, trem=True)
    for n, nm in [(38, 'tuba'), (39, 'tbn'), (44, 'tbn')]:
        N(nm, n, t_hir, 3.5, 0.6, -18, 0, 0.5, 1.5, 1.0, sw=(-10, 0))
    t_dev = T('ve kapını yumruklamaya'); t_g2 = T('ediyor güm güm güm'); t_acm = T('kapıyı açmıyorsun')
    ostinato(t_dev, t_acm, 120, [('Dm', 1), ('Eb', 1)], 'vc_spic', -15, 0.75, (0, 0, 0, 0, 1, 1, 0, 0), 38)
    tension_cluster(t_dev, t_acm, -16, 74)
    HIT(t_g2 + 0.12, 1, 'Dm', -10, brass=False)
    # decision -> false relief -> mystery
    t_dk = T('doğru karar'); t_az = T('en azından o an'); t_cu = T('çünkü o sabah'); t_hd = T('bu hikayedeki')
    pad(t_acm + 0.3, t_dk, [('Dm', 1)], 60, -26, 0.3, ('vc', 'vla'), 0.8, 1.0)
    pad(t_dk, t_az, [('F', 1)], 60, -22, 0.4, ('cb', 'vc', 'vla', 'vln'), 0.5, 1.5)
    soft_hit(t_az, -12, -6); ending_at('cresc', t_az, -20, 0.5, i=6, length=2.0)
    pad(t_az, t_cu, [('Bbm', 1)] if False else [('Bb', 1)], 60, -24, 0.4, ('cb', 'vc', 'vla'), 0.3, 1.0, trem=True)
    melody(t_cu, KILER, 96, 'harp', -15, 0.6, 0.2, 0.5, 1.8)
    melody(t_cu, KILER, 96, 'glock', -26, 0.5, -0.2, 0.6, 1.0)
    drone(t_cu, t_hd + 1.5, 'D2', 'cb_trem', -22)
    t_br = t_hd + 1.25
    ending_at('cresc', t_br, -14, 0.5, i=7, length=2.5)
    HIT(t_br, 3, 'Dm', -4)
    # ===== MAIN THEME (Peki Türkiye'de...)
    t_mt = T('peki türkiyede gerçekten'); t_son = T('ve sonunda göreceksin')
    bpm = 84; b = 60 / bpm
    ostinato(t_mt, t_son, bpm, MAIN_PROG, 'vc_spic', -15, 0.65, (0, 0, 7, 0, 12, 0, 7, 0), 38)
    ostinato(t_mt + 8 * b, t_son, bpm, MAIN_PROG, 'vln_spic', -21, 0.5, (12, 15, 19, 15, 12, 15, 19, 24), 62, accent=(0,))
    drums(t_mt, t_son, bpm, [(0, 'bdrum', -6, -3), (1.5, 'bdrum', -11, -3), (2, 'framedrum', -10, -3), (3, 'bdrum', -8, -3), (3.5, 'framedrum', -12, -2)], 4, -4)
    pad(t_mt, t_son, MAIN_PROG, bpm, -21, 0.55, ('cb', 'vc', 'vla', 'vln'), 0.4, 1.0)
    melody(t_mt + 4 * b, MAIN, bpm, 'horn', -11, 0.7, 0.15, 0.5, 1.0, t1=t_son)
    melody(t_mt + 4 * b, MAIN, bpm, 'vc', -17, 0.6, -0.2, 0.4, 1.0, transpose=-12, t1=t_son)
    t_ordu = T('ordu ne de'); t_asi = T('bir aşı türkiye')
    HIT(t_ordu, 1, 'Bb', -9); HIT(t_asi + 0.25, 1, 'A', -9)
    t_yer = T('türkiye yi kurtaracak şey yüzyıllardır')
    pad(t_son, t_ordu, [('Dm', 1)], 60, -24, 0.4, ('cb', 'vc', 'vla'), 0.6, 0.8, trem=True)
    # foreshadow hope: underground warmth
    t_d1 = T('birinci gün')
    arp(t_yer, t_d1, [('D', 1), ('Bb', 1)], 76, 'harp', -15, 0.6, (0, 1, 2, 3, 2, 1), 3, 50, 0.6, 0.0)
    pad(t_yer, t_d1, [('Dadd9', 1), ('Bb', 1)] if False else [('D', 1), ('Bb', 1)], 76, -20, 0.45, ('cb', 'vc', 'vla', 'vln'), 1.0, 1.5, top=78)
    N('tbells', 74, t_yer + 0.4, 4.0, 0.5, -20, 0.2, 0.6)
    # ===== DAY 1
    day_card(t_d1, 'Dm')
    t_ch = T('aile grubu yanıyor'); t_tey = T('teyzen sesli mesaj'); t_lim = T('limonla sarımsağı'); t_eni = T('eniştenin attığı')
    t_tv = T('televizyonu açıyorsun'); t_dk2 = T('ama durum kontrol')
    heartbeat(t_d1 + 1.2, t_ch, 66, -16)
    pad(t_d1 + 1.0, t_ch, [('Dm', 1)], 60, -26, 0.35, ('cb', 'vc'), 1.0, 1.0)
    # anxious-comic pizzicato (family group)
    bp_ = 112
    ostinato(t_ch, t_eni, bp_, [('Dm', 1), ('Gm', 1), ('A', 1), ('Dm', 1)], 'vln_pizz', -17, 0.6, (12, None, 7, None, 12, 15, 14, None), 62)
    ostinato(t_ch, t_eni, bp_, [('Dm', 1), ('Gm', 1), ('A', 1), ('Dm', 1)], 'vc_pizz', -15, 0.6, (0, None, None, None, 7, None, None, None), 38)
    melody(t_lim, COMIC + COMIC, 116, 'bsn_stac', -12, 0.7, 0.1, 0.3, 1.0, transpose=-3, t1=t_eni)
    # hospital video: urgent
    ostinato(t_eni, t_tv, 132, [('Dm', 1), ('Eb', 1)], 'vc_spic', -14, 0.75, (0, 0, 0, 0, 0, 0, 1, 1), 38)
    ending_at('snareroll', t_tv, -16, 0.4)
    tension_cluster(t_eni, t_tv, -17, 74)
    # TV news: pulses
    drums(t_tv, t_dk2, 90, [(0, 'timp_hit', -12, -5), (2, 'timp_hit', -15, -5)], 4, -2)
    pad(t_tv, t_dk2, [('Dm', 1), ('Bb', 1)], 90, -24, 0.4, ('cb', 'vc', 'vla'), 0.5, 1.0)
    HIT(t_dk2, 2, 'Dm', -8)
    t_uc = T('üç gün önce'); t_kur = T('kurallar basit'); t_ve = T('ve onları en çok'); t_112 = T('112 yi arıyorsun')
    drone(t_dk2 + 0.5, t_kur, 'D2', 'cb', -20)
    pad(t_uc, t_kur, [('Dm', 2), ('Eb', 2)], 60, -24, 0.4, ('vc', 'vla', 'vln'), 2.0, 1.5, trem=True)
    ostinato(T('başlayan şey şehrin'), t_kur, 96, [('Dm', 1)], 'cb_pizz', -15, 0.6, (0, None, None, 0, None, None, 0, None), 26)
    # rules: ticking
    ostinato(t_kur, t_112, 100, [('Dm', 1), ('Eb', 1)], 'vln_pizz', -20, 0.5, (24, 24, 24, 24, 24, 24, 24, 24), 62, accent=(0, 4))
    pad(t_kur, t_112, [('Dm', 1), ('Eb', 1)], 100, -24, 0.4, ('cb', 'vc'), 0.5, 1.0, trem=True)
    stinger(T('birkaç saat içinde') + 0.3, -14)
    # call / window: suspense
    t_m = T('market kuyruğu'); t_kids = T('öğrenciler seviniyor'); t_frz = T('en azından ilk birkaç'); t_yol = T('ama asıl panik')
    pad(t_112, t_m, [('Dm', 2), ('Bb', 2)], 60, -25, 0.35, ('cb', 'vc', 'vla'), 1.0, 1.5)
    # market: anxious busy
    bm = 124
    ostinato(t_m, t_kids, bm, [('Dm', 1), ('C', 1), ('Bb', 1), ('A', 1)], 'vln_spic', -19, 0.6, (12, 7, 12, 15, 12, 7, 10, 7), 62)
    ostinato(t_m, t_kids, bm, [('Dm', 1), ('C', 1), ('Bb', 1), ('A', 1)], 'vc_spic', -16, 0.6, (0, None, 0, 0, None, 0, 0, None), 38)
    drums(t_m, t_kids, bm, [(0, 'framedrum', -12, 0), (1, 'framedrum', -16, 2), (2, 'framedrum', -13, 0), (3, 'framedrum', -16, 2), (3.5, 'framedrum', -18, 3)], 4, -6)
    arp(t_kids, t_frz, [('D', 1)], 140, 'vln_pizz', -16, 0.7, (0, 1, 2, 3), 2, 62, 0.4, 0.0)
    for k, n in enumerate([62, 61, 60]): N('tbn_fall', 62 - k, t_frz + 0.1 + k * 0.25, 0.8, 0.6, -16, 0, 0.4)
    soft_hit(t_frz + 0.05, -14, -7)
    # roads / memleket: aksak drive
    t_mem = T('memlekete gitmek'); t_ist = T('istanbulda 15'); t_ogl = T('öğlene doğru'); t_fel = T('işte felaket')
    pad(t_yol, t_mem, [('Dm', 1), ('Bb', 1)], 90, -21, 0.5, ('cb', 'vc', 'vla', 'vln'), 0.8, 0.8, trem=True)
    ending_at('timp_roll', t_mem, -14, 0.4, length=3.5)
    HIT(t_mem, 2, 'Dm', -7)
    hicaz_groove(t_mem, t_fel, 330, -13)
    pad(t_mem, t_fel, [('Dm', 2), ('Eb', 1), ('Dm', 1)], 73.3, -23, 0.45, ('cb', 'vc', 'vla'), 0.3, 0.8)
    melody(t_mem + 3.3, HICAZ, 110, 'oboe', -18, 0.55, 0.25, 0.45, 1.0, t1=t_ist)
    melody(t_ogl, HICAZ, 110, 'flute', -18, 0.55, 0.25, 0.45, 1.0, t1=t_fel)
    HIT(t_fel, 2, 'Dm', -7)
    t_cor = T('çünkü o arabaların'); t_kur2 = T('istanbuldan kaçan her araba'); t_ogs = T('öğleden sonra kapındaki'); t_ses = T('sessizlik bu sessizlik')
    for n, nm in [(38, 'tuba'), (45, 'tbn'), (50, 'tbn')]:
        N(nm, n, t_fel + 0.5, t_ogs - t_fel - 0.5, 0.5, -22, 0, 0.5, 2.0, 1.5)
    tension_cluster(t_cor, t_kur2, -20, 62)
    ostinato(t_kur2, t_ogs, 100, [('Dm', 1), ('Eb', 1)], 'vc_spic', -16, 0.6, (0, 0, 0, 12, 0, 0, 1, 0), 38)
    pad(t_ogs, t_ses, [('Dm', 1)], 60, -26, 0.3, ('cb', 'vc'), 1.0, 2.0)
    N('vln', 86, t_ses, T('ikinci gün') - t_ses, 0.25, -32, 0.2, 0.7, 1.5, 1.0)
    tension_cluster(T('bu sessizlik seni'), T('ikinci gün'), -21, 50)
    # ===== DAY 2
    t_d2 = T('ikinci gün'); day_card(t_d2, 'Gm')
    t_tv2 = T('sabah televizyonda'); t_pan = T('pandemiden hatırlıyorsun'); t_pol = T('ama bu sefer yasağa'); t_son2 = T('sonuç internet'); t_aks = T('akşamüstü telefonun')
    drums(t_tv2, t_pan, 96, [(0, 'snare', -14, 0), (0.5, 'snare', -22, 0), (0.75, 'snare', -22, 0), (1, 'snare', -18, 0), (2, 'snare', -15, 0), (3, 'snare', -18, 0), (3.5, 'snare', -21, 0)], 4, -6)
    for n, nm in [(31, 'tuba'), (43, 'tbn'), (50, 'tbn'), (55, 'horn')]:
        N(nm, n, t_tv2, t_pan - t_tv2, 0.5, -21, 0, 0.5, 1.0, 1.0)
    melody(t_tv2 + 1.2, [('G4', .5), ('G4', .5), ('D5', 1), ('C5', .5), ('Bb4', .5), ('A4', 2)], 96, 'horn', -15, 0.7, 0.2, 0.5, 1.0)
    melody(t_pan, [('D5', 1), ('C5', 1), ('A4', 2), ('Bb4', 1), ('A4', 1), ('F4', 2)], 70, 'piano', -18, 0.35, 0, 0.35, 1.3, t1=t_pol)
    ostinato(t_pol, t_son2, 116, [('Gm', 1), ('Eb', 1), ('Cm', 1), ('D', 1)], 'vln_pizz', -18, 0.55, (12, 12, 7, 12, 15, 12, 7, 5), 55)
    ostinato(t_pol, t_son2, 116, [('Gm', 1), ('Eb', 1), ('Cm', 1), ('D', 1)], 'vc_spic', -16, 0.55, (0, None, 0, None, 0, 0, None, 0), 36)
    tension_cluster(T('ve 86 milyonluk'), t_son2, -18, 67)
    HIT(t_son2, 2, 'Gm', -8)
    # mother: tender
    t_bir = T('bir sesli mesaj'); t_ciz = T('cızırtı'); t_kay = T('kayıt bitiyor')
    pad(t_aks, t_ciz, [('Dm', 1), ('Bb', 1), ('F', 1), ('C', 1)], 66, -22, 0.4, ('cb', 'vc', 'vla', 'vln'), 1.0, 1.6)
    arp(t_aks, t_ciz, [('Dm', 1), ('Bb', 1), ('F', 1), ('C', 1)], 66, 'piano', -20, 0.35, (0, 1, 2, 3, 2, 1), 2, 57)
    melody(t_bir + 0.5, [('A4', 2), ('D5', 1), ('C5', 1), ('Bb4', 2), ('A4', 2), ('G4', 1), ('A4', 1), ('C5', 2), ('A4', 4)], 66, 'solo_vln', -17, 0.5, 0.2, 0.5, 1.0, t1=t_ciz)
    # kiler mystery (Hicaz motif)
    t_nk = T('ninenin kilerinin'); t_ama = T('ama bunu düşünmeye'); t_bal = T('çünkü o gece balkona'); t_d3 = T('üçüncü gün')
    drone(t_kay, t_ama, 'D2', 'cb', -22)
    for k in range(3):
        tt = t_nk + k * 3.2
        if tt < t_ama: melody(tt, KILER, 92, 'harp', -15, 0.6, 0.2, 0.5, 1.8); melody(tt, KILER, 92, 'glock', -27, 0.4, -0.2, 0.6, 1.0)
    N('vln', 87, t_nk, t_ama - t_nk, 0.2, -32, 0.3, 0.7, 2.0, 1.5)
    ostinato(t_ama, t_bal, 132, [('Dm', 1)], 'vln_pizz', -19, 0.6, (24, 24, 24, 24, 24, 24, 24, 24), 62)
    ending_at('cresc', t_bal, -18, 0.5, i=8, length=2.0)
    # bridges going dark: descending lament
    pad(t_bal, t_d3, [('Bb', 1), ('F', 1), ('Gm', 1), ('Dm', 1)], 56, -21, 0.45, ('cb', 'vc', 'vla', 'vln'), 1.2, 1.8)
    melody(t_bal + 1.0, [('F5', 1.5), ('E5', .5), ('D5', 2), ('C5', 1.5), ('Bb4', .5), ('A4', 2), ('G4', 2), ('A4', 2)], 56, 'vln', -16, 0.5, 0.1, 0.5, 1.0, t1=t_d3)
    # ===== DAY 3
    t_d3 = T('üçüncü gün'); day_card(t_d3, 'Dm')
    t_bh = T('sabah haberi alıyorsun'); t_tem = T('15 temmuz'); t_fsm = T('fatih sultan'); t_yss = T('yavuz sultan'); t_iki = T('istanbul ikiye')
    drums(t_bh, t_iki, 100, [(0, 'snare', -14, 0), (0.25, 'snare', -22, 0), (0.5, 'snare', -20, 0), (1, 'timp_hit', -14, -5), (2, 'snare', -15, 0), (2.5, 'snare', -21, 0), (3, 'timp_hit', -16, -5), (3.5, 'snare', -20, 0)], 4, -5)
    ostinato(t_bh, t_iki, 100, [('Dm', 1), ('Bb', 1), ('C', 1), ('Dm', 1)], 'vc_spic', -16, 0.6, (0, 0, 0, None, 0, 0, 7, None), 38)
    for tt, ch in [(t_tem, 'Dm'), (t_fsm, 'Bb'), (t_yss, 'C')]:
        HIT(tt, 1, ch, -10, brass=True)
    HIT(t_iki + 0.3, 2, 'Dm', -8)
    t_kag = T('kağıt üstünde'); t_ise = T('ama işe gidip'); t_sah = T('sahilde insanlar'); t_sen = T('sen anadolu yakasındasın'); t_ord = T('ama ordu bir adım')
    pad(t_iki + 0.5, t_ise, [('Dm', 1), ('Gm', 1)], 60, -24, 0.4, ('cb', 'vc', 'vla'), 1.0, 1.2, trem=True)
    # family split: adagio
    pad(t_ise, t_sen, [('Dm', 1), ('Bb', 1), ('Gm', 1), ('A', 1)], 58, -21, 0.45, ('cb', 'vc', 'vla', 'vln'), 1.0, 1.8)
    melody(t_ise + 0.6, [('A4', 2), ('Bb4', 1), ('A4', 1), ('G4', 2), ('F4', 2), ('E4', 1), ('F4', 1), ('G4', 2), ('A4', 4)], 58, 'vc', -14, 0.55, -0.2, 0.45, 1.0, transpose=-12, t1=t_sen)
    # hopeful glimmer (Üsküdar / road on your side)
    pad(t_sen, t_ord, [('F', 1), ('C', 1), ('Bb', 1), ('F', 1)], 72, -22, 0.4, ('vc', 'vla', 'vln'), 0.8, 1.2)
    arp(t_sen, t_ord, [('F', 1), ('C', 1), ('Bb', 1), ('F', 1)], 72, 'harp', -17, 0.55, (0, 1, 2, 3), 2, 53, 0.5, 0.0)
    # quarantine & cage
    t_kaf = T('15 milyon insan'); t_gec = T('ve o gece ilk kez'); t_dug = T('düğünlerde havaya'); t_osl = T('o silahlar artık'); t_bgs = T('bütün gece süren'); t_ele = T('elektrikler gidiyor')
    ending_at('timp_roll', t_kaf, -14, 0.4, length=8.0)
    for n, nm in [(38, 'tuba'), (45, 'tbn'), (50, 'horn'), (51, 'horn')]:
        N(nm, n, t_ord, t_kaf - t_ord, 0.6, -21, 0, 0.5, 3.0, 1.0, sw=(-12, 0))
    ostinato(t_ord, t_kaf, 108, [('Dm', 1), ('Eb', 1)], 'vc_spic', -16, 0.65, (0, 0, 0, 0, 1, 1, 1, 1), 38)
    HIT(t_kaf + 0.35, 3, 'Dm', -6)
    pad(t_kaf + 0.5, t_gec, [('Dm', 1)], 60, -24, 0.4, ('cb', 'vc', 'vla'), 0.5, 1.5, trem=True)
    drone(t_gec, t_dug, 'D1', 'cb', -20)
    drone(t_osl, t_bgs, 'D1', 'cb', -20)
    tension_cluster(T('yani her silah sesi'), t_bgs, -19, 61)
    heartbeat(t_bgs, t_ele, 70, -15)
    tension_cluster(t_bgs + 1, t_ele, -16, 74)
    ending_at('cresc', t_ele, -15, 0.5, i=9, length=4.0)
    # ===== DAY 4: darkness
    t_d4 = T('birinci gün', after=t_ele); day_card(t_d4, 'Dm', soft=True)
    t_ezo = T('sabah ezanı okunmadı'); t_hay = T('hayatın boyunca'); t_isan = T('işte o an anlıyorsun'); t_ogs2 = T('öğleden sonra kapın çalıyor')
    drone(t_d4 + 1, t_ezo, 'D1', 'cb', -22, 0.3, 4.0, 3.0)
    for i, wg in enumerate(range(3)):
        O('wineglass', t_d4 + 2 + i * 5.5, -28, rnd(-0.5, 0.5), 0.7, -2)
    O('cymb_bow', t_d4 + 8, -26, 0, 0.7, i=0)
    # lament (Hicaz) on clarinet/flute
    melody(t_ezo, [('D5', 2), ('Eb5', 1), ('D5', 1), ('C5', 1), ('Bb4', 1), ('A4', 2), ('Bb4', 1), ('A4', 1), ('G4', 1), ('F#4', 1), ('Eb4', 1), ('D4', 3)], 60, 'clar', -15, 0.5, 0.15, 0.55, 1.0, t1=t_isan)
    drone(t_ezo, t_isan, 'D2', 'vc', -24)
    ostinato(t_isan, t_ogs2, 90, [('Dm', 1), ('Bb', 1)], 'cb_pizz', -16, 0.6, (0, None, 0, None, 0, None, 0, None), 26)
    pad(t_isan, t_ogs2, [('Dm', 1), ('Bb', 1)], 90, -24, 0.4, ('vc', 'vla', 'vln'), 1.0, 1.0)
    # apartment meeting: comic warmth
    t_tik = T('tık tık tık'); t_gel = T('gelen apartman yöneticisi'); t_evet = T('evet kıyamet kopmuş'); t_amat = T('ama türkiyede bir apartmanda')
    t_mum = T('mum ışığında 16'); t_ark = T('arkadaşlar bu durumda'); t_her = T('herkes ona bakıyor'); t_gul = T('sonra biri gülmeye'); t_kap = T('kapıcı recep'); t_don = T('ama toplantıdan dönerken')
    pad(t_ogs2, t_gel, [('F', 1)], 60, -26, 0.3, ('vc', 'vla'), 1.0, 1.0)
    pad(t_gel, t_evet, [('F', 1), ('Bb', 1)], 70, -24, 0.4, ('cb', 'vc', 'vla'), 0.8, 1.0)
    HIT(t_evet, 1, 'Dm', -11)
    melody(t_amat + 0.5, [('F3', .5), ('C3', .5), ('F3', 1), (None, 1), ('C3', .5), ('F2', 1.5)], 100, 'bsn_stac', -10, 0.8, 0.1, 0.3, 1.0)
    ostinato(t_mum, t_her, 108, [('F', 1), ('C', 1), ('Dm', 1), ('Bb', 1)], 'vln_pizz', -18, 0.55, (12, None, 7, None, 16, None, 7, None), 60)
    ostinato(t_mum, t_her, 108, [('F', 1), ('C', 1), ('Dm', 1), ('Bb', 1)], 'vc_pizz', -15, 0.55, (0, None, None, None, 7, None, None, None), 36)
    melody(T('birinci gündem') + 0.5, COMIC * 3, 108, 'bsn_stac', -13, 0.6, 0.15, 0.3, 1.0, t1=t_ark)
    melody(t_ark, [('C4', 1), ('B3', 1), ('Bb3', 1), ('A3', 3)], 80, 'tbn', -16, 0.5, 0, 0.4, 1.0, t1=t_her)
    # comic silence at "herkes ona bakıyor", then warm laughter
    pad(t_gul + 0.4, t_kap, [('F', 1), ('Bb', 1), ('C', 1), ('F', 1)], 84, -20, 0.5, ('cb', 'vc', 'vla', 'vln'), 0.6, 1.2)
    arp(t_gul + 0.4, t_kap, [('F', 1), ('Bb', 1), ('C', 1), ('F', 1)], 84, 'piano', -19, 0.4, (0, 1, 2, 3), 2, 60)
    melody(t_kap + 0.3, [('C4', .5), ('F4', .5), ('A4', 1), ('G4', .5), ('F4', .5), ('C5', 2)], 92, 'horn', -15, 0.6, 0.2, 0.5)
    pad(t_kap, t_don, [('F', 1), ('C', 1)], 92, -25, 0.4, ('vc', 'vla'), 0.5, 1.0)
    # back to suspense: Kemal's empty home
    t_kya = T('kemal amcanın kapısı aralık'); t_d5 = T('beşinci gün')
    ostinato(t_don, t_d5, 84, [('Dm', 1)], 'cb_pizz', -16, 0.6, (0, None, None, None, 1, None, None, None), 26)
    melody(t_kya, KILER, 84, 'glock', -24, 0.5, 0, 0.6, 1.0)
    pad(t_kya, t_d5, [('Dm', 1), ('Eb', 1)], 60, -24, 0.4, ('cb', 'vc', 'vla'), 1.5, 1.5, trem=True)
    # ===== DAY 5
    day_card(t_d5, 'Dm')
    t_ama5 = T('ama önce aklına'); t_tit = T('titrek bir el'); t_ben = T('beni markette'); t_rad = T('radyoyu da al'); t_hak = T('hakkını helal et kemal'); t_osab = T('o sabah kapını yumruklayan')
    ostinato(t_d5 + 1.0, t_ama5, 96, [('Dm', 1)], 'vln_pizz', -21, 0.45, (24, None, 24, None, 24, None, 24, None), 62)
    drone(t_d5 + 1.0, t_ama5, 'D2', 'cb', -22)
    N('vln', 88, t_ama5, t_tit - t_ama5, 0.2, -31, 0.3, 0.7, 2.0, 1.5); N('vln', 81, t_ama5 + 1, t_tit - t_ama5 - 1, 0.2, -31, -0.3, 0.7, 2.0, 1.5)
    O('wineglass', t_ama5 + 2, -27, 0.3, 0.7)
    # THE LETTER: Kemal theme, piano then strings
    bk = 70
    melody(t_tit, KEMAL, bk, 'piano', -12, 0.55, 0.0, 0.3, 1.6, t1=t_osab, rel=1.2)
    arp(t_tit, t_osab, KEMAL_PROG, bk, 'piano', -21, 0.32, (0, 2, 1, 2), 2, 45)
    pad(t_ben - 1, t_osab, KEMAL_PROG[1:] + KEMAL_PROG[:1], bk, -22, 0.45, ('cb', 'vc', 'vla', 'vln'), 1.5, 2.0)
    melody(t_rad, KEMAL[12:], bk, 'vc', -16, 0.55, -0.2, 0.45, 1.0, transpose=-12, t1=t_osab)
    for n, nm in [(50, 'horn'), (57, 'horn')]:
        N(nm, n, t_hak, 3.5, 0.5, -19, 0.2, 0.55, 1.2, 2.0)
    # reveal: tragic swell
    t_ves = T('ve sen ondan korktun'); t_not = T('notu cebine')
    pad(t_osab, t_ves + 0.8, [('Dm', 1), ('Bb', 1), ('Gm', 1), ('A7', 1)], 64, -18, 0.75, ('cb', 'vc', 'vla', 'vln'), 0.6, 2.0, top=79)
    melody(t_osab + 0.3, [('A5', 2), ('F5', 1), ('D5', 1), ('D6', 2), ('C6', 1), ('Bb5', 1), ('Bb5', 2), ('A5', 1), ('G5', 1), ('A5', 4)], 64, 'vln', -13, 0.75, 0.1, 0.5, 1.0, t1=t_ves + 0.8)
    O('timp_roll', t_osab, -18, 0, 0.4, cut=4.0, fadein=3.5)
    HIT(t_ves, 1, 'Dm', -12, brass=False)
    # streets: desolate solo cello
    t_koc = T('kocaman kara'); t_kb = T('karabaş birden'); t_kos = T('koşuyorsun'); t_bak = T('bir bakkalın'); t_kar = T('karanlıkta nefes')
    drone(t_not, t_koc, 'D2', 'cb', -22)
    melody(t_not + 0.5, [('D4', 2), ('F4', 1), ('E4', 1), ('D4', 2), ('A3', 2), ('Bb3', 2), ('A3', 4)], 64, 'vc', -16, 0.5, -0.1, 0.5, 1.0, transpose=-12, t1=t_koc)
    # Karabaş: warm horn motif
    pad(t_koc, t_kb, [('F', 1), ('Bb', 1)], 76, -22, 0.45, ('cb', 'vc', 'vla'), 0.6, 1.0)
    melody(t_koc + 0.3, [('C4', 1), ('F4', 1), ('G4', .5), ('A4', .5), ('F4', 2), ('C5', 1.5), ('Bb4', .5), ('A4', 2)], 76, 'horn', -14, 0.6, 0.15, 0.5, t1=t_kb)
    tension_cluster(t_kb, T('ara sokaktan üç tane'), -17, 62)
    stinger(T('ara sokaktan üç tane') + 0.4, -10)
    # chase: action
    ba = 150
    ostinato(t_kos, t_bak + 3.4, ba, [('Dm', 1), ('Bb', 1), ('C', 1), ('A', 1)], 'vc_spic', -13, 0.8, (0, 0, 12, 0, 0, 12, 0, 12), 38)
    ostinato(t_kos, t_bak + 3.4, ba, [('Dm', 1), ('Bb', 1), ('C', 1), ('A', 1)], 'vln_spic', -18, 0.7, (12, 15, 19, 15, 12, 15, 19, 15), 62)
    drums(t_kos, t_bak + 3.4, ba, [(0, 'bdrum', -6, -3), (0.75, 'bdrum', -10, -3), (1, 'snare', -12, 0), (2, 'bdrum', -7, -3), (2.5, 'bdrum', -10, -3), (3, 'snare', -11, 0), (3.5, 'snare', -16, 0)], 4, -4)
    HIT(t_kos, 2, 'Dm', -8)
    sh = shots('shutter')[0]; t_sl = sh['t0'] + sh['p']['_k'][0]
    HIT(t_sl, 2, 'Dm', -7)
    # dark shop: breath held
    t_kop = T('köpekler onları'); t_vez = T('ve zombiler köpeklerle'); t_yani = T('yani türkiye nin sokaklarındaki'); t_bak2 = T('bakkaldan su'); t_ned = T('neden bilmiyorsun'); t_ogu = T('o günden sonra'); t_vgm = T('ve gece mum ışığında')
    N('vln', 88, t_kar, t_yani - t_kar, 0.2, -30, 0.2, 0.7, 1.0, 1.0); N('vln', 89, t_kar + 0.5, t_vez - t_kar, 0.2, -32, -0.2, 0.7, 1.0, 1.0)
    heartbeat(t_kar, t_vez, 84, -17)
    arp(t_kop, t_vez, [('Dm', 1)], 80, 'harp', -19, 0.5, (0, 2, 1, 3), 2, 62, 0.5, 0.0)
    # early warning system / staying human: warm
    pad(t_yani, t_vgm, [('F', 1), ('C', 1), ('Dm', 1), ('Bb', 1)], 72, -22, 0.45, ('cb', 'vc', 'vla', 'vln'), 0.8, 1.5)
    arp(t_yani, t_ned, [('F', 1), ('C', 1), ('Dm', 1), ('Bb', 1)], 72, 'harp', -18, 0.5, (0, 1, 2, 3, 2, 1), 3, 53, 0.5, 0.0)
    melody(t_ned, [('A4', 2), ('C5', 1), ('Bb4', 1), ('A4', 2), ('F4', 2), ('G4', 2), ('A4', 2), ('F4', 4)], 72, 'piano', -14, 0.5, 0, 0.3, 1.5, t1=t_vgm, rel=1.0)
    melody(t_ogu + 0.2, [('C4', 1), ('F4', 1), ('G4', .5), ('A4', .5), ('F4', 2)], 72, 'horn', -17, 0.5, 0.2, 0.5, t1=t_vgm)
    # ===== DAY 6
    t_d6 = T('altıncı gün')
    drone(t_vgm, t_d6, 'D2', 'cb', -24)
    day_card(t_d6, 'Dm', soft=True)
    t_sab = T('sabaha karşı bir ses'); t_bubi = T('bu bir acil durum'); t_ic = T('iç anadoluda'); t_ys = T('yeraltı şehirleri bunu'); t_kaf2 = T('kafanda bir şey')
    drone(t_d6 + 1, t_sab, 'D1', 'cb', -22)
    O('wineglass', t_d6 + 3, -28, 0, 0.7)
    arp(t_sab, t_bubi, [('D', 1)], 90, 'harp', -18, 0.5, (0, 1, 2, 3, 4), 2, 50, 0.5, 0.0)
    pad(t_bubi, t_ys, [('Dm', 1), ('Bb', 1), ('F', 1), ('C', 1)], 76, -23, 0.45, ('cb', 'vc', 'vla', 'vln'), 0.8, 1.2)
    pad(t_ic, t_ys, [('F', 1), ('C', 1)], 76, -24, 0.4, ('vln',), 0.8, 1.2, top=79)
    # Yeraltı şehirleri: wonder
    HIT(t_ys, 2, 'D', -9, brass=True)
    O('marktree', t_ys + 0.1, -16, 0.2, 0.6, i=0)
    pad(t_ys, t_kaf2 + 2, [('D', 1)], 60, -19, 0.55, ('cb', 'vc', 'vla', 'vln'), 0.3, 2.0, top=78)
    N('tbells', 74, t_ys + 0.05, 5, 0.6, -18, 0.2, 0.6)
    # Derinkuyu: Anatolian mystery theme
    t_kap2 = T('kapadokyada yerin altında'); t_63 = T('1963te nevşehirde'); t_vog = T('ve o geçit'); t_bus = T('bu şehir tam olarak'); t_ekd = T('en kalabalık döneminde')
    t_gir = T('girişleri yüzlerce'); t_veo = T('ve en önemlisi'); t_ann = T('annenin sesi kulağında'); t_yok = T('yoksa çocukken'); t_kb2 = T('karar belli'); t_kem = T('kemal amcanın anahtarını'); t_d7 = T('yedinci gün')
    hicaz_groove(t_kap2, t_63, 260, -15)
    drone(t_kap2, t_63 + 1, 'D2', 'cb', -20)
    arp(t_kap2, t_63, [('Dhic', 1)], 88, 'harp', -17, 0.55, (0, 1, 2, 3, 2, 1), 2, 50, 0.5, 0.0)
    melody(t_kap2 + 2.0, HICAZ, 88, 'flute', -15, 0.55, 0.2, 0.5, 1.0, t1=t_63)
    for hk in shots('wall1963', 'break')[0]['p']['_k']:
        pass
    pad(t_63, t_vog, [('Dm', 1)], 60, -24, 0.4, ('cb', 'vc'), 0.8, 1.2, trem=True)
    # descend into the city
    for k in range(5):
        N('harp', 74 - k * 5, t_vog + k * 0.45, 2.0, 0.6, -16, 0.2 - k * 0.1, 0.6)
    pad(t_vog, t_bus, [('Dm', 1), ('Bb', 1)], 66, -21, 0.5, ('cb', 'vc', 'vla', 'vln'), 0.8, 1.5, top=70)
    tension_cluster(t_bus, t_ekd, -18, 62)
    # 20.000 people: grand
    pad(t_ekd, t_gir, [('D', 1), ('Bb', 1), ('F', 1), ('C', 1)], 70, -19, 0.55, ('cb', 'vc', 'vla', 'vln'), 0.6, 1.5)
    melody(t_ekd + 0.4, [('D4', 1), ('A4', 1), ('G4', .5), ('F4', .5), ('E4', 1), ('F4', 2), ('C5', 2)], 70, 'horn', -14, 0.65, 0.15, 0.5, t1=t_gir)
    drums(t_ekd, t_gir, 70, [(0, 'timp_hit', -12, -5), (2, 'timp_hit', -15, -3)], 4, -2)
    # stone doors / hidden entrances: mysterious motion
    hicaz_groove(t_gir, t_ann, 280, -17)
    arp(t_gir, t_ann, [('Dhic', 1), ('Cm', 1)], 84, 'harp', -18, 0.5, (0, 2, 1, 3), 2, 50, 0.5, 0.0)
    pad(t_gir, t_ann, [('Dm', 1), ('Cm', 1)], 84, -25, 0.4, ('cb', 'vc', 'vla'), 1.0, 1.0)
    # mother's echo -> kiler reveal: wonder
    melody(t_ann + 0.5, KILER, 88, 'harp', -15, 0.6, 0.2, 0.6, 1.8); melody(t_ann + 0.5, KILER, 88, 'glock', -25, 0.45, -0.2, 0.7)
    drone(t_ann, t_yok + 1, 'D2', 'vc', -24)
    O('marktree', t_yok, -15, 0, 0.6, i=1)
    pad(t_yok, t_kb2, [('D', 1), ('G', 1)], 60, -20, 0.5, ('cb', 'vc', 'vla', 'vln'), 0.5, 1.5, top=78)
    # decision: determination march
    bd = 92
    ostinato(t_kb2, t_kem, bd, [('Dm', 1), ('Bb', 1), ('F', 1), ('C', 1)], 'vc_spic', -15, 0.65, (0, 0, 7, 0, 12, 0, 7, 0), 38)
    drums(t_kb2, t_kem, bd, [(0, 'bdrum', -8, -3), (2, 'bdrum', -10, -3), (3.5, 'snare', -18, 0)], 4, -4)
    pad(t_kb2, t_kem, [('Dm', 1), ('Bb', 1), ('F', 1), ('C', 1)], bd, -21, 0.55, ('cb', 'vc', 'vla', 'vln'), 0.5, 1.0)
    melody(T('ve 700 kilometreden'), MAIN[:11], bd, 'horn', -14, 0.65, 0.15, 0.5, t1=t_kem)
    pad(t_kem, t_d7, [('Dm', 1)], 60, -24, 0.45, ('cb', 'vc', 'vla'), 1.0, 1.5)
    N('tbells', 62, t_kem + 0.2, 4, 0.4, -24, 0.3, 0.6)
    # ===== DAY 7
    day_card(t_d7, 'Dm')
    t_sb = T('sabah beşte'); t_kbs = T('köşede beyaz'); t_ama7 = T('ama arabanın yanında'); t_kz = T('kemal amca ya da'); t_ars = T('arabasının yanında'); t_kh = T('karabaş hırlıyor'); t_kyd = T('kemal amca yavaşça')
    t_kia = T('kapıyı açıp içeri'); t_anc = T('anahtarı çeviriyorsun'); t_tip = T('tıpkı o sabahki'); t_ucd = T('üçüncü denemede'); t_gar = T('garajdan çıkarken'); t_hk2 = T('hakkını helal et kemal amca'); t_sim = T('şimdi asıl sorun')
    drone(t_sb, t_ama7, 'D1', 'cb', -20); heartbeat(t_sb + 1, t_kbs, 64, -18)
    N('vln', 86, t_kbs, t_ama7 - t_kbs, 0.25, -30, 0.2, 0.7, 1.0, 1.0)
    tension_cluster(t_ama7, t_kz, -16, 62)
    stinger(t_kz, -7)
    # tragic: Kemal theme in minor on solo cello / low strings
    melody(t_ars, [('A3', 2), ('D4', 1), ('C4', 1), ('A3', 2), ('Bb3', 2), ('A3', 4)], 60, 'vc', -14, 0.55, -0.1, 0.5, 1.0, t1=t_kh)
    pad(t_ars, t_kh, [('Dm', 1), ('Gm', 1)], 60, -24, 0.4, ('cb', 'vla', 'vln'), 1.0, 1.5)
    tension_cluster(t_kh, t_kia, -15, 63)
    # frantic: escape attempts
    ostinato(t_kia, t_ucd, 152, [('Dm', 1), ('Eb', 1)], 'vc_spic', -13, 0.8, (0, 0, 12, 0, 1, 1, 13, 1), 38)
    ostinato(t_anc, t_ucd, 152, [('Dm', 1), ('Eb', 1)], 'vln_spic', -19, 0.7, (24, 25, 24, 25, 24, 25, 24, 25), 62)
    drums(t_kia, t_ucd, 152, [(0, 'bdrum', -7, -3), (1, 'bdrum', -10, -3), (2, 'bdrum', -7, -3), (3, 'snare', -12, 0)], 4, -5)
    soft_hit(t_tip, -11, -6)
    HIT(t_ucd + 0.6, 3, 'D', -5, riser=2.5)
    # goodbye
    melody(t_gar + 0.4, KEMAL[:12], 66, 'piano', -14, 0.5, 0, 0.3, 1.6, t1=t_sim, rel=1.0)
    pad(t_gar, t_sim, [('Dm', 1), ('Gm', 1), ('C', .5), ('A7', .5), ('Dm', 1)], 66, -21, 0.5, ('cb', 'vc', 'vla', 'vln'), 1.2, 2.0)
    # quarantine line: tension
    t_tem2 = T('temde kocaeli'); t_oto = T('otoyola hiç'); t_tor = T('torpidoda kemal'); t_koy = T('köy yollarından'); t_sah2 = T('şahin in en büyük'); t_aks2 = T('akşam olduğunda'); t_amo = T('ama önünde anadolu'); t_bd8 = T('bolu dağı sekizinci')
    drums(t_sim, t_oto, 100, [(0, 'snare', -16, 0), (0.5, 'snare', -22, 0), (1, 'timp_hit', -15, -5), (2, 'snare', -17, 0), (3, 'timp_hit', -16, -5)], 4, -5)
    pad(t_sim, t_oto, [('Dm', 1), ('Eb', 1)], 100, -23, 0.45, ('cb', 'vc', 'vla'), 0.5, 1.0, trem=True)
    # road trip: adventure (aksak + strings + flute)
    pad(t_oto, t_tor, [('Dm', 1)], 60, -24, 0.4, ('vc', 'vla'), 0.5, 1.0)
    arp(t_tor, t_koy, [('Dm', 1), ('F', 1)], 80, 'harp', -18, 0.5, (0, 1, 2, 3), 2, 57, 0.5, 0.0)
    hicaz_groove(t_koy, t_amo, 320, -14)
    ostinato(t_koy, t_amo, 106.7, [('Dm', 1), ('C', 1), ('Bb', 1), ('A', 1)], 'vc_spic', -16, 0.6, (0, None, 7, 0, 12, None, 7, 5), 38)
    pad(t_koy, t_amo, [('Dm', 1), ('C', 1), ('Bb', 1), ('A', 1)], 106.7, -23, 0.45, ('cb', 'vla', 'vln'), 0.4, 1.0)
    melody(t_koy + 1.0, HICAZ, 106, 'flute', -14, 0.6, 0.2, 0.45, t1=t_aks2)
    melody(t_sah2 + 0.5, [('D4', 1), ('F4', 1), ('A4', 1), ('C5', 1), ('Bb4', 2), ('A4', 2)], 106, 'oboe', -17, 0.55, -0.2, 0.45, t1=t_aks2)
    pad(t_aks2, t_amo, [('F', 1), ('C', 1)], 80, -22, 0.45, ('cb', 'vc', 'vla', 'vln'), 0.6, 1.2)
    for n, nm in [(38, 'tuba'), (45, 'tbn'), (50, 'horn')]:
        N(nm, n, t_amo, t_bd8 - t_amo + 0.5, 0.6, -20, 0, 0.5, 2.5, 1.0, sw=(-12, 0))
    HIT(t_bd8, 2, 'Dm', -7)
    # ===== DAY 8: tunnel
    t_d8 = T('sekizinci gün'); day_card(t_d8, 'Dm', soft=True)
    t_gy = T('gece yarısı bolu'); t_bay = T('bayram yolculuklarında'); t_abu = T('ama bu sefer içerisi'); t_tun = T('tünelin ortasında'); t_kbs2 = T('karabaş birden susuyor'); t_sca = T('sonra çok alçak')
    t_oto2 = T('otobüsün içinde bir şey'); t_vto = T('ve tam o anda'); t_tsa = T('tünelde sadece senin'); t_kno = T('kemal amcanın notu aklına'); t_elf = T('el fenerini kapıp'); t_ark2 = T('arkanda otobüsün'); t_arb = T('arabaya atlıyorsun'); t_tc = T('tünelden çıktığında'); t_akt = T('arka koltukta karabaş'); t_d9 = T('dokuzuncu gün')
    drone(t_gy, t_bay, 'D1', 'cb', -21)
    pad(t_bay, t_abu, [('F', 1), ('C', 1)], 76, -23, 0.4, ('vc', 'vla', 'vln'), 0.5, 1.0)
    drone(t_abu, t_tun, 'D1', 'cb', -20); tension_cluster(t_abu + 2, t_tun, -21, 62)
    heartbeat(T('farları yakıp'), t_tun, 62, -18)
    stinger(t_tun + 0.2, -10)
    pad(t_tun + 1, t_kbs2, [('Dm', 1)], 60, -24, 0.4, ('cb', 'vc'), 1.0, 0.6, trem=True)
    drone(t_sca, t_vto, 'D1', 'cb_trem', -18)
    tension_cluster(t_oto2, t_vto, -15, 74)
    heartbeat(t_tsa, t_kno, 92, -15)
    melody(t_kno + 0.3, [('A4', 1), ('D5', 1), ('C5', 1), ('A4', 1)], 70, 'piano', -16, 0.45, 0, 0.35, 1.6)
    ending_at('timp_roll', t_arb, -13, 0.4, length=8.0)
    tension_cluster(t_elf, t_arb, -14, 75)
    ostinato(t_ark2, t_arb, 150, [('Dm', 1)], 'vc_spic', -14, 0.8, (0, 1, 0, 1, 0, 1, 0, 1), 38)
    HIT(T('kontak şahin çalışıyor'), 3, 'Dm', -5)
    ostinato(t_arb, t_tc, 156, [('Dm', 1), ('Bb', 1), ('C', 1), ('A', 1)], 'vc_spic', -13, 0.8, (0, 0, 12, 0, 0, 12, 0, 12), 38)
    ostinato(t_arb, t_tc, 156, [('Dm', 1), ('Bb', 1), ('C', 1), ('A', 1)], 'vln_spic', -18, 0.7, (12, 15, 19, 15, 12, 15, 19, 24), 62)
    drums(t_arb, t_tc, 156, [(0, 'bdrum', -6, -3), (0.75, 'bdrum', -10, -3), (1, 'snare', -11, 0), (2, 'bdrum', -7, -3), (3, 'snare', -10, 0), (3.5, 'snare', -15, 0)], 4, -4)
    melody(T('aynaları tünelin'), MAIN[:11], 156, 'horn', -11, 0.85, 0.15, 0.5, t1=t_tc)
    # SUNRISE: release
    HIT(t_tc, 3, 'D', -6)
    pad(t_tc, t_akt, [('D', 1), ('A', 1), ('Bm', 1), ('G', 1)], 70, -17, 0.7, ('cb', 'vc', 'vla', 'vln'), 0.3, 2.0, top=78)
    melody(t_tc + 0.4, HOPE[:11], 70, 'horn', -12, 0.75, 0.15, 0.55, t1=t_akt + 1)
    pad(t_akt, t_d9, [('F', 1), ('C', 1), ('Dm', 1), ('Bb', 1)], 66, -22, 0.4, ('vc', 'vla', 'vln'), 1.0, 2.0)
    arp(t_akt, t_d9, [('F', 1), ('C', 1), ('Dm', 1), ('Bb', 1)], 66, 'piano', -19, 0.35, (0, 1, 2, 3, 2, 1), 2, 57)
    # ===== DAY 9: Anatolia
    day_card(t_d9, 'D', soft=True)
    t_sal = T('salgın burada'); t_seb = T('sebebi basit'); t_aman = T('ama anadolu nun da'); t_ord2 = T('ordu ankara nın'); t_ben2 = T('benzin bitmek'); t_d10 = T('onuncu gün')
    pad(t_d9 + 1, t_seb, [('Dm', 1), ('C', 1), ('Bb', 1), ('C', 1)], 64, -22, 0.45, ('cb', 'vc', 'vla', 'vln'), 1.0, 1.5)
    melody(t_d9 + 1.5, HICAZ, 76, 'flute', -15, 0.55, 0.2, 0.55, t1=t_seb)
    arp(t_d9 + 1, t_seb, [('Dm', 1), ('C', 1), ('Bb', 1), ('C', 1)], 64, 'harp', -19, 0.5, (0, 1, 2, 3), 2, 57, 0.5, 0.0)
    ostinato(t_seb, t_aman, 104, [('Dm', 1), ('F', 1), ('C', 1), ('Dm', 1)], 'vln_pizz', -19, 0.5, (12, None, 7, None, 12, 14, 12, None), 62)
    ostinato(t_seb, t_aman, 104, [('Dm', 1), ('F', 1), ('C', 1), ('Dm', 1)], 'vc_pizz', -16, 0.55, (0, None, None, None, 7, None, None, None), 36)
    HIT(t_aman, 1, 'Dm', -11)
    drums(t_ord2, t_ben2, 96, [(0, 'snare', -16, 0), (0.5, 'snare', -22, 0), (1, 'timp_hit', -15, -5), (2, 'snare', -17, 0), (3, 'timp_hit', -16, -5)], 4, -6)
    pad(t_ord2, t_ben2, [('Dm', 1), ('Bb', 1)], 96, -23, 0.45, ('cb', 'vc', 'vla'), 0.5, 1.0)
    ostinato(t_ben2, t_d10, 96, [('Dm', 1)], 'vln_pizz', -21, 0.45, (24, None, 24, None, 24, None, 24, None), 62)
    drone(t_ben2, t_d10, 'D2', 'cb', -22)
    # ===== DAY 10: village
    day_card(t_d10, 'Dm', soft=True)
    t_dur = T('dur nereden'); t_istd = T('istanbuldan bu kelime'); t_sen2 = T('seni arabadan'); t_cik = T('çıkınca muhtarın'); t_sk = T('sonra seni köy kahvesine'); t_ist2 = T('işte burada türkiye')
    t_kis = T('kışlık'); t_sehm = T('şehirde marketler'); t_ama10 = T('ama bir anadolu teyzesinin'); t_su = T('su için kuyular'); t_yasn = T('yani sanılanın'); t_muh = T('muhtar yolun kalanı'); t_d11 = T('on birinci gün')
    hicaz_groove(t_d10 + 1, t_dur, 220, -16)
    pad(t_d10 + 1, t_dur, [('Dm', 1)], 60, -24, 0.4, ('cb', 'vc', 'vla'), 1.0, 1.0, trem=True)
    HIT(t_dur, 1, 'Dm', -10); stinger(t_istd + 0.1, -11)
    heartbeat(t_sen2, t_cik, 80, -17)
    N('vln', 86, t_sen2, t_cik - t_sen2, 0.25, -30, 0, 0.7, 1.0, 1.0)
    pad(t_cik, t_sk, [('Dm', 1), ('Bb', 1)], 60, -22, 0.45, ('cb', 'vc', 'vla', 'vln'), 0.8, 1.5)
    # kahve: warm Anatolian
    pad(t_sk, t_ist2, [('Dm', 1), ('C', 1)], 76, -22, 0.45, ('cb', 'vc', 'vla'), 1.0, 1.2)
    melody(t_sk + 0.5, [('D4', 1), ('E4', 1), ('F4', 1), ('G4', .5), ('F4', .5), ('E4', 2), ('D4', 2)], 76, 'flute', -16, 0.5, 0.2, 0.5, t1=t_ist2)
    HIT(t_ist2 + 0.2, 2, 'D', -8)
    HIT(t_kis, 1, 'D', -9, brass=False)
    # proud & playful pantry
    hicaz_groove(t_kis + 0.6, t_yasn, 330, -15)
    ostinato(t_kis + 0.6, t_yasn, 110, [('D', 1), ('G', 1), ('A', 1), ('D', 1)], 'vln_pizz', -18, 0.6, (12, None, 7, 12, 16, None, 14, None), 62)
    ostinato(t_kis + 0.6, t_yasn, 110, [('D', 1), ('G', 1), ('A', 1), ('D', 1)], 'vc_pizz', -15, 0.6, (0, None, None, 7, 0, None, None, None), 38)
    melody(t_ama10, [('A4', 1), ('B4', .5), ('A4', .5), ('F#4', 1), ('D4', 1), ('E4', 1), ('F#4', 1), ('D4', 2)], 110, 'horn', -15, 0.6, 0.2, 0.5, t1=t_yasn)
    pad(t_yasn, t_d11, [('D', 1), ('G', 1), ('A', 1), ('D', 1)], 80, -20, 0.55, ('cb', 'vc', 'vla', 'vln'), 0.5, 1.5)
    melody(t_yasn + 0.4, HOPE[:11], 80, 'vln', -15, 0.6, 0.1, 0.5, t1=t_muh)
    arp(t_muh, t_d11, [('D', 1), ('G', 1)], 80, 'harp', -18, 0.5, (0, 1, 2, 3), 2, 57, 0.5, 0.0)
    # ===== DAY 11: arrival
    day_card(t_d11, 'D', soft=True)
    t_ufk = T('ve sonunda ufukta'); t_koy2 = T('köyüne giriyorsun'); t_kal = T('kalbin sıkışıyor'); t_nin = T('ninenin evine koşuyorsun'); t_an2 = T('annenin sesi yine'); t_kap3 = T('kapağı açıyorsun')
    t_cg = T('bir çocuk gülüşü'); t_bas = T('basamaklara iniyorsun'); t_son3 = T('sonunda kayaya'); t_vk = T('ve kalabalığın arasından'); t_d30 = T('otuzuncu gün')
    pad(t_ufk, t_koy2, [('D', 1), ('Bm', 1), ('G', 1), ('A', 1)], 70, -19, 0.6, ('cb', 'vc', 'vla', 'vln'), 0.8, 2.0, top=78)
    arp(t_ufk, t_koy2, [('D', 1), ('Bm', 1), ('G', 1), ('A', 1)], 70, 'harp', -17, 0.55, (0, 1, 2, 3, 2, 1), 3, 50, 0.6, 0.0)
    N('tbells', 74, t_ufk + 0.2, 5, 0.5, -20, 0.2, 0.6)
    drone(t_koy2, t_an2, 'D2', 'cb', -22)
    O('wineglass', t_koy2 + 1, -27, 0, 0.7)
    tension_cluster(t_koy2 + 2, t_kal, -21, 62)
    heartbeat(t_kal, t_nin + 1, 80, -14)
    pad(t_nin, t_an2, [('Dm', 1), ('Bb', 1)], 60, -24, 0.4, ('vc', 'vla', 'vln'), 1.0, 1.5)
    melody(t_an2, KILER + KILER, 84, 'harp', -15, 0.6, 0.2, 0.55, 1.8, t1=t_kap3)
    melody(t_an2, KILER + KILER, 84, 'glock', -25, 0.45, -0.2, 0.65, t1=t_kap3)
    O('marktree', t_kap3, -16, 0, 0.6, i=4)
    pad(t_kap3, t_bas, [('D', 1), ('G', 1)], 60, -22, 0.45, ('vc', 'vla', 'vln'), 1.0, 1.5, top=78)
    # descending: each floor a step of rising tension/wonder
    dsh = shots('descend')[0]
    for k, kt in enumerate(dsh['p']['_k']):
        O('timp_hit', dsh['t0'] + kt, -12 + k * 2, 0, 0.5, -5 + k * 2)
    pad(t_bas, t_son3, [('Dm', 1), ('Bb', 1), ('C', 1)], 70, -21, 0.5, ('cb', 'vc', 'vla', 'vln'), 0.8, 1.0, sw=(-8, 0))
    ending_at('cresc', t_son3, -14, 0.6, i=10, length=3.0)
    # the hall & reunion (hope theme)
    HIT(t_son3, 2, 'D', -8)
    pad(t_son3, t_vk, [('D', 1), ('G', 1), ('A', 1), ('D', 1)], 72, -18, 0.6, ('cb', 'vc', 'vla', 'vln'), 0.4, 1.5, top=78)
    arp(t_son3, t_vk, [('D', 1), ('G', 1), ('A', 1), ('D', 1)], 72, 'harp', -17, 0.55, (0, 1, 2, 3), 2, 57, 0.5, 0.0)
    pad(t_vk, t_d30 + 1, HOPE_PROG, 72, -16, 0.8, ('cb', 'vc', 'vla', 'vln'), 0.5, 2.5, top=79)
    melody(t_vk + 0.2, HOPE, 72 * 1.6, 'horn', -11, 0.8, 0.15, 0.55, t1=t_d30 + 1)
    melody(t_vk + 0.2, HOPE, 72 * 1.6, 'vln', -15, 0.75, -0.1, 0.5, transpose=12, t1=t_d30 + 1)
    O('timp_roll', t_vk, -17, 0, 0.4, cut=3.5, fadein=3.0)
    HIT(t_vk + 3.4, 2, 'D', -9)
    # ===== DAY 30: underground life
    day_card(t_d30, 'D', soft=True)
    t_hay2 = T('burada hayat yüzyıllar'); t_ninb = T('ninen bir akşam'); t_dus = T('düşman değişir'); t_rgh = T('radyodan gelen haberlerle'); t_ibo = T('istanbul büyük ölçüde'); t_ada = T('ama adalarda')
    t_ank = T('ankara ordunun'); t_kar2 = T('karadenizde insanlar'); t_rb = T('radyoya bağlanan'); t_dog = T('doğu nun dağ köyleri'); t_tpp = T('türkiye parça parça ama'); t_vo = T('ve o akşam yukarıdan')
    pad(t_d30 + 1, t_ninb, [('D', 1), ('Bm', 1), ('G', 1), ('A', 1)], 68, -21, 0.5, ('cb', 'vc', 'vla', 'vln'), 1.0, 1.5)
    arp(t_d30 + 1, t_ninb, [('D', 1), ('Bm', 1), ('G', 1), ('A', 1)], 68, 'harp', -18, 0.5, (0, 1, 2, 3, 2, 1), 3, 50, 0.5, 0.0)
    melody(t_hay2, HICAZ, 80, 'flute', -16, 0.5, 0.2, 0.55, t1=t_ninb)
    melody(t_ninb + 0.3, KEMAL[12:], 70, 'piano', -15, 0.45, 0, 0.3, 1.6, t1=t_dus, rel=1.0)
    pad(t_ninb, t_dus, [('F', 1), ('C', 1), ('Gm', 1), ('Dm', 1)], 70, -23, 0.4, ('vc', 'vla', 'vln'), 1.0, 1.5)
    HIT(t_dus, 2, 'D', -9)
    pad(t_dus, t_rgh, [('D', 1), ('G', 1)], 60, -17, 0.65, ('cb', 'vc', 'vla', 'vln'), 0.3, 2.0, top=78)
    melody(t_dus + 0.2, [('D4', 1), ('A4', 1), ('B4', 1), ('A4', 1), ('D5', 3)], 60, 'horn', -12, 0.75, 0.15, 0.55, t1=t_rgh + 1)
    pad(t_rgh, t_ibo, [('Dm', 1)], 60, -25, 0.4, ('cb', 'vc', 'vla'), 1.0, 1.0)
    pad(t_ibo, t_ada, [('Dm', 1), ('Bb', 1)], 60, -21, 0.5, ('cb', 'vc', 'vla', 'vln'), 0.5, 1.5)
    pad(t_ada, t_kar2, [('F', 1), ('C', 1), ('D', 1)], 70, -22, 0.45, ('cb', 'vc', 'vla', 'vln'), 0.8, 1.5)
    # Karadeniz: horon-flavoured playful spiccato (7/8)
    hb = 152; e = 60 / hb / 2; t = t_kar2
    horon = [74, 76, 77, 79, 81, 79, 77, 76, 77, 79, 77, 76, 74, 73]
    while t < t_dog - 0.05:
        for k, n in enumerate(horon):
            if t + k * e >= t_dog: break
            N('vln_spic', n, t + k * e, e * 0.9, 0.7 if k % 7 in (0, 2, 4) else 0.5, -17, 0.1, 0.35)
            if k % 7 in (0, 4): O('darbuka', t + k * e, -16, 0, 0.3, -4)
            else: O('darbuka', t + k * e, -22, 0.1, 0.3, 0)
        ostinato(t, min(t + 14 * e, t_dog), hb, [('Dm', 1)], 'vc_pizz', -16, 0.6, (0, None, None, None, 7, None, None), 38)
        t += 14 * e
    # Türkiye pieces alive: hope
    pad(t_dog, t_vo, [('D', 1), ('A', 1), ('Bm', 1), ('G', 1)], 72, -19, 0.6, ('cb', 'vc', 'vla', 'vln'), 0.6, 2.0, top=78)
    melody(t_tpp, [('F#4', 1), ('A4', 1), ('D5', 2)], 72, 'horn', -13, 0.7, 0.15, 0.55, t1=t_vo + 0.5)
    # ===== EZAN: respectful near-silence, then dawn
    t_ezn = T('akşam ezanı okunuyor'); t_ist3 = T('istanbulda o ses kesildiğinde'); t_sim3 = T('şimdi bir şeyin'); t_ceb = T('cebinden kemal')
    drone(t_vo, t_ist3, 'D2', 'cb', -26, 0.3, 3.0, 3.0)
    N('vln', 81, T('köyün imamı'), t_ist3 - T('köyün imamı'), 0.2, -33, 0.2, 0.75, 4.0, 3.0)
    pad(t_ist3, t_sim3, [('Dm', 1)], 60, -27, 0.3, ('vc', 'vla'), 1.5, 1.0)
    pad(t_sim3, t_ceb, [('D', 1), ('G', 1)], 60, -19, 0.6, ('cb', 'vc', 'vla', 'vln'), 1.0, 2.5, top=79)
    O('marktree', t_sim3, -18, 0, 0.6, i=2)
    # ===== ENDING
    t_vd = T('ve düşünüyorsun'); t_osk = T('o sabah kapıyı açsaydın'); t_bde = T('belki de şimdi'); t_pek = T('peki sen o sabah'); t_bu = T('bu hikaye seni'); t_cun = T('çünkü bir sonraki')
    melody(t_ceb + 0.3, KEMAL[:12], 70, 'piano', -13, 0.5, 0, 0.3, 1.6, t1=t_osk, rel=1.2)
    pad(t_ceb, t_osk, [('Dm', 1), ('Gm', 1), ('C', .5), ('A7', .5), ('Dm', 1)], 70, -22, 0.45, ('cb', 'vc', 'vla', 'vln'), 1.2, 2.0)
    N('piano', 38, t_osk, 3, 0.5, -16, 0, 0.4); N('piano', 39, t_osk + 0.02, 3, 0.4, -18, 0, 0.4); N('piano', 44, t_osk + 0.04, 3, 0.4, -18, 0, 0.4)
    pad(t_osk, t_bde, [('Bb', 1)], 60, -23, 0.4, ('vc', 'vla', 'vln'), 0.8, 1.0, trem=True)
    tension_cluster(t_bde, t_pek, -17, 62); soft_hit(t_bde + 0.2, -12, -7)
    # CTA: light main-theme pulse
    bpm = 96
    ostinato(t_pek, t_cun, bpm, MAIN_PROG, 'vc_spic', -17, 0.6, (0, 0, 7, 0, 12, 0, 7, 0), 38)
    drums(t_pek, t_cun, bpm, [(0, 'framedrum', -12, -2), (1.5, 'framedrum', -15, 0), (2, 'framedrum', -13, -2), (3, 'framedrum', -15, 0)], 4, -6)
    pad(t_pek, t_cun, MAIN_PROG, bpm, -23, 0.5, ('cb', 'vla', 'vln'), 0.5, 1.0)
    melody(t_bu, MAIN[:11], bpm, 'horn', -15, 0.6, 0.15, 0.5, t1=t_cun)
    # finale: darkness
    tension_cluster(t_cun, t_cun + 4.0, -15, 62)
    HIT(t_cun + 4.0, 3, 'Dm', -6)
    drone(t_cun + 4.0, TOTAL - 0.5, 'D1', 'cb', -20, 0.4, 0.5, 2.5)

def day_card(t, chord, soft=False):
    ending_at('cresc', t, -15 if not soft else -18, 0.6, i=3, length=2.5)
    if soft: HIT(t, 2, chord, -10, brass=True)
    else: HIT(t, 3, chord, -6)

# ------------------------------------------------------------------ render
def master(dry, wet, roomw):
    import pedalboard as pb
    ir = make_ir(3.0, 0.03, 5500, 14, 1.0, seed=11)
    ir_room = make_ir(1.1, 0.01, 7000, 10, 0.8, seed=5)
    print('reverb...', flush=True)
    rv = convolve(hp(wet, 140), ir) * 0.55
    rr = convolve(roomw, ir_room) * 0.35
    mix = dry + rv + rr
    mix = hp(mix, 28, 2)
    mix = peak(mix, 250, -1.5, 0.8)       # de-mud
    mix = peak(mix, 2800, -2.5, 0.9)      # leave room for the voice-over presence band
    board = pb.Pedalboard([pb.Compressor(threshold_db=-18, ratio=2.0, attack_ms=30, release_ms=250), pb.Limiter(threshold_db=-1.5, release_ms=200)])
    return board(mix, SR) * db(-1.2)

if __name__ == '__main__':
    t0 = time.time()
    compose()
    print('composed in %.0fs' % (time.time() - t0), flush=True)
    out = master(M.x, R.x, RS.x)
    np.save('music_raw.npy', out.astype(np.float32))
    print('done %.0fs peak %.2f' % (time.time() - t0, np.abs(out).max()))
