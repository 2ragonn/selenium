"""Arka plan müziği: 36 bölümlük sinematik partisyon (D minör / Hicaz renkli), sahnelere senkron.
Çıktı: out/music.mid -> fluidsynth (MuseScore General) -> out/music_dry.wav"""
import os, subprocess
import numpy as np
from mscore import Score, P
from tl import cue, END

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
S = Score(seed=7)


def C(spec, fb=None):
    try:
        return cue(spec)
    except Exception:
        if fb is None:
            raise
        return cue(fb)


# ------------------------------------------------------------ armoni yardımcıları
Q = {'': (0, 4, 7), 'm': (0, 3, 7), '7': (0, 4, 7, 10), 'm7': (0, 3, 7, 10), 'maj7': (0, 4, 7, 11), 'sus4': (0, 5, 7), 'sus2': (0, 2, 7),
     'dim': (0, 3, 6), 'add9': (0, 4, 7, 14), 'madd9': (0, 3, 7, 14), 'm(b9)': (0, 1, 3, 7), '5': (0, 7)}


def chord_pcs(name):
    r = name[0]; i = 1
    if len(name) > 1 and name[1] in '#b':
        r += name[1]; i = 2
    base = P(r + '0') - 12
    qs = name[i:].split('/')[0]
    bass = name.split('/')[1] if '/' in name else None
    pcs = [(base + q) % 12 for q in Q[qs]]
    return pcs, (P(bass + '0') % 12 if bass else pcs[0])


def tones_in(pcs, lo, hi):
    return [p for p in range(P(lo), P(hi) + 1) if p % 12 in pcs]


def lead(prev, pcs, n, lo, hi):
    cand = tones_in(pcs, lo, hi)
    if not prev:
        mid = (P(lo) + P(hi)) / 2
        out = sorted(cand, key=lambda p: abs(p - mid))[:n]
        return sorted(out)
    out = []
    for v in prev:
        best = min(cand, key=lambda p: abs(p - v) + (6 if p in out else 0))
        out.append(best)
    # tüm akor tonlarını kapsamaya çalış
    miss = [pc for pc in pcs[:3] if pc not in [o % 12 for o in out]]
    if miss and len(out) >= 3:
        dup = [i for i in range(len(out)) if [o % 12 for o in out].count(out[i] % 12) > 1]
        if dup:
            i = dup[-1]; out[i] = min(tones_in([miss[0]], lo, hi), key=lambda p: abs(p - out[i]))
    return sorted(out)


def seq_prog(prog, t0, t1, bpm):
    """[(akor, vuruş)] döngüsünü [t0,t1) aralığına yayar -> [(t, süre, akor)]."""
    b = 60 / bpm; t = t0; out = []
    while t < t1 - 0.05:
        for ch, beats in prog:
            d = beats * b
            if t >= t1 - 0.05:
                break
            out.append((t, min(d, t1 - t), ch)); t += d
    return out


def strings_pad(t0, t1, prog, bpm, vel=70, insts=('vln', 'vla', 'vc', 'cb'), top=('A3', 'D6'), dyn=None, legato=1.04, octaves=False):
    for nm in insts:
        S.use(nm, t0)
    prev = None
    for t, d, ch in seq_prog(prog, t0, t1, bpm):
        pcs, bass = chord_pcs(ch)
        upper = lead(prev, pcs, 4, *top); prev = upper
        dd = d * legato
        if 'cb' in insts:
            b = [p for p in range(P('D1'), P('D2') + 1) if p % 12 == bass][0]
            S.note('cb', t, dd, b, vel)
        if 'vc' in insts:
            b = [p for p in range(P('C2'), P('B2') + 1) if p % 12 == bass][0]
            S.note('vc', t, dd, b, vel); S.note('vc', t, dd, b + 7 if (b + 7) % 12 in pcs else b + 12, vel * 0.8)
        if 'vla' in insts:
            for p in upper[:2]:
                S.note('vla', t, dd, p, vel * 0.9)
        if 'vln' in insts:
            for p in upper[2:]:
                S.note('vln', t, dd, p + (12 if octaves else 0), vel)
    if dyn:
        for nm in insts:
            S.dyn(nm, dyn)


def brass_pad(t0, t1, prog, bpm, vel=70, inst='hn', lo='F3', hi='F4', dyn=None):
    S.use(inst, t0); prev = None
    for t, d, ch in seq_prog(prog, t0, t1, bpm):
        pcs, _ = chord_pcs(ch)
        v = lead(prev, pcs, 3, lo, hi); prev = v
        for p in v:
            S.note(inst, t, d * 1.03, p, vel)
    if dyn:
        S.dyn(inst, dyn)


def arp(inst, t0, t1, prog, bpm, step=0.5, order=(0, 1, 2, 3, 2, 1), lo='D4', hi='D6', vel=60, oct_add=0):
    S.use(inst, t0)
    for t, d, ch in seq_prog(prog, t0, t1, bpm):
        pcs, _ = chord_pcs(ch)
        tones = sorted(tones_in(pcs, lo, hi))[:6]
        if not tones:
            continue
        n = int(round(d / (step * 60 / bpm)))
        for i in range(n):
            p = tones[order[i % len(order)] % len(tones)] + oct_add
            S.note(inst, t + i * step * 60 / bpm, step * 60 / bpm * 1.6, p, vel * (1.1 if i % 4 == 0 else 0.92))


def bassline(inst, t0, t1, prog, bpm, pattern=((0, 0, 1.0), (2, 7, 0.8)), octave='D2', vel=75, dur=0.9):
    """pattern: (vuruş, kök+aralık, vurgu)"""
    S.use(inst, t0)
    for t, d, ch in seq_prog(prog, t0, t1, bpm):
        pcs, bass = chord_pcs(ch)
        root = [p for p in range(P(octave) - 6, P(octave) + 6) if p % 12 == bass][0]
        beats = d / (60 / bpm)
        for bt, iv, acc in pattern:
            if bt < beats - 0.01:
                S.note(inst, t + bt * 60 / bpm, dur * 60 / bpm, root + iv, vel * acc)


def ostinato(inst, t0, t1, bpm, cells, vel=70, step=0.5, accent=(1.15, 0.85, 0.95, 0.85), tr=0, dur=0.85):
    S.use(inst, t0); b = 60 / bpm * step; i = 0; t = t0
    while t < t1 - 0.02:
        p = cells[i % len(cells)]
        if p is not None:
            S.note(inst, t, b * dur, P(p) + tr, vel * accent[i % len(accent)])
        t += b; i += 1


def drums(t0, t1, bpm, pat, bar=4, vel=90):
    """pat: {'enstrüman:perde': [vuruşlar]} — 'taiko:D2', 'kit:sn', 'timp:D2', 'bd:D2'."""
    b = 60 / bpm; t = t0
    for k in pat:
        S.use(k.split(':')[0], t0)
    while t < t1 - 0.05:
        for k, beats in pat.items():
            nm, p = k.split(':')
            for bt in beats:
                tt_ = t + bt * b
                if tt_ < t1:
                    v = vel * (1.12 if abs(bt - round(bt)) < 1e-6 and int(round(bt)) % bar == 0 else 0.9)
                    S.note(nm, tt_, 0.4, p, v)
        t += bar * b


def mel(inst, t0, notes, bpm, vel=80, tr=0, legato=1.03, dyn=None, t_end=None):
    S.use(inst, t0)
    e = S.line(inst, t0, notes, bpm, vel, legato, tr, t_end=t_end)
    if dyn:
        S.dyn(inst, dyn)
    return e


def fade_all(t, names, dur=1.5):
    for nm in names:
        S.dyn(nm, [(t, 90), (t + dur, 0)])


def silence_all(t):
    for nm in ('vln', 'vla', 'vc', 'cb', 'hn', 'tbn', 'choir', 'glass', 'cl', 'tpt'):
        S.dyn(nm, [(t, 0), (t + 0.05, 0)])


# ------------------------------------------------------------ temalar
DEDE = [('A4', 1.5), ('D5', .5), ('C5', 1), ('A4', 1), ('Bb4', 1), ('A4', .5), ('G4', .5), ('F4', 2), ('G4', 1), ('A4', .5), ('Bb4', .5), ('D5', 1), ('C5', 1),
        ('A4', 3), (None, 1), ('D5', 1.5), ('E5', .5), ('F5', 1), ('E5', 1), ('D5', 1), ('C5', .5), ('Bb4', .5), ('A4', 2), ('G4', 1), ('Bb4', 1), ('A4', 1), ('G4', .5), ('E4', .5), ('D4', 4)]
DEDE_P = [('Dm', 4), ('Gm', 2), ('F', 2), ('Gm', 2), ('C', 2), ('F', 2), ('A7', 2), ('Dm', 2), ('C', 2), ('Bb', 2), ('F/A', 2), ('Gm', 2), ('A7', 2), ('Dm', 4)]
UMUT = [('D4', 1), ('F4', 1), ('A4', 1), ('D5', 1), ('C5', 1.5), ('Bb4', .5), ('A4', 2), ('Bb4', 1), ('A4', 1), ('G4', 1), ('F4', 1), ('E4', 2), ('A4', 2),
        ('D5', 1), ('C5', 1), ('Bb4', 1), ('A4', 1), ('G4', 1.5), ('A4', .5), ('Bb4', 2), ('A4', 1), ('G4', 1), ('F4', 1), ('E4', 1), ('D4', 4)]
UMUT_P = [('Dm', 4), ('F', 4), ('Gm', 4), ('A', 4), ('Bb', 4), ('Eb', 4), ('A7', 4), ('Dm', 4)]
UMUT_PD = UMUT_P[:-1] + [('D', 4)]
GEMI = [('D2', 2), ('Eb2', 1), ('D2', 1), ('A1', 4)]
MAH = [('A4', .5), ('Bb4', .5), ('A4', .5), ('G4', .5), ('F#4', .5), ('G4', .5), ('A4', 1), ('G4', .5), ('F#4', .5), ('Eb4', .5), ('F#4', .5), ('D4', 1), (None, 1),
       ('D4', .5), ('Eb4', .5), ('F#4', .5), ('G4', .5), ('A4', .5), ('Bb4', .5), ('C5', .5), ('Bb4', .5), ('A4', .5), ('G4', .5), ('F#4', .5), ('Eb4', .5), ('D4', 2)]
MAH_P = [('D', 4), ('Cm', 2), ('D', 2), ('Gm', 2), ('Eb', 2), ('D', 4)]
NEY = [('E5', 2), ('D5', 1), ('C5', 1), ('Bb4', 2), ('A4', 3), (None, 1), ('C5', 1.5), ('D5', .5), ('E5', 1), ('F5', 1), ('E5', 2), ('D5', 1), ('C5', 1), ('Bb4', 1.5), ('C5', .5), ('A4', 4)]
TENS = ['D3', 'D3', 'Eb3', 'D3', 'D3', 'D3', 'C3', 'D3']
TENS2 = ['D3', 'A3', 'D3', 'Eb3', 'D3', 'A3', 'C4', 'Bb3']

# ======================================================================= CUE'LAR
# M01 Soğuk açılış (0 – 50.4)
t = C('1:uzaylılar') - 0.03
S.hit(t, 1.0)
S.use('cb', 0); S.note('cb', t + 0.1, 50.2 - t, 'D1', 80); S.dyn('cb', [(t, 95), (t + 4, 45), (44, 45), (50.2, 0)])
S.use('vc', 0); S.note('vc', t + 0.2, 50.0 - t, 'D2', 70); S.dyn('vc', [(t, 80), (t + 5, 38), (44, 40), (50.2, 0)])
S.use('vln', 3.0); S.chord('vln', 3.2, 47.0, ['A5', 'D6'], 60); S.dyn('vln', [(3.2, 5), (8, 45), (13.4, 60), (19, 48), (28, 32), (37.9, 28), (46, 40), (50.2, 0)])
S.use('pno', 3.0)
for tt_, p in [(3.3, 'D5'), (4.15, 'A4'), (5.0, 'E5'), (5.85, 'F5'), (7.4, 'A4'), (8.3, 'C5'), (9.2, 'D5')]:
    S.note('pno', tt_, 2.5, p, 58)
S.use('cel', 5.0)
for i, p in enumerate(['A5', 'D6', 'A5', 'F5']):
    S.note('cel', C('2:saat') + 0.1 + i * 0.32, 1.0, p, 52)
S.use('oohs', 9.0); S.chord('oohs', 9.8, 4.0, ['D4', 'F4', 'A4', 'C5'], 70); S.dyn('oohs', [(9.8, 10), (12.5, 70), (13.6, 30)])
S.use('vla', 13.0); S.chord('vla', 13.4, 6.0, ['Bb3', 'D4', 'F4'], 66); S.dyn('vla', [(13.4, 30), (15, 70), (18.5, 50), (19.4, 0)])
arp('cel', 13.4, 19.0, [('Bbadd9', 4)], 72, step=0.5, order=(0, 2, 1, 3, 2, 4), lo='D5', hi='D6', vel=48)
mel('hn', 19.6, GEMI, 50, vel=72, tr=12, dyn=[(19.6, 20), (22, 60), (26, 55), (29.2, 0)])
mel('hn', 29.4, GEMI[:2], 48, vel=60, tr=12, dyn=[(29.4, 10), (30.5, 50), (33.5, 0)])
S.use('timp', 30)
for sp in ('11', '11:motor', '11:alev'):
    S.note('timp', C(sp) + 0.05, 1.2, 'D2', 70)
S.use('pizz', 37.0)
ostinato('vcP', 37.9, 50.2, 104, ['D3', 'D3', 'A2', 'D3', 'D3', 'D3', 'Eb3', 'D3'], vel=64)
S.dyn('vcP', [(37.9, 60), (49.5, 105), (50.2, 0)])
mel('glass', 37.9, [('A4', 8), ('Bb4', 8)], 80, vel=50, dyn=[(37.9, 0), (44, 45), (50.2, 0)])
# → 50.4–60: SESSİZLİK (geminin uğultusu SFX'te)

# M02 Uğultu & kararma (60.5 – 89.0)
t0, t1 = C('15:sonra') - 0.2, C('20:susuyor') + 0.05
S.use('cb', t0); S.note('cb', t0, t1 - t0, 'D1', 85); S.dyn('cb', [(t0, 20), (t0 + 6, 70), (t1 - 0.5, 90), (t1, 0)])
ostinato('vcF', t0, t1, 92, TENS, vel=72, step=0.5)
S.dyn('vcF', [(t0, 20), (t0 + 8, 55), (78, 75), (86, 105), (t1 - 0.05, 110), (t1, 0)])
for sp in ('15:odanın', '16', '16:karşıyakanın', '16:köprülerin', '17:aynı'):
    S.note('timp', C(sp) + 0.02, 1.4, 'D2', 82)
S.use('vlnT', 68.0); S.chord('vlnT', C('17:telefonun'), t1 - C('17:telefonun'), ['D5', 'Eb5', 'A5'], 70)
S.dyn('vlnT', [(C('17:telefonun'), 10), (79, 40), (84, 75), (t1 - 0.05, 115), (t1, 0)])
S.use('hn', 74.0); mel('hn', 74.3, GEMI, 46, vel=78, tr=12, dyn=[(74.3, 30), (78, 70), (84, 0)])
S.use('choir', 80.0); S.chord('choir', C('19:köpekler'), t1 - C('19:köpekler'), ['D4', 'Eb4', 'A4', 'Bb4'], 72)
S.dyn('choir', [(C('19:köpekler'), 0), (t1 - 1.0, 95), (t1, 0)])

# M03 Soru / tez (92.5 – 118.4)
t0 = C('20:peki') - 0.1; t1 = C('25') - 0.2
strings_pad(t0, t1, [('Dm', 4), ('Bb', 4), ('Gm', 4), ('A', 4)], 66, vel=72, dyn=[(t0, 10), (t0 + 4, 50), (104, 60), (110, 70), (t1 - 1.0, 55), (t1, 0)])
mel('hn', t0 + 3.6, UMUT[:8], 66, vel=70, dyn=[(t0 + 3.6, 40), (104, 65), (106, 40)])
arp('harp', 99.6, 105.6, [('Dm', 4), ('Bb', 4)], 66, step=0.5, lo='D4', hi='D5', vel=50)
S.note('timp', C('23:ne') + 0.02, 1.0, 'D2', 70)
S.note('timp', C('24:ne') + 0.02, 1.0, 'A2', 70)
mel('pno', C('24:dedenin') - 0.05, DEDE[:8], 70, vel=64)
S.cc('pno', C('24:dedenin') - 0.1, 64, 127); S.cc('pno', t1 - 0.2, 64, 0)

# M04 Birinci gün — mahalle (118.6 – 166.5)
t0 = C('25'); t1 = C('33:gündüz') - 0.3
S.hit(t0 + 0.02, 0.7, low_brass=False)
tm = t0 + 2.2
bassline('pizz', tm, t1, MAH_P, 96, pattern=((0, 0, 1.0), (1.5, 7, 0.7), (2, 0, 0.85), (3, 7, 0.7)), octave='D3', vel=70, dur=0.6)
S.dyn('pizz', [(tm, 90), (t1 - 2, 90), (t1, 0)])
arp('koto', tm, t1, MAH_P, 96, step=0.5, order=(0, 2, 1, 2), lo='D4', hi='A5', vel=56)
k = tm + 4 * 60 / 96 * 2
while k < 160:
    k = mel('cl', k, MAH, 96, vel=72, legato=0.9) + 60 / 96 * 4
S.dyn('cl', [(tm, 80), (140, 80), (147, 60), (150, 85), (160, 70), (t1, 0)])
S.use('wblock', tm)
for i in range(int((t1 - tm) / (60 / 96 * 2))):
    S.note('wblock', tm + i * 60 / 96 * 2 + 60 / 96, 0.1, 'G4' if i % 2 else 'C5', 40)
strings_pad(tm, t1, MAH_P, 96, vel=55, insts=('vla', 'vc'), top=('D3', 'A4'), dyn=[(tm, 20), (125, 40), (t1 - 1, 40), (t1, 0)])

# M05 Gölgesiz (166.6 – 205.6)
t0 = C('33:gündüz') - 0.2; t1 = C('39:sanki') + 0.3
S.use('vln', t0); S.chord('vln', t0, t1 - t0, ['A5', 'Bb5', 'D6'], 60)
S.dyn('vln', [(t0, 0), (t0 + 4, 35), (178, 50), (182.0, 85), (183, 40), (196, 35), (t1 - 0.5, 50), (t1, 0)])
S.use('glass', t0); S.chord('glass', t0, 15, ['D4', 'A4'], 50); S.dyn('glass', [(t0, 0), (t0 + 5, 45), (t0 + 14, 0)])
ostinato('vcP', 172.5, 182.2, 72, ['D3', None, 'D3', None, 'Eb3', None, 'D3', None], vel=70)
S.dyn('vcP', [(172.5, 80), (182.2, 80)])
S.roll('timp', 179.5, C('35:geminin') + 0.05, 'D2', 20, 95)
S.hit(C('35:geminin') + 0.08, 0.85, low_brass=True)
mel('hn', 184.0, GEMI, 44, vel=70, tr=12, dyn=[(184, 20), (188, 60), (194, 0)])
ostinato('vcP', 190.0, 205.0, 84, ['D3', None, None, None, 'D3', None, 'A2', None], vel=66)
for sp in ('38', '38:bilgisayarlar', '38:trafik', '38:metro', '39:asansörler', '39:hepsi'):
    S.note('timp', C(sp, '38') + 0.02, 0.9, 'D2', 66)
S.note('pno', C('39:fişini') + 0.15, 4, 'D1', 90)

# M06 Rıza Dede (206.0 – 231.5)
t0 = C('39:deden') - 0.6; t1 = C('43:bu') - 0.4
strings_pad(t0, t1, DEDE_P, 66, vel=62, top=('F3', 'A5'), dyn=[(t0, 0), (t0 + 3, 50), (220, 60), (227, 55), (t1 - 1, 30), (t1, 0)])
arp('harp', t0 + 0.0, t1, DEDE_P, 66, step=0.5, order=(0, 1, 2, 1), lo='D3', hi='D5', vel=50)
mel('cl', t0 + 60 / 66 * 0, DEDE, 66, vel=74, legato=1.0, dyn=[(t0, 60), (220, 75), (t1 - 1, 50), (t1, 0)], t_end=t1)

# M07 Lamba & bilim (232 – 268)
t0 = C('43:bu') - 0.2; t1 = C('49:ve') - 0.3
prog = [('Dm', 4), ('Bb', 4), ('C', 4), ('A', 4)]
bassline('pizz', t0, t1, prog, 100, pattern=((0, 0, 1), (1, 12, .6), (2, 7, .8), (3, 12, .6)), octave='D3', vel=66, dur=0.5)
S.dyn('pizz', [(t0, 80), (t1, 80)])
arp('cel', t0, t1, prog, 100, step=0.5, order=(0, 1, 2, 3), lo='D5', hi='D6', vel=40)
strings_pad(t0, t1, prog, 100, vel=55, insts=('vln', 'vla', 'vc'), top=('A3', 'A5'), dyn=[(t0, 20), (t0 + 4, 45), (t1 - 1, 45), (t1, 0)])
drums(234.2, 242.5, 100, {'kit:sn': [0, 0.5, 0.75, 2, 2.5, 2.75], 'kit:bd': [0, 2]}, vel=55)
S.hit(C('45:ama') + 0.02, 0.6, low_brass=False, cym=True)

# M08 Tam bizden (268 – 300.5)
t0 = C('49:ve') - 0.2; t1 = C('54:akşam') - 0.2
bassline('pizz', t0, t1, MAH_P, 108, pattern=((0, 0, 1.0), (1.5, 7, 0.7), (2, 0, 0.85), (3, 7, 0.7)), octave='D3', vel=70, dur=0.5)
arp('koto', t0, t1, MAH_P, 108, step=0.5, order=(0, 2, 1, 2, 3, 2, 1, 2), lo='D4', hi='A5', vel=58)
k = t0 + 60 / 108 * 4
while k < t1 - 6:
    k = mel('cl', k, MAH, 108, vel=74, legato=0.85, tr=12 if int(k) % 2 else 0) + 60 / 108 * 2
S.dyn('cl', [(t0, 75), (t1 - 2, 75), (t1, 0)])
S.use('wblock', t0)
for i in range(int((t1 - t0) / (60 / 108))):
    S.note('wblock', t0 + i * 60 / 108, 0.08, 'C5' if i % 2 == 0 else 'G4', 36 if i % 2 else 46)
S.use('glock', 290.0)
for i, p in enumerate(['A5', 'D6', 'F#6', 'A6']):
    S.note('glock', C('53:gram') + i * 0.12, 1.2, p, 60)

# M09 Ne istiyor (300.5 – 307.6)
t0 = C('54:akşam') - 0.1; t1 = C('56') - 0.1
S.use('vlnT', t0); S.chord('vlnT', t0, t1 - t0, ['D5', 'Eb5', 'Ab5'], 70); S.dyn('vlnT', [(t0, 10), (t1 - 0.4, 100), (t1, 0)])
mel('tbn', t0 + 1.0, [('D3', 2), ('Eb3', 1), ('D3', 1), ('A2', 3)], 50, vel=80, dyn=[(t0 + 1, 30), (t1 - .3, 90), (t1, 0)])

# M10 İkinci gün — devlet (307.7 – 327.3)
t0 = C('56'); t1 = C('59:akşam') - 0.2
S.hit(t0 + 0.02, 0.8)
drums(t0 + 0.9, t1, 88, {'kit:sn': [0, 0.25, 0.5, 2, 2.25, 2.5, 3.5], 'timp:D2': [0, 2], 'timp:A2': [3]}, vel=62)
ostinato('vcF', t0 + 0.9, t1, 88, TENS2, vel=66)
S.dyn('vcF', [(t0, 50), (t1 - 1, 80), (t1, 0)])
brass_pad(t0 + 0.9, t1, [('Dm', 4), ('Bb', 4), ('Gm', 4), ('A', 4)], 88, vel=70, inst='hn', dyn=[(t0, 30), (315, 70), (t1 - 1, 55), (t1, 0)])
for c_ in ('56:olağanüstü#2',):
    S.hit(C(c_) + 0.02, 0.6, low_brass=False)

# M11 Tartışma programı (327.3 – 356.4)
t0 = C('59:akşam') - 0.2; t1 = C('64:ama') - 0.2
prog = [('Dm', 2), ('A7', 2), ('Dm', 2), ('Gm', 1), ('A7', 1)]
bassline('pizz', t0, t1, prog, 112, pattern=((0, 0, 1), (1, 7, .7)), octave='D3', vel=66, dur=0.4)
S.use('bsn', t0)
quirky = [('D3', .5), (None, .5), ('F3', .5), ('A3', .5), ('G#3', .5), ('A3', .5), (None, 1), ('E3', .5), ('G3', .5), ('Bb3', .5), ('A3', .5), ('F3', .5), ('D3', .5), (None, 1)]
k = t0 + 2
while k < t1 - 3:
    k = S.line('bsn', k, quirky, 112, 70, 0.6) + 60 / 112 * 2
S.dyn('bsn', [(t0, 80), (t1, 80)])
S.use('glock', 351)
S.note('glock', C('63:rüyasında') + 0.1, 2.0, 'F6', 55); S.note('glock', C('63:rüyasında') + 0.45, 2.0, 'Bb6', 50)
S.note('timp', C('64') + 0.4, 0.8, 'A2', 60); S.note('timp', C('64') + 0.75, 1.5, 'D2', 75)

# M12 İHA'lar ve jetler (356.5 – 387.4)
t0 = C('64:ama') - 0.1; t1 = C('69') - 0.2
ostinato('vcF', t0, t1, 120, TENS, vel=72)
S.dyn('vcF', [(t0, 40), (368, 85), (371, 100), (375, 70), (381, 95), (385, 50), (t1, 0)])
ostinato('vlaF', t0 + 4, t1, 120, ['A3', 'A3', 'Bb3', 'A3', 'A3', 'A3', 'G3', 'A3'], vel=62)
S.dyn('vlaF', [(t0 + 4, 30), (370, 80), (385, 40), (t1, 0)])
drums(362.7, 384.5, 120, {'taiko:D2': [0, 1.5, 2, 3], 'timp:D2': [0], 'kit:sn': [3.5, 3.75]}, vel=72)
S.hit(C('66:taş') + 0.02, 0.8)
brass_pad(375.3, 384.7, [('Dm', 4), ('Eb', 4), ('Dm', 4)], 120, vel=82, inst='tbn', lo='D2', hi='D3', dyn=[(375.3, 40), (379, 95), (382.5, 100), (384.7, 0)])
S.use('vlnT', 377); S.chord('vlnT', 377.3, 7.0, ['D6', 'Eb6'], 70); S.dyn('vlnT', [(377.3, 0), (381.3, 90), (384.3, 0)])
strings_pad(384.7, t1, [('Dm', 4)], 60, vel=55, insts=('vla', 'vc', 'cb'), top=('D3', 'A4'), dyn=[(384.7, 50), (t1, 0)])

# M13 Göç, çaycı, bakkal (387.6 – 436.6)
t0 = C('69'); t1 = C('76:ve') - 0.3
prog = [('Dm', 4), ('Gm', 4), ('Bb', 4), ('A', 4)]
strings_pad(t0, 400.0, prog, 72, vel=60, dyn=[(t0, 20), (t0 + 3, 55), (399, 55), (400.5, 0)])
drums(t0, 400.0, 72, {'kit:sn': [0, 1.75, 2]}, vel=40)
mel('hn', t0 + 0.5, [(p, b * 1) for p, b in UMUT[:12]], 72, vel=66, dyn=[(t0, 50), (399, 55), (400.5, 0)])
bassline('pizz', 400.2, 411.0, MAH_P, 112, pattern=((0, 0, 1), (1, 7, .7), (2, 0, .85), (3, 7, .7)), octave='D3', vel=66, dur=0.4)
arp('koto', 400.2, 411.0, MAH_P, 112, step=0.5, order=(0, 2, 1, 2), lo='D4', hi='A5', vel=58)
mel('cl', 401.2, MAH[:13], 112, vel=72, legato=0.85)
strings_pad(411.5, t1, DEDE_P, 70, vel=56, insts=('vla', 'vc', 'cb'), top=('F3', 'F4'), dyn=[(411.5, 10), (414, 50), (430, 52), (t1 - 1, 40), (t1, 0)])
mel('pno', 412.0, DEDE, 70, vel=58, t_end=t1)
S.cc('pno', 411.9, 64, 127); S.cc('pno', t1, 64, 0)
mel('cl', 424.0, DEDE[:7], 70, vel=60, legato=0.95)

# M14 İlk ders: Duyan var mı (436.8 – 492.5)
t0 = C('76:ve') - 0.1; t1 = C('86:bulurlar') - 0.1
S.use('oohs', t0); S.chord('oohs', t0, t1 - t0, ['D4', 'Eb4', 'Ab4'], 60)
S.dyn('oohs', [(t0, 0), (t0 + 6, 35), (455, 40), (461, 55), (474, 70), (478.5, 20), (t1 - 1, 50), (t1, 0)])
S.use('cb', t0); S.note('cb', t0, t1 - t0, 'D1', 80); S.dyn('cb', [(t0, 0), (t0 + 4, 50), (478.5, 30), (t1, 70)])
S.use('pno', t0)
for tt_ in np.arange(t0 + 2, t1 - 2, 6.5):
    S.note('pno', tt_, 5, 'D1', 70); S.note('pno', tt_ + 0.01, 5, 'D2', 50)
S.use('vlnT', 455); S.chord('vlnT', C('79:gece'), t1 - C('79:gece'), ['D6', 'Eb6'], 60)
S.dyn('vlnT', [(C('79:gece'), 0), (458, 55), (464, 40), (470, 80), (476, 100), (478.4, 0), (486, 0), (t1 - 1, 60), (t1, 0)])
S.use('glass', 455); S.chord('glass', 457.6, 20, ['A4', 'Bb4', 'E5'], 50); S.dyn('glass', [(457.6, 0), (462, 50), (477, 55), (478.4, 0)])
S.note('timp', C('86:bulurlar') - 0.02, 2, 'D2', 100)

# M15 Üçüncü gün: bor (493.2 – 557.6)
t0 = C('86:üçüncü'); t1 = C('96:bor') - 0.1
S.hit(t0 + 0.02, 0.7, low_brass=False)
ostinato('vcP', t0 + 1.5, 514.0, 84, ['D3', None, 'D3', 'A2', 'D3', None, 'Eb3', 'D3'], vel=66)
S.dyn('vcP', [(t0, 80), (514, 80)])
strings_pad(t0 + 1.5, 514.0, [('Dm', 4), ('Eb', 4)], 84, vel=55, insts=('vla', 'vc', 'cb'), top=('D3', 'A4'), dyn=[(t0, 0), (500, 50), (513, 50), (514.3, 0)])
S.use('cb', 514); S.note('cb', 514.3, 26, 'A1', 80); S.dyn('cb', [(514.3, 10), (518, 60), (539, 60), (540.5, 0)])
S.use('vc', 514); S.note('vc', 514.3, 26, 'E2', 70); S.dyn('vc', [(514.3, 10), (518, 50), (539, 50), (540.5, 0)])
mel('ney', 515.0, NEY, 60, vel=74, legato=1.02, dyn=[(515, 50), (520, 75), (533, 70), (540, 0)])
S.use('glass', 524); S.chord('glass', 524.0, 13, ['E5', 'A5', 'B5'], 50); S.dyn('glass', [(524, 0), (527, 55), (536, 30), (537, 0)])
S.use('choir', 526); S.chord('choir', 526.6, 8, ['A3', 'E4', 'B4'], 60); S.dyn('choir', [(526.6, 0), (529, 60), (534.5, 0)])
strings_pad(540.5, t1, [('Dm', 4), ('Bb', 4), ('Gm', 4), ('A', 4)], 70, vel=58, top=('A3', 'D5'), dyn=[(540.5, 10), (548, 50), (552, 70), (t1, 85)])
S.roll('timp', 554.5, t1, 'D2', 20, 100)
S.swell_cym(t1, 2.5, 80)

# M16 Bor bilimi (557.6 – 604.4)
t0 = C('96:bor') - 0.05; t1 = C('103:bir') - 0.2
S.hit(t0 + 0.05, 1.0)
prog = [('Dm', 4), ('Bb', 4), ('F', 4), ('C', 4)]
arp('pno', t0 + 1.0, 598.0, prog, 104, step=0.5, order=(0, 2, 1, 3), lo='D4', hi='D5', vel=56)
S.cc('pno', t0 + 1, 64, 0)
strings_pad(t0 + 1.0, 598.0, prog, 104, vel=58, top=('A3', 'A5'), dyn=[(t0 + 1, 20), (565, 55), (590, 60), (597, 45), (598.5, 0)])
ostinato('vcF', 568.0, 598.0, 104, ['D3', 'D3', 'A2', 'D3', 'Bb2', 'Bb2', 'F2', 'Bb2'], vel=58)
S.dyn('vcF', [(568, 30), (580, 55), (598, 40)])
strings_pad(598.0, t1, [('Dm', 4), ('Gm', 4)], 60, vel=52, insts=('vla', 'vc', 'cb'), top=('D3', 'A4'), dyn=[(598, 40), (t1, 0)])
mel('cl', 598.6, [('A4', 1), ('G4', .5), ('F4', .5), ('E4', 2), ('D4', 3)], 60, vel=62)

# M17 Borumuza dokunma (604.4 – 617.0)
t0 = C('103:bir') - 0.1; t1 = C('106:ve') - 0.2
bassline('pizz', t0, t1, MAH_P, 116, pattern=((0, 0, 1), (1, 7, .7), (2, 0, .85), (3, 7, .7)), octave='D3', vel=66, dur=0.4)
arp('koto', t0, t1, MAH_P, 116, step=0.5, order=(0, 2, 1, 2), lo='D4', hi='A5', vel=58)
mel('cl', t0 + 1.0, MAH, 116, vel=70, legato=0.8)

# M18 Kalkan (617.1 – 645.3)
t0 = C('106:ve') - 0.1; t1 = C('110:dördüncü') - 0.1
strings_pad(t0, t1, [('Dm', 4), ('Eb', 4), ('Dm', 4), ('Bb', 2), ('A', 2)], 80, vel=62, dyn=[(t0, 20), (630, 60), (636, 50), (641, 75), (t1 - 0.5, 90), (t1, 0)])
mel('tbn', 624.0, GEMI, 52, vel=80, tr=12, dyn=[(624, 40), (630, 80), (634, 50)])
mel('hn', 633.0, [('D4', 2), ('Eb4', 1), ('D4', 1), ('A3', 4)], 52, vel=70, dyn=[(633, 40), (640, 70), (644, 50)])
S.roll('timp', 642.0, t1, 'D2', 30, 105)
S.swell_cym(t1, 2.0, 85)

# M19 Dördüncü gün: açıklık (645.4 – 703.6)
t0 = C('110:dördüncü'); t1 = C('120:ve#2') - 0.2
S.hit(t0 + 0.02, 0.7, low_brass=False)
S.use('pizz', t0)
for i in range(int((684 - (t0 + 1.2)) / 0.5)):
    S.note('pizz', t0 + 1.2 + i * 0.5, 0.15, 'A4' if i % 2 == 0 else 'D5', 46 if i % 2 else 56)
strings_pad(t0 + 1.2, 684.0, [('Dm', 4), ('Bb', 4), ('Gm', 4), ('A', 4)], 60, vel=55, top=('A3', 'D5'), dyn=[(t0 + 1.2, 10), (655, 50), (670, 55), (680, 65), (684.2, 70)])
mel('cel', 660.4, [('D6', .5), ('A5', .5), ('F6', .5), ('E6', 1.5)], 80, vel=50)
strings_pad(684.0, 696.0, UMUT_P[:4], 76, vel=72, top=('D4', 'A5'), dyn=[(684, 60), (688, 85), (694, 70), (696.3, 30)])
mel('hn', 684.2, UMUT[:12], 76, vel=80, dyn=[(684.2, 60), (688, 90), (695, 60), (696.3, 0)])
S.hit(C('117:dedenin') - 0.02, 0.6, low_brass=False)
ostinato('vcF', 696.0, t1, 96, TENS, vel=60)
S.dyn('vcF', [(696, 30), (t1 - 1, 60), (t1, 0)])
S.use('vla', 696); S.chord('vla', 696.4, t1 - 696.4, ['D4', 'Eb4'], 55); S.dyn('vla', [(696.4, 10), (t1, 50)])

# M20 Mors (703.8 – 752.6) — minimal: alttan dron + uzak nabız, ışık yaklaşınca küme
t0 = C('120:ve#2') - 0.1; t1 = C('128') - 0.25
S.use('cb', t0); S.note('cb', t0, t1 - t0, 'D1', 80); S.dyn('cb', [(t0, 20), (t0 + 3, 50), (745, 45), (t1, 30)])
S.use('vc', t0); S.note('vc', t0, t1 - t0, 'A2', 60); S.dyn('vc', [(t0, 10), (t0 + 3, 40), (t1, 30)])
S.use('timp', t0)
for tt_ in np.arange(714.8, 740.0, 1.6):
    S.note('timp', tt_, 0.8, 'D2', 40 + 20 * (tt_ - 714.8) / 25)
S.use('vlnT', 727); S.chord('vlnT', 727.2, 16, ['A5', 'Bb5', 'E6'], 60)
S.dyn('vlnT', [(727.2, 0), (733.7, 55), (737, 90), (739.9, 70), (742.5, 20), (743.4, 0)])
S.use('oohs', 744); S.chord('oohs', 744.0, t1 - 744, ['D4', 'A4'], 50); S.dyn('oohs', [(744, 0), (748, 40), (t1, 30)])

# M21 Ankara cevap veriyor (752.6 – 774.0)
t0 = C('128') - 0.2; t1 = C('131:5') - 0.1
strings_pad(t0 + 0.6, 764.5, [('F', 4), ('C', 4), ('Dm', 4), ('Bb', 4)], 60, vel=64, top=('A3', 'C6'), dyn=[(t0 + 0.6, 10), (756, 70), (762, 60), (764.8, 0)])
S.use('harp', t0); arp('harp', t0 + 0.6, 764.5, [('F', 4), ('C', 4), ('Dm', 4), ('Bb', 4)], 60, step=0.5, order=(0, 1, 2, 3, 4, 3, 2, 1), lo='C4', hi='C6', vel=52)
bassline('pizz', 764.9, t1, [('F', 2), ('C', 2), ('Dm', 2), ('A', 2)], 112, pattern=((0, 0, 1), (1, 7, .7)), octave='F3', vel=62, dur=0.4)
mel('cl', 765.4, [('F5', .5), ('E5', .5), ('D5', .5), ('C5', .5), ('Bb4', 1), ('A4', 1), (None, 1), ('C5', .5), ('D5', .5), ('E5', .5), ('F5', 1.5)], 112, vel=64, legato=0.85)

# M22 Beş gün (774.1 – 821.4)
t0 = C('131:5'); t1 = C('139:ama') - 0.2
S.hit(t0 + 0.02, 0.6, low_brass=False)
strings_pad(t0 + 1.0, t1, DEDE_P, 72, vel=58, top=('F3', 'A5'), dyn=[(t0 + 1, 10), (782, 50), (800, 58), (815, 70), (t1 - 1, 50), (t1, 0)])
arp('pno', t0 + 1.0, t1, DEDE_P, 72, step=0.5, order=(0, 2, 1, 3), lo='D4', hi='D5', vel=48)
k = mel('cl', 790.0, DEDE, 72, vel=68, legato=1.0, t_end=t1 - 2)
mel('vln', 806.0, [(p, b) for p, b in DEDE[15:]], 72, vel=66, tr=0, t_end=t1)

# M23 12:34 (821.6 – 886.6)
t0 = C('139:ama') - 0.1; t1 = C('150:merdivenlere') - 0.1
S.use('pizz', t0)
for i in range(int((856.3 - (t0 + 1)) / 0.5)):
    S.note('pizz', t0 + 1 + i * 0.5, 0.12, 'A4' if i % 2 == 0 else 'E5', 44 + (i * 0.4 if i < 60 else 24))
S.use('cb', t0); S.note('cb', t0, 856.3 - t0, 'D1', 80); S.dyn('cb', [(t0, 10), (t0 + 4, 45), (850, 65), (856.3, 80)])
S.use('vlnT', 840); S.chord('vlnT', 842.0, 14.3, ['A5', 'Bb5'], 60); S.dyn('vlnT', [(842, 0), (850, 50), (855.9, 100), (856.3, 0)])
S.roll('timp', 853.5, 856.3, 'D2', 30, 100)
t_o = C('145:ve')
S.hit(855.6, 1.0)
S.use('choir', t_o); S.chord('choir', 856.4, 14.0, ['D4', 'F#4', 'A4', 'D5'], 80); S.dyn('choir', [(856.4, 20), (861, 80), (866, 90), (870.4, 30)])
brass_pad(856.4, 870.4, [('D', 6), ('Bb', 6), ('C', 6)], 60, vel=80, inst='hn', lo='D3', hi='D4', dyn=[(856.4, 30), (862, 85), (868, 90), (870.4, 20)])
strings_pad(856.4, 870.4, [('D', 6), ('Bb', 6), ('C', 6)], 60, vel=70, top=('D4', 'A5'), dyn=[(856.4, 30), (862, 80), (868, 85), (870.4, 30)])
ostinato('vcF', 866.5, 880.4, 132, ['D3', 'D3', 'A3', 'D3', 'F3', 'D3', 'A3', 'D3'], vel=60)
S.dyn('vcF', [(866.5, 50), (874, 40), (880, 30)])
S.use('vla', 870); S.chord('vla', 870.4, 10, ['D4', 'A4'], 50); S.dyn('vla', [(870.4, 40), (880.4, 30)])
ostinato('vlnF', 880.5, t1 + 0.3, 140, ['D5', 'Eb5', 'D5', 'A4', 'D5', 'Eb5', 'F5', 'Eb5'], vel=70, step=0.25)
S.dyn('vlnF', [(880.5, 30), (884, 80), (t1 + 0.3, 110)])
S.roll('timp', 882.5, t1 + 0.4, 'D2', 30, 105)

# M24 Kaçış ve saklanma (886.7 – 937.3)
t0 = C('150:merdivenlere') - 0.1; t1 = C('158:altıncı') - 0.2
drums(t0, C('150:çarpıyorsun') - 0.3, 150, {'taiko:D2': [0, 0.5, 1, 1.5], 'timp:D2': [0, 1]}, bar=2, vel=90)
ostinato('vcF', t0, C('150:çarpıyorsun') - 0.35, 150, TENS, vel=90, step=0.5)
S.hit(C('150:çarpıyorsun') - 0.35, 0.8, low_brass=True)
t_h = C('150:çarpıyorsun') + 0.6
S.use('vlnT', t_h); S.chord('vlnT', t_h, 30, ['D6', 'Eb6'], 50); S.dyn('vlnT', [(t_h, 0), (t_h + 2, 30), (905, 40), (911.5, 80), (912.5, 25), (918, 30), (919, 0)])
S.use('cb', t_h); S.note('cb', t_h, t1 - t_h, 'D1', 70); S.dyn('cb', [(t_h, 0), (t_h + 2, 40), (918, 50), (t1, 30)])
S.use('timp', t_h)
for tt_ in np.arange(t_h + 1, 918, 1.0):
    S.note('timp', tt_, 0.5, 'D2', 34)
S.use('oohs', 911); S.chord('oohs', C('154:duyan') - 0.1, 5, ['D4', 'Eb4', 'Ab4', 'A4'], 70); S.dyn('oohs', [(C('154:duyan') - 0.1, 90), (916, 30), (917, 0)])
strings_pad(919.5, t1, [('Dm', 4), ('Bb', 4), ('Gm', 4), ('Eb', 4)], 60, vel=52, insts=('vla', 'vc', 'cb'), top=('D3', 'A4'), dyn=[(919.5, 0), (924, 45), (933, 45), (t1, 0)])
S.use('pno', 925)
for tt_, p in [(925.0, 'D5'), (927.0, 'A4'), (929.0, 'Bb4'), (931.0, 'A4'), (933.0, 'E4')]:
    S.note('pno', tt_, 2.5, p, 50)

# M25 Plan (937.5 – 1019.0)
t0 = C('158:altıncı'); t1 = C('172:ama') - 0.2
S.hit(t0 + 0.02, 0.8)
prog = [('Dm', 4), ('Bb', 4), ('C', 4), ('Dm', 4)]
ostinato('vcF', t0 + 1.0, 973.8, 100, TENS2, vel=64)
S.dyn('vcF', [(t0 + 1, 40), (950, 65), (973.8, 70)])
drums(946.3, 973.8, 100, {'kit:sn': [0, 0.5, 0.75, 1, 2, 2.5, 2.75, 3, 3.5], 'kit:bd': [0, 2], 'timp:D2': [0], 'timp:A2': [2]}, vel=58)
brass_pad(946.3, 973.8, prog, 100, vel=72, inst='hn', dyn=[(946.3, 40), (958, 70), (970, 80), (973.8, 60)])
S.hit(946.34, 0.7, low_brass=False)
S.use('tpt', 970); S.line('tpt', 970.8, [('A4', 1), ('D5', 1), ('D5', .5), ('E5', .5), ('F5', 2)], 100, 76, 0.95)
S.dyn('tpt', [(970.8, 80)])
bassline('cbP', 974.0, 989.0, [('D', 2), ('A', 2), ('D', 2), ('G', 1), ('A', 1)], 116, pattern=((0, 0, 1), (1, 7, .7)), octave='D3', vel=70, dur=0.45)
S.use('bsn', 974.0)
k = 975.0
while k < 988:
    k = S.line('bsn', k, [('D3', .5), ('F3', .5), ('A3', .5), (None, .5), ('G3', .5), ('Bb3', .5), ('A3', 1), ('F3', .5), ('E3', .5), ('D3', 1), (None, 1)], 116, 70, 0.6)
S.use('tba', 983)
S.note('tba', C('166:ağır') - 0.35, 0.6, 'D2', 95); S.note('tba', C('166:ağır') + 0.05, 1.0, 'A1', 100)
strings_pad(989.0, t1, [('Dm', 4), ('Eb', 4), ('Dm', 4), ('A', 4)], 84, vel=60, top=('A3', 'D5'), dyn=[(989, 20), (995, 50), (1005, 60), (1013, 70), (t1, 55)])
ostinato('vcF', 995.0, t1, 84, TENS, vel=58)
S.dyn('vcF', [(995, 30), (1010, 60), (t1, 50)])
S.use('timp', 1001); S.note('timp', C('169:ama') + 0.02, 1.0, 'D2', 70)

# M26 Yemin (1019.0 – 1052.6)
t0 = C('172:ama') - 0.1; t1 = C('177:çatıya') - 0.2
strings_pad(t0, 1037.5, DEDE_P, 66, vel=60, top=('F3', 'A5'), dyn=[(t0, 10), (1025, 50), (1032, 60), (1037.6, 70)])
mel('pno', t0 + 0.3, DEDE[:14], 66, vel=58, t_end=1037.0)
S.cc('pno', t0 + 0.2, 64, 127); S.cc('pno', 1037.3, 64, 0)
S.note('timp', C('175:gözlüğünü') + 0.05, 1.2, 'A2', 60)
S.hit(C('175:bulsunlar') - 0.05, 0.6, low_brass=False)
strings_pad(1038.0, t1, UMUT_PD, 70, vel=78, top=('D4', 'D6'), dyn=[(1038, 50), (1044, 80), (1048, 100), (t1 - 0.5, 110), (t1, 0)])
mel('hn', 1038.3, UMUT, 70, vel=86, dyn=[(1038.3, 60), (1046, 100), (t1, 105)], t_end=t1)
S.use('choir', 1044); S.chord('choir', 1044.2, t1 - 1044.2, ['D4', 'A4', 'D5'], 80); S.dyn('choir', [(1044.2, 20), (1050, 100), (t1 - 0.2, 110), (t1, 0)])
S.roll('timp', 1048.5, t1 - 0.05, 'D2', 30, 110)
S.hit(t1 - 0.05, 1.0)

# M27 Kuşatma (1052.6 – 1103.0)
t0 = C('177:çatıya') - 0.1; t1 = C('185:yedinci') - 0.15
S.use('cb', t0); S.note('cb', t0, t1 - t0, 'D1', 85); S.dyn('cb', [(t0, 30), (t0 + 4, 55), (1086, 75), (1096, 95), (C('185:susuyor'), 100), (C('185:susuyor') + 0.1, 0)])
ostinato('vcF', t0 + 2, C('185:susuyor'), 96, TENS, vel=60)
S.dyn('vcF', [(t0 + 2, 20), (1068, 55), (1086, 80), (1096, 100), (C('185:susuyor') - 0.05, 110), (C('185:susuyor'), 0)])
S.use('choir', 1068); S.chord('choir', C('180:gemiden'), C('185:susuyor') - C('180:gemiden'), ['D4', 'Eb4', 'A4', 'Bb4'], 70)
S.dyn('choir', [(C('180:gemiden'), 0), (1074, 55), (1086, 75), (1096, 100), (C('185:susuyor'), 0)])
for sp in ('182:beşinci', '182:dördüncü'):
    S.hit(C(sp) + 0.02, 0.5, low_brass=False, cym=False)
S.hit(C('183') + 0.02, 0.9)
mel('vln', 1087.5, [(p, b) for p, b in DEDE[:8]], 60, vel=80, tr=12, dyn=[(1087.5, 60), (1094, 95), (1097, 100), (C('185:susuyor'), 0)], t_end=C('185:susuyor') - 0.1)
S.roll('timp', 1094.0, C('184:penceresi') + 0.4, 'D2', 30, 110)
S.hit(C('184:penceresi') + 0.4, 1.2)
# 'Mors susuyor' -> mutlak sessizlik

# M28 Yedinci gün: şafak (1103.2 – 1143.1)
t0 = C('185:yedinci') - 0.05; t1 = 1143.12
S.hit(t0 + 0.02, 0.6, low_brass=False)
S.use('vln', t0 + 1); S.chord('vln', t0 + 1.2, t1 - t0 - 1.2, ['A5', 'E6'], 50); S.dyn('vln', [(t0 + 1.2, 0), (1112, 30), (1130, 45), (1137, 70), (t1 - 0.05, 85), (t1, 0)])
S.use('cb', t0); S.note('cb', t0 + 1.2, t1 - t0 - 1.2, 'D1', 75); S.dyn('cb', [(t0 + 1.2, 0), (1110, 40), (1131, 60), (t1 - 0.05, 85), (t1, 0)])
S.use('pizz', t0)
for i in range(int((1137 - (t0 + 2)) / 1.0)):
    S.note('pizz', t0 + 2 + i * 1.0, 0.1, 'D4', 42)
for sp in [1121.86, 1124.97, 1127.76, 1130.92]:
    S.note('timp', sp, 1.0, 'D2', 74)
S.use('oohs', 1131); S.chord('oohs', C('190:geminin'), t1 - C('190:geminin'), ['D4', 'A4', 'E5'], 70); S.dyn('oohs', [(C('190:geminin'), 0), (1136, 70), (t1 - 0.05, 95), (t1, 0)])
for a, L in [(1140.24, .3), (1141.07, .72), (1141.89, .3), (1142.64, .5)]:
    S.note('timp', a, 0.4, 'A2', 80)
silence_all(1143.15)

# M29 ATEŞ (1145.9 – 1168.2)
t0 = 1145.9; t1 = C('196:metalin') - 0.35
S.hit(t0, 1.3)
bpm = 138
drums(t0, t1, bpm, {'taiko:D2': [0, 0.75, 1.5, 2, 2.5, 3, 3.5], 'kit:bd': [0, 2], 'timp:D2': [0, 1, 2, 3], 'kit:sn': [1, 3, 3.5]}, vel=100)
ostinato('vcF', t0, t1, bpm, ['D3', 'D3', 'D3', 'D3', 'Eb3', 'D3', 'C3', 'D3'], vel=100, step=0.5)
ostinato('cbF', t0, t1, bpm, ['D2', 'D2', 'D2', 'D2', 'Eb2', 'D2', 'C2', 'D2'], vel=100, step=0.5)
ostinato('vlnF', t0 + 60 / bpm * 8, t1, bpm, ['D5', 'A4', 'D5', 'F5', 'E5', 'D5', 'C5', 'A4'], vel=90, step=0.5)
S.dyn('vcF', [(t0, 110), (t1, 120)]); S.dyn('cbF', [(t0, 110), (t1, 120)]); S.dyn('vlnF', [(t0, 80), (t1, 115)])
mel('hn', t0 + 60 / bpm * 8, [(p, b) for p, b in UMUT], bpm / 2, vel=100, dyn=[(t0, 90), (t1, 120)], t_end=t1)
brass_pad(t0 + 60 / bpm * 8, t1, [(c, b / 2) for c, b in UMUT_P], bpm / 2, vel=95, inst='tbn', lo='D2', hi='D3', dyn=[(t0, 90), (t1, 120)])
S.use('choir', t0); S.chord('choir', t0 + 3.5, t1 - t0 - 3.5, ['D4', 'A4', 'D5'], 90); S.dyn('choir', [(t0 + 3.5, 40), (1155, 90), (t1, 115)])
for sp in ('194:ilk', '194:sonra', '195:sonra'):
    S.hit(C(sp, '195'), 1.0, low_brass=False)
S.hit(t1 - 0.02, 1.3)
silence_all(t1 + 0.4)

# M30 Uyanış (1173.0 – 1204.6)
t0 = C('196:ve#2', '196:ve') + 0.3; t1 = C('202') - 0.2
strings_pad(t0, 1192.0, [('Bb', 4), ('F', 4), ('Gm', 4), ('D', 4)], 72, vel=62, top=('D4', 'D6'), dyn=[(t0, 0), (1180, 55), (1188, 75), (1192.3, 50)])
arp('pno', t0 + 1.0, 1192.0, [('Bb', 4), ('F', 4), ('Gm', 4), ('D', 4)], 72, step=0.25, order=(0, 1, 2, 3, 4, 3, 2, 1), lo='D4', hi='D6', vel=50)
S.cc('pno', t0, 64, 0)
bassline('pizz', 1192.0, t1, [('D', 2), ('A', 2), ('Bm', 2), ('A', 2)], 116, pattern=((0, 0, 1), (1, 7, .7)), octave='D3', vel=64, dur=0.4)
arp('koto', 1192.0, t1, [('D', 2), ('A', 2), ('Bm', 2), ('A', 2)], 116, step=0.5, order=(0, 2, 1, 2), lo='D4', hi='A5', vel=52)

# M31 Zafer (1204.8 – 1233.0)
t0 = C('202') - 0.05; t1 = C('206:koşuyorsun') - 0.4
S.hit(t0 + 0.05, 0.9)
strings_pad(t0 + 0.4, 1216.0, [('Dm', 4), ('Bb', 4), ('F', 4), ('A', 4)], 64, vel=64, top=('A3', 'D6'), dyn=[(t0, 30), (1210, 70), (1215, 85), (1216.3, 60)])
mel('hn', t0 + 0.6, UMUT[:12], 64, vel=82, dyn=[(t0, 60), (1214, 95)], t_end=1216.2)
S.hit(C('204:bütün') - 0.05, 1.1)
bpm = 120
strings_pad(1216.3, t1, UMUT_PD, bpm / 2, vel=82, top=('D4', 'D6'), dyn=[(1216.3, 80), (t1 - 1, 100), (t1, 0)])
mel('tpt', 1216.5, UMUT, bpm / 2, vel=90, tr=0, dyn=[(1216.5, 85), (t1 - 0.5, 105), (t1, 0)], t_end=t1)
S.use('choir', 1216); S.chord('choir', 1216.4, t1 - 1216.4, ['D4', 'F#4', 'A4', 'D5'], 85); S.dyn('choir', [(1216.4, 60), (t1 - 1, 100), (t1, 0)])
drums(1216.3, t1, bpm, {'kit:bd': [0, 1.5, 2], 'kit:sn': [1, 3, 3.5], 'kit:crash': [0], 'taiko:D2': [0, 2]}, vel=90)
arp('koto', 1216.3, t1, [('D', 4), ('G', 4), ('A', 4), ('D', 4)], bpm, step=0.25, order=(0, 1, 2, 3, 2, 1), lo='D4', hi='D6', vel=58)

# M32 Oda boş (1233.0 – 1261.2)
t0 = C('206:koşuyorsun') - 0.35; t1 = C('211') - 0.15
ostinato('vlnF', t0, C('207:kapı') - 0.2, 144, ['D5', 'Eb5', 'D5', 'A4'], vel=80, step=0.25)
S.dyn('vlnF', [(t0, 70), (C('207:kapı') - 0.2, 90), (C('207:kapı'), 0)])
S.note('timp', C('207:oda') - 0.02, 2.0, 'D2', 80)
tp = C('207:oda') + 1.0
S.cc('pnoM', tp - 0.1, 64, 127); S.use('pnoM', tp)
e = S.line('pnoM', tp, DEDE, 52, 54, 1.0, t_end=t1 - 3)
S.cc('pnoM', t1 - 0.2, 64, 0)
strings_pad(tp + 4, t1, DEDE_P, 52, vel=48, insts=('vla', 'vc', 'cb'), top=('F3', 'F4'), dyn=[(tp + 4, 0), (1248, 35), (1256, 40), (C('210:dede#2', '210') - 0.2, 45), (t1, 0)])
S.note('pnoM', C('210:dede#2', '210:yok') + 0.3, 6.0, 'D2', 50); S.note('pnoM', C('210:dede#2', '210:yok') + 0.31, 6.0, 'A2', 40)

# M33 Onuncu gün (1261.4 – 1326.6)
t0 = C('211'); t1 = C('222') - 0.2
S.hit(t0 + 0.02, 0.6, low_brass=False)
prog = [('F', 4), ('C', 4), ('Dm', 4), ('Bb', 4)]
strings_pad(t0 + 0.8, t1, prog, 84, vel=58, top=('A3', 'C6'), dyn=[(t0 + 0.8, 10), (1270, 55), (1310, 58), (t1 - 1.5, 40), (t1, 0)])
arp('pno', t0 + 0.8, t1, prog, 84, step=0.5, order=(0, 2, 1, 3), lo='C4', hi='C6', vel=48)
k = 1272.0
while k < 1300:
    k = mel('cl', k, [('C5', 1), ('D5', .5), ('E5', .5), ('F5', 2), ('E5', 1), ('D5', 1), ('C5', 2), ('A4', 1), ('Bb4', 1), ('C5', 2), ('D5', 1), ('C5', 1), ('A4', 4)], 84, vel=64, legato=0.98) + 60 / 84 * 2
bassline('pizz', 1309.2, 1319.0, [('F', 2), ('C', 2), ('Dm', 2), ('A', 2)], 120, pattern=((0, 0, 1), (1, 7, .7)), octave='F3', vel=62, dur=0.4)
S.use('glock', 1314); S.note('glock', C('219:bor') + 0.1, 1.0, 'D6', 50); S.note('glock', C('219:bor') + 0.4, 1.0, 'A5', 46)

# M34 Ritim (1326.8 – 1373.6)
t0 = C('222') - 0.1; t1 = C('229:ama') - 0.2
S.use('cb', t0); S.note('cb', t0, 1351.0 - t0, 'D1', 70); S.dyn('cb', [(t0, 0), (t0 + 4, 35), (1350, 35), (1351.2, 0)])
S.use('vla', t0); S.chord('vla', t0 + 1, 1337.4 - t0 - 1, ['F3', 'A3', 'D4'], 50); S.dyn('vla', [(t0 + 1, 0), (1332, 35), (1337.5, 0)])
S.use('vlnT', 1339); S.chord('vlnT', 1341.3, 8, ['D6'], 50); S.dyn('vlnT', [(1341.3, 0), (1344, 30), (1349, 0)])
tp = C('225:morz') - 0.3
strings_pad(tp, t1, DEDE_P, 60, vel=62, top=('F3', 'A5'), dyn=[(tp, 0), (1352, 45), (1357, 55), (1363, 70), (1369, 85), (t1 - 1, 70), (t1, 0)])
S.cc('pno', tp - 0.1, 64, 127)
mel('pno', tp + 0.2, DEDE, 60, vel=60, t_end=t1 - 1)
S.cc('pno', t1, 64, 0)
mel('vln', 1363.2, [(p, b) for p, b in DEDE[15:]], 60, vel=78, dyn=[(1363.2, 60), (1369, 95), (t1 - 1, 80), (t1, 0)], t_end=t1)

# M35 Makine (1373.8 – 1416.9)
t0 = C('229:ama') - 0.1; t1 = C('236:duyan') - 0.1
S.use('glass', t0); S.chord('glass', t0, t1 - t0, ['D4', 'C#5', 'G5'], 55); S.dyn('glass', [(t0, 0), (1378, 50), (1395, 60), (1405, 40), (t1, 70)])
S.use('cb', t0); S.note('cb', t0, t1 - t0, 'D1', 80); S.dyn('cb', [(t0, 10), (1380, 50), (1395, 70), (1407, 40), (t1, 90)])
S.use('vc', t0); S.note('vc', t0 + 0.5, t1 - t0, 'Eb2', 60); S.dyn('vc', [(t0 + 0.5, 0), (1385, 40), (1397, 60), (t1, 60)])
S.use('timp', 1376)
for tt_ in np.arange(1376.4, 1388.0, 0.75):
    S.note('timp', tt_, 0.3, 'D2', 50, hum=False)
mel('tbn', 1389.0, GEMI, 46, vel=86, tr=12, dyn=[(1389, 50), (1395, 95), (1400, 60)])
S.hit(C('233:z', '233') + 0.2, 0.9)
S.use('vlnT', 1407); S.chord('vlnT', 1407.0, t1 - 1407, ['D6', 'Eb6', 'A6'], 60); S.dyn('vlnT', [(1407, 0), (1411, 40), (t1 - 0.1, 110), (t1, 0)])
S.roll('timp', 1413.0, t1, 'D2', 20, 112)
S.swell_cym(t1, 3.5, 90)

# M36 Duyan var mı? (1417 – END)
t0 = C('236:duyan') - 0.05
S.hit(t0, 1.3)
S.use('cb', t0); S.note('cb', t0 + 0.1, END - t0 - 0.5, 'D1', 80); S.dyn('cb', [(t0, 100), (t0 + 4, 45), (END - 3, 30), (END - 0.3, 0)])
S.use('oohs', t0); S.chord('oohs', t0 + 0.5, END - t0 - 1, ['D4', 'A4'], 60); S.dyn('oohs', [(t0 + 0.5, 0), (1422, 40), (END - 4, 30), (END - 0.5, 0)])
S.use('mbox', t0 + 2)
S.line('mbox', t0 + 3.0, DEDE[:14], 66, 48, 1.0, tr=12, t_end=END - 3)
S.dyn('mbox', [(t0 + 3, 90), (END - 6, 70), (END - 0.5, 0)])

n = S.save(os.path.join(OUT, 'music.mid'), END + 2)
print('olay', n)
