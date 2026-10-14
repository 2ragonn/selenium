"""Müzik yazım çekirdeği: saniye tabanlı olaylar -> MIDI (MuseScore General / GeneralUser GS kanalları)."""
import mido, numpy as np

NOTE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def P(n):
    """'D4' / 'Eb2' / 'F#5' -> MIDI; sayı ise aynen."""
    if n is None or isinstance(n, (int, np.integer)):
        return n
    s = n[0].upper(); i = 1; acc = 0
    while i < len(n) and n[i] in '#b':
        acc += 1 if n[i] == '#' else -1; i += 1
    return 12 * (int(n[i:]) + 1) + NOTE[s] + acc


# enstrüman -> (kanal, bank, program, ifade CC'si)
INST = {
    'vln': (0, 21, 49, 2), 'vlnF': (0, 21, 48, 2), 'vlnT': (0, 21, 44, 2),
    'vla': (1, 31, 49, 2), 'vlaF': (1, 31, 48, 2), 'vlaT': (1, 31, 44, 2), 'vln2': (1, 26, 49, 2), 'vln2T': (1, 26, 44, 2),
    'vc': (2, 41, 49, 2), 'vcF': (2, 41, 48, 2), 'vcT': (2, 41, 44, 2),
    'cb': (3, 51, 49, 2), 'cbF': (3, 51, 48, 2),
    'pizz': (4, 0, 45, 11), 'vcP': (4, 40, 45, 11), 'cbP': (4, 50, 45, 11),
    'hn': (5, 17, 60, 2),
    'tbn': (6, 17, 57, 2), 'tba': (6, 17, 58, 2), 'brass': (6, 17, 61, 2),
    'choir': (7, 17, 52, 2), 'oohs': (7, 17, 53, 2),
    'timp': (8, 0, 47, 11),
    'kit': (9, 128, 48, 11),
    'pno': (10, 0, 0, 11), 'pnoM': (10, 8, 0, 11),
    'cl': (11, 17, 71, 2), 'fl': (11, 17, 73, 2), 'ob': (11, 17, 68, 2), 'ney': (11, 17, 77, 2), 'bsn': (11, 17, 70, 2), 'eh': (11, 17, 69, 2),
    'cel': (12, 0, 8, 11), 'harp': (12, 0, 46, 11), 'koto': (12, 0, 107, 11), 'mbox': (12, 0, 10, 11),
    'taiko': (13, 0, 116, 11), 'bd': (13, 8, 116, 11),
    'glass': (14, 17, 92, 2), 'rcym': (14, 0, 119, 11), 'tbells': (14, 0, 14, 11), 'glock': (14, 0, 9, 11),
    'tpt': (15, 17, 56, 2), 'vlnS': (15, 17, 40, 2), 'wblock': (15, 1, 115, 11),
}
KIT = {'bd': 35, 'bd2': 36, 'sn': 38, 'sn2': 40, 'crash': 49, 'crash2': 57, 'ride': 51, 'china': 52, 'splash': 55, 'tam': 52, 'tri': 81, 'ttri': 80}


class Score:
    def __init__(self, seed=1):
        self.ev = []  # (t, prio, msg)
        self.rng = np.random.default_rng(seed)
        self.cur = {}  # kanal -> (bank, prog) zaman sıralı ekleme beklenmez; program değişimlerini 'use' ile açıkça veririz

    # ------------------------------------------------ temel
    def _add(self, t, msg, prio=1):
        self.ev.append((max(0.0, float(t)), prio, msg))

    def use(self, name, t):
        ch, bank, prog, ecc = INST[name]
        if self.cur.get(ch) == (bank, prog):
            return
        self.cur[ch] = (bank, prog)
        if ch != 9:
            self._add(t - 0.02, mido.Message('control_change', channel=ch, control=0, value=bank), 0)
            self._add(t - 0.02, mido.Message('control_change', channel=ch, control=32, value=0), 0)
        self._add(t - 0.019, mido.Message('program_change', channel=ch, program=prog), 0)

    def cc(self, name, t, num, val):
        ch = INST[name][0]
        self._add(t, mido.Message('control_change', channel=ch, control=num, value=int(np.clip(val, 0, 127))), 0)

    def dyn(self, name, pts, rate=12):
        """İfade eğrisi: pts=[(t, değer 0..127)], Expr enstrümanlarında CC2, diğerlerinde CC11."""
        ch, _, _, num = INST[name]
        ts = np.array([p[0] for p in pts]); vs = np.array([p[1] for p in pts])
        n = max(2, int((ts[-1] - ts[0]) * rate))
        last = None
        for t in np.linspace(ts[0], ts[-1], n):
            v = int(np.interp(t, ts, vs))
            if v != last:
                self._add(t, mido.Message('control_change', channel=ch, control=num, value=int(np.clip(v, 0, 127))), 0)
                last = v

    def note(self, name, t, dur, pitch, vel=80, hum=True):
        if pitch is None or dur <= 0:
            return
        ch = INST[name][0]
        p = P(pitch) if name != 'kit' else (KIT[pitch] if isinstance(pitch, str) else pitch)
        if hum:
            t = t + self.rng.normal(0, 0.006); vel = vel + self.rng.normal(0, 4)
        v = int(np.clip(vel, 1, 127))
        self._add(t, mido.Message('note_on', channel=ch, note=int(p), velocity=v), 2)
        self._add(t + dur, mido.Message('note_off', channel=ch, note=int(p), velocity=0), 1)

    # ------------------------------------------------ müzikal kalıplar
    def chord(self, name, t, dur, pitches, vel=70, spread=0.0):
        for i, p in enumerate(pitches):
            self.note(name, t + i * spread, dur, p, vel)

    def line(self, name, t0, notes, bpm, vel=80, legato=1.02, tr=0, accent=None, t_end=None):
        """notes: [(perde|None, vuruş)] — legato>1 bindirme (yaylı/üflemeli legato)."""
        b = 60.0 / bpm; t = t0
        for i, (p, beats) in enumerate(notes):
            if t_end is not None and t >= t_end:
                break
            d = beats * b
            if p is not None:
                pp = P(p) + tr if not isinstance(p, list) else None
                v = vel * (accent[i % len(accent)] if accent else 1.0)
                if isinstance(p, list):
                    for q in p:
                        self.note(name, t, d * legato, P(q) + tr, v)
                else:
                    self.note(name, t, d * legato, pp, v)
            t += d
        return t

    def loop(self, name, t0, t1, notes, bpm, vel=80, legato=0.9, tr=0, accent=None):
        t = t0
        while t < t1 - 0.05:
            t = self.line(name, t, notes, bpm, vel, legato, tr, accent, t_end=t1)

    def roll(self, name, t0, t1, pitch, v0, v1, rate=16):
        n = max(1, int((t1 - t0) * rate))
        for i in range(n):
            k = i / max(1, n - 1)
            self.note(name, t0 + i / rate, 1.0 / rate * 1.5, pitch, v0 + (v1 - v0) * k, hum=False)

    def hit(self, t, size=1.0, root='D', low_brass=True, cym=True):
        """Orkestral vuruş: timpani + konser davulu + alçak bakır + zil."""
        r = P(root + '2')
        self.use('timp', t); self.use('kit', t)
        self.note('timp', t, 1.5, r, 105 * size)
        self.note('timp', t + 0.003, 1.5, r + 7, 85 * size)
        self.note('kit', t, 2.0, 'bd', 112 * size)
        if cym:
            self.note('kit', t, 3.0, 'crash', 96 * size)
        if low_brass:
            self.use('tbn', t)
            self.chord('tbn', t, 1.2 * size, [r, r + 7, r + 12], 105 * size)
            self.dyn('tbn', [(t - 0.05, 110), (t + 1.2 * size, 60)])

    def swell_cym(self, t_end, dur=2.5, vel=70):
        self.roll('kit', t_end - dur, t_end, 'ride', 15, vel, rate=18)

    def save(self, path, end_t):
        mid = mido.MidiFile(ticks_per_beat=960); tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage('set_tempo', tempo=500000))
        # başlangıç: tüm kanallarda ses/pan/ifade
        init = []
        pans = {0: 40, 1: 52, 2: 76, 3: 84, 4: 60, 5: 46, 6: 80, 7: 64, 8: 66, 10: 58, 11: 70, 12: 50, 13: 64, 14: 64, 15: 56}
        for ch in range(16):
            if ch == 9:
                continue
            init += [mido.Message('control_change', channel=ch, control=7, value=110), mido.Message('control_change', channel=ch, control=10, value=pans.get(ch, 64)),
                     mido.Message('control_change', channel=ch, control=11, value=127), mido.Message('control_change', channel=ch, control=2, value=90),
                     mido.Message('control_change', channel=ch, control=91, value=0), mido.Message('control_change', channel=ch, control=93, value=0)]
        for m in init:
            tr.append(m)
        ev = sorted(self.ev, key=lambda e: (e[0], e[1]))
        last = 0
        for t, _, m in ev:
            tick = int(round(t * 1920))
            m = m.copy(time=max(0, tick - last)); last = max(last, tick)
            tr.append(m)
        tr.append(mido.MetaMessage('end_of_track', time=int(max(0, end_t * 1920 - last))))
        mid.save(path)
        return len(ev)
