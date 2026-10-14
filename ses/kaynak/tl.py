"""Animasyon zaman çizelgesi: sahne/kelime zamanları ve cue() (JS motoruyla birebir)."""
import json, re, os

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, 'timeline.json')))
SCRIPT = D['SCRIPT']            # [baş, son, metin]
END = D['end']
WORDS = {}
for n, w, t in D['words']:
    WORDS.setdefault(n, []).append((w, t))


def tr_lower(s):
    s = s.replace('I', 'ı').replace('İ', 'i').lower()
    return re.sub(r'[^a-zçğıöşü0-9]', '', s)


WT = {n: [(tr_lower(w), t) for w, t in arr] for n, arr in WORDS.items()}


def cue(spec):
    if isinstance(spec, (int, float)):
        return float(spec)
    off = 0.0
    m = re.match(r'^(.*?)([+-]\d*\.?\d+)$', spec)
    if m and not re.match(r'^\d+$', spec):
        spec, off = m.group(1), float(m.group(2))
    m = re.match(r'^(\d+)(?:%([\d.]+))?(?::(.+?))?(?:#(\d+))?$', spec)
    if not m:
        raise ValueError('bad cue ' + spec)
    n = int(m.group(1))
    s0, s1 = SCRIPT[n - 1][0], SCRIPT[n - 1][1]
    if m.group(2):
        return s0 + (s1 - s0) * float(m.group(2)) + off
    if m.group(3):
        key = tr_lower(m.group(3)); occ = int(m.group(4) or 1); k = 0
        for w, t in WT[n]:
            if w.startswith(key):
                k += 1
                if k == occ:
                    return t + off
        raise ValueError('word not found %s' % spec)
    return s0 + off


def scene(n):
    return SCRIPT[n - 1][0], SCRIPT[n - 1][1]


SHOTS = D['shots']
MORSE_SHOTS = D['morse']
MORSE = D['MORSE']
SIG7 = D['SIG7']
EVTS = D['evts']
