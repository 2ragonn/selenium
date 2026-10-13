"""Animation timeline access (exported from the HTML): phrase times + shots."""
import json, os, re
D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'timeline.json')))
WD = D['words']; SHOTS = D['shots']; END = D['end']; DAYS = D['days']
TOTAL = END + 3.0

def nz(w):
    w = w.replace('I', 'ı').replace('İ', 'i').lower()
    return re.sub(r'[^a-zçğıöşü0-9]', '', w)

def T(phrase, after=0.0, lead=0.0):
    ps = [nz(x) for x in phrase.split() if nz(x)]
    for i in range(len(WD)):
        if WD[i]['t'] < after - 0.01: continue
        if all(i + j < len(WD) and WD[i + j]['w'] == ps[j] for j in range(len(ps))):
            return WD[i]['t'] - lead
    raise KeyError(phrase)

def shots(tpl=None, m=None):
    out = []
    for s in SHOTS:
        if tpl and s['tpl'] != tpl: continue
        if m is not None and s['p'].get('m') != m: continue
        out.append(s)
    return out

def shot_at(t):
    for s in SHOTS:
        if s['t0'] <= t < s['t1']: return s
    return SHOTS[-1]
