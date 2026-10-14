"""Ses işleme çekirdeği: gürültüler, filtreler, rezonatörler, Doppler, evrişimli yankı, dinamik."""
import numpy as np
import scipy.signal as sg
from numba import njit

SR = 48000
F32 = np.float32


def T(sec):
    return int(round(sec * SR))


def tt(n):
    return np.arange(n, dtype=np.float64) / SR


# ---------------------------------------------------------------- gürültüler
_PB = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
_PA = np.array([1, -2.494956002, 2.017265875, -0.522189400])


def white(n, rng):
    return rng.standard_normal(n)


def pink(n, rng):
    x = sg.lfilter(_PB, _PA, rng.standard_normal(n + 2000))[2000:]
    return x / (np.std(x) + 1e-12)


def brown(n, rng, leak=0.997):
    x = sg.lfilter([1.0], [1, -leak], rng.standard_normal(n + 4000))[4000:]
    x = hp(x, 12)
    return x / (np.std(x) + 1e-12)


def smooth_noise(n, rate_hz, rng, order=2):
    """Yavaş rastgele dalgalanma (-1..1 civarı) — modülasyon için."""
    m = max(4, int(n * rate_hz / SR) + 4)
    pts = rng.standard_normal(m)
    x = np.linspace(0, m - 3, n)
    i = np.floor(x).astype(int); f = x - i
    # Catmull-Rom
    p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[np.minimum(i + 3, m - 1)]
    y = 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)
    return y / 1.2


# ---------------------------------------------------------------- filtreler
def _cl(fc):
    return float(min(max(fc, 5.0), SR * 0.47))


def lp(x, fc, order=2):
    return sg.sosfilt(sg.butter(order, _cl(fc), 'low', fs=SR, output='sos'), x, axis=-1)


def hp(x, fc, order=2):
    return sg.sosfilt(sg.butter(order, _cl(fc), 'high', fs=SR, output='sos'), x, axis=-1)


def bp(x, lo, hi, order=2):
    return sg.sosfilt(sg.butter(order, [_cl(lo), _cl(hi)], 'band', fs=SR, output='sos'), x, axis=-1)


def _biquad(b, a, x):
    return sg.lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=-1)


def peq(x, f0, q, gain_db):
    A = 10 ** (gain_db / 40); w = 2 * np.pi * f0 / SR; al = np.sin(w) / (2 * q)
    return _biquad([1 + al * A, -2 * np.cos(w), 1 - al * A], [1 + al / A, -2 * np.cos(w), 1 - al / A], x)


def shelf(x, f0, gain_db, high=True):
    A = 10 ** (gain_db / 40); w = 2 * np.pi * f0 / SR; al = np.sin(w) / 2 * np.sqrt(2); c = np.cos(w); s = 2 * np.sqrt(A) * al
    if high:
        b = [A * ((A + 1) + (A - 1) * c + s), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - s)]
        a = [(A + 1) - (A - 1) * c + s, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - s]
    else:
        b = [A * ((A + 1) - (A - 1) * c + s), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - s)]
        a = [(A + 1) + (A - 1) * c + s, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - s]
    return _biquad(b, a, x)


@njit(cache=True)
def _svf(x, fc, q, mode):
    n = x.shape[0]; y = np.empty(n); ic1 = 0.0; ic2 = 0.0; k = 1.0 / q
    for i in range(n):
        g = np.tan(np.pi * min(max(fc[i], 5.0), 22000.0) / 48000.0)
        a1 = 1.0 / (1.0 + g * (g + k)); a2 = g * a1; a3 = g * a2
        v3 = x[i] - ic2; v1 = a1 * ic1 + a2 * v3; v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2 * v1 - ic1; ic2 = 2 * v2 - ic2
        if mode == 0:
            y[i] = v2
        elif mode == 1:
            y[i] = v1
        else:
            y[i] = x[i] - k * v1 - v2
    return y


def svf(x, fc, q=0.707, mode='lp'):
    """Zamanla değişen kesimli durum-değişkenli filtre (fc dizi ya da sayı)."""
    fc = np.broadcast_to(np.asarray(fc, dtype=np.float64), x.shape).copy()
    return _svf(np.ascontiguousarray(x, dtype=np.float64), fc, float(q), {'lp': 0, 'bp': 1, 'hp': 2}[mode])


@njit(cache=True)
def _resbank(exc, f, t60, amp, out):
    n = exc.shape[0]
    for m in range(f.shape[0]):
        r = np.exp(-6.907755 / (t60[m] * 48000.0)); w = 2 * np.pi * f[m] / 48000.0
        c1 = 2 * r * np.cos(w); c2 = -r * r; y1 = 0.0; y2 = 0.0; g = amp[m] * (1 - r)
        for i in range(n):
            y = g * exc[i] + c1 * y1 + c2 * y2
            out[i] += y; y2 = y1; y1 = y
    return out


def resbank(exc, freqs, t60s, amps):
    """Rezonatör bankası (modal sentez): cam, metal, tahta gövde."""
    f = np.asarray(freqs, float); keep = f < SR * 0.45
    out = np.zeros(len(exc))
    return _resbank(np.ascontiguousarray(exc, dtype=np.float64), f[keep], np.asarray(t60s, float)[keep] if np.ndim(t60s) else np.full(keep.sum(), float(t60s)),
                    np.asarray(amps, float)[keep] if np.ndim(amps) else np.full(keep.sum(), float(amps)), out)


@njit(cache=True)
def _vdelay(x, d):
    n = d.shape[0]; y = np.zeros(n); m = x.shape[0]
    for i in range(n):
        p = i - d[i]
        if p < 1 or p >= m - 2:
            continue
        j = int(p); f = p - j
        xm1 = x[j - 1]; x0 = x[j]; x1 = x[j + 1]; x2 = x[j + 2]
        c1 = 0.5 * (x1 - xm1); c2 = xm1 - 2.5 * x0 + 2 * x1 - 0.5 * x2; c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1)
        y[i] = ((c3 * f + c2) * f + c1) * f + x0
    return y


def vdelay(x, delay_samples):
    return _vdelay(np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(delay_samples, dtype=np.float64))


def varispeed(x, rate):
    """Değişken hızda okuma (rate dizi: 1=normal). Perde kaydırma/Doppler için."""
    pos = np.cumsum(np.broadcast_to(np.asarray(rate, float), (len(rate) if np.ndim(rate) else len(x),)))
    pos = pos - pos[0]
    pos = pos[pos < len(x) - 3]
    return np.interp(pos, np.arange(len(x)), x)


# ---------------------------------------------------------------- zarflar
def env(n, pts, curve=None):
    """Parçalı doğrusal zarf: pts = [(saniye, değer), ...]."""
    ts = np.array([p[0] for p in pts]) * SR; vs = np.array([p[1] for p in pts])
    e = np.interp(np.arange(n), ts, vs)
    return e if curve is None else e ** curve


def decay(n, t60, attack=0.0):
    e = np.exp(-6.907755 * np.arange(n) / (t60 * SR))
    if attack > 0:
        a = T(attack); e[:a] *= np.linspace(0, 1, a) if a > 0 else 1
    return e


def fade(x, fin=0.0, fout=0.0):
    x = x.copy(); n = x.shape[-1]
    a = min(T(fin), n); b = min(T(fout), n)
    if a > 0:
        x[..., :a] *= np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    if b > 0:
        x[..., n - b:] *= np.cos(np.linspace(0, np.pi / 2, b)) ** 2
    return x


def db(v):
    return 10 ** (v / 20)


# ---------------------------------------------------------------- stereo
def pan(x, p=0.0):
    """Mono -> stereo eşit güç (p: -1 sol, +1 sağ; dizi olabilir)."""
    a = (np.asarray(p) + 1) * np.pi / 4
    return np.vstack([x * np.cos(a), x * np.sin(a)])


def widen(x, rng, amount=1.0, ms=11.0):
    """Mono'dan geniş stereo: kısa farklı gecikmeler + tümgeçiren dekorelasyon."""
    d = T(ms / 1000)
    l = x; r = np.concatenate([np.zeros(d), x[:-d]]) if d > 0 else x
    ap = lambda s, g, k: sg.lfilter([g] + [0] * (k - 1) + [1], [1] + [0] * (k - 1) + [g], s)
    r = ap(ap(r, 0.6, 113), 0.5, 337)
    l = ap(l, 0.55, 211)
    mid = (l + r) / 2; side = (l - r) / 2 * amount
    return np.vstack([mid + side, mid - side])


def st(x, p=0.0):
    return x if x.ndim == 2 else pan(x, p)


# ---------------------------------------------------------------- yankı
def make_ir(t60=1.8, pre=0.015, er_span=0.06, er_n=10, damp=1.0, width=1.0, seed=7, bright=1.0):
    """Sentetik stereo darbe tepkisi: frekansa bağlı sönüm + erken yansımalar."""
    rng = np.random.default_rng(seed)
    n = T(pre + t60 * 1.15 + 0.05); t = tt(n)
    out = np.zeros((2, n))
    bands = [(20, 300, 1.25), (300, 2000, 1.0), (2000, 6000, 0.62 * damp), (6000, 16000, 0.32 * damp)]
    for ch in range(2):
        w = rng.standard_normal(n)
        acc = np.zeros(n)
        for lo, hi, mul in bands:
            e = np.exp(-6.907755 * np.maximum(t - pre, 0) / (t60 * mul))
            g = 1.0 if hi <= 6000 else 0.7 * bright
            acc += bp(w, lo, hi, 2) * e * g
        acc[:T(pre)] = 0
        # yumuşak başlangıç (difüzyon)
        k = T(0.02); acc[T(pre):T(pre) + k] *= np.linspace(0, 1, k)
        out[ch] = acc
    # erken yansımalar
    for i in range(er_n):
        for ch in range(2):
            p = pre * 0.5 + rng.random() * er_span
            a = (0.7 - 0.5 * i / er_n) * (0.6 + 0.4 * rng.random()) * (1 if rng.random() > .3 else -1)
            out[ch, min(T(p), n - 1)] += a * 6
    m = (out[0] + out[1]) / 2; s = (out[0] - out[1]) / 2 * width
    out = np.vstack([m + s, m - s])
    out /= np.sqrt(np.sum(out ** 2) / 2) + 1e-12
    return out


def convolve(x, ir):
    """x: (2,n) ya da mono; ir: (2,m) -> (2, n+m-1)."""
    if x.ndim == 1:
        return np.vstack([sg.oaconvolve(x, ir[0]), sg.oaconvolve(x, ir[1])])
    return np.vstack([sg.oaconvolve(x[0], ir[0]), sg.oaconvolve(x[1], ir[1])])


# ---------------------------------------------------------------- dinamik
@njit(cache=True)
def _release(gs, rc):
    n = gs.shape[0]; g = 1.0; out = np.empty(n)
    for i in range(n):
        t = gs[i]
        if t < g:
            g = t
        else:
            g = t + (g - t) * rc
        out[i] = g
    return out


def limiter(x, ceil_db=-1.0, look_ms=3.0, rel_ms=80.0):
    """İleri bakışlı tepe sınırlayıcı (stereo bağlı)."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    ceil = db(ceil_db); L = max(2, T(look_ms / 1000))
    pk = np.max(np.abs(x), axis=0)
    need = np.minimum(1.0, ceil / np.maximum(pk, 1e-12))
    # ileri pencere minimumu: gmin[i] = min(need[i:i+L])
    gmin = minimum_filter1d(need, size=L, origin=-(L // 2), mode='nearest')
    # nedensel ortalama: sm[i] = mean(gmin[i-L+1:i+1])
    sm = uniform_filter1d(gmin, size=L, origin=(L - 1) // 2, mode='nearest')
    sm = np.minimum(sm, gmin * 0 + sm)
    g = _release(sm, np.exp(-1.0 / max(1, T(rel_ms / 1000))))
    return x * g


@njit(cache=True)
def _env_follow(x, att, rel):
    n = x.shape[0]; y = np.empty(n); e = 0.0
    a = np.exp(-1.0 / att); r = np.exp(-1.0 / rel)
    for i in range(n):
        v = abs(x[i])
        e = v + (e - v) * (a if v > e else r)
        y[i] = e
    return y


def env_follow(x, att_ms=5, rel_ms=120):
    return _env_follow(np.ascontiguousarray(x, dtype=np.float64), max(1, T(att_ms / 1000)), max(1, T(rel_ms / 1000)))


def compress(x, thr_db=-18, ratio=3.0, att_ms=10, rel_ms=150, makeup_db=0):
    mono = np.max(np.abs(x), axis=0) if x.ndim == 2 else np.abs(x)
    e = env_follow(mono, att_ms, rel_ms)
    edb = 20 * np.log10(e + 1e-9)
    over = np.maximum(edb - thr_db, 0)
    g = db(-over * (1 - 1 / ratio) + makeup_db)
    return x * g


def saturate(x, drive=1.0):
    return np.tanh(x * drive) / np.tanh(drive)


def norm_peak(x, peak_db=-1.0):
    m = np.max(np.abs(x)) + 1e-12
    return x * (db(peak_db) / m)


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x) ** 2)) + 1e-12)
