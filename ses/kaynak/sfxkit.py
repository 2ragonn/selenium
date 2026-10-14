"""Gerçekçi ses efekti tarifleri (fiziksel modelleme + katmanlama).
Her fonksiyon kuru (yankısız) mono ya da stereo numpy dizisi döndürür; yankı veri yolunda eklenir."""
import os, subprocess, tempfile
import numpy as np
import soundfile as sf
import mido
from lib import *

HERE = os.path.dirname(os.path.abspath(__file__))
GS = os.path.join(HERE, 'assets', 'GeneralUser-GS.sf2')


# ============================================================ yardımcılar
def burst(n, rng):
    return rng.standard_normal(n) * decay(n, n / SR * 0.9)


def click(rng, freqs, t60s, amps, n_exc=24, dur=0.12, hp_hz=300):
    """Mekanik tık: kısa gürültü darbesi -> rezonatör bankası."""
    n = T(dur); exc = np.zeros(n); exc[:n_exc] = rng.standard_normal(n_exc) * np.hanning(n_exc * 2)[n_exc:]
    y = resbank(exc, freqs, t60s, amps)
    y += exc * 0.15
    return hp(y, hp_hz)


def place(dst, src, t0, gain=1.0):
    """src (mono/stereo) dizisini dst (2,N) içine t0 (s) anında ekler."""
    s = st(src)
    i0 = T(t0)
    if i0 < 0:
        s = s[:, -i0:]; i0 = 0
    if i0 >= dst.shape[1]:
        return
    n = min(s.shape[1], dst.shape[1] - i0)
    dst[:, i0:i0 + n] += s[:, :n] * gain


def silence(sec):
    return np.zeros((2, T(sec)))


# ============================================================ GS soundfont örnekleri
_gs_cache = {}


def gs_note(bank, prog, note, dur, vel=110, drum=False, tail=2.5, sfont=None):
    """GeneralUser GS'den tek nota/efekt örneği (fluidsynth, yankısız)."""
    key = (bank, prog, note, round(dur, 3), vel, drum, sfont)
    if key in _gs_cache:
        return _gs_cache[key]
    mid = mido.MidiFile(ticks_per_beat=960); tr = mido.MidiTrack(); mid.tracks.append(tr)
    ch = 9 if drum else 0
    tr.append(mido.MetaMessage('set_tempo', tempo=500000))
    if not drum:
        tr.append(mido.Message('control_change', channel=ch, control=0, value=bank))
        tr.append(mido.Message('control_change', channel=ch, control=32, value=0))
    tr.append(mido.Message('program_change', channel=ch, program=prog))
    tr.append(mido.Message('control_change', channel=ch, control=7, value=127))
    tr.append(mido.Message('note_on', channel=ch, note=note, velocity=vel, time=480))
    tr.append(mido.Message('note_off', channel=ch, note=note, velocity=0, time=int(dur * 960 * 2)))
    tr.append(mido.Message('note_off', channel=ch, note=note, velocity=0, time=int(tail * 960 * 2)))
    with tempfile.TemporaryDirectory() as d:
        mp = os.path.join(d, 'a.mid'); wp = os.path.join(d, 'a.wav'); mid.save(mp)
        subprocess.run(['fluidsynth', '-ni', '-q', '-R', '0', '-C', '0', '-g', '1.0', '-r', str(SR), '-F', wp, sfont or GS, mp],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        a, sr = sf.read(wp, always_2d=True)
    a = a.T[:, T(0.5):]
    # sondaki sessizliği kırp
    e = np.max(np.abs(a), axis=0); nz = np.nonzero(e > 1e-4)[0]
    a = a[:, :nz[-1] + 1] if len(nz) else a[:, :1]
    _gs_cache[key] = a
    return a


# ============================================================ GEMİ
def ship_hum(n, rng, base=31.0):
    """Gemi uğultusu yatağı: alt bas kısmi tonlar + gürültü gövde + boğaz rezonansı (sürekli)."""
    t = tt(n)
    drift = smooth_noise(n, 0.05, rng) * 0.6
    y = np.zeros(n)
    for k, (mul, a) in enumerate([(1, 1.0), (1.013, 0.8), (2.0, 0.45), (3.02, 0.25), (4.71, 0.12), (6.05, 0.06)]):
        f = base * mul + drift * mul
        ph = 2 * np.pi * np.cumsum(f) / SR + rng.random() * 6.28
        am = 0.75 + 0.25 * smooth_noise(n, 0.08 + 0.03 * k, rng)
        y += np.sin(ph) * a * am
    rum = lp(brown(n, rng), 90, 2) * 0.9 * (0.7 + 0.3 * smooth_noise(n, 0.1, rng))
    throat = svf(pink(n, rng), 190 + 40 * smooth_noise(n, 0.07, rng), 3.0, 'bp') * 0.35
    air = bp(white(n, rng), 2500, 6000) * 0.012 * (0.5 + 0.5 * smooth_noise(n, 0.15, rng))
    y = saturate(y * 0.5 + rum * 0.5, 1.6) + throat + air
    return widen(y, rng, 0.6, 17)


def ship_horn(dur, rng, f0=45.0):
    """'Derin, uzun, göğsü titreten' uğultu: süzgeçten geçen zengin bas korna (Savaş-of-the-Worlds tarzı)."""
    n = T(dur); t = tt(n)
    e = env(n, [(0, 0), (dur * .18, .7), (dur * .45, 1), (dur * .8, .85), (dur, 0)], 1.4)
    f = f0 * (1 + 0.02 * np.sin(2 * np.pi * 0.11 * t)) * (1 + 0.03 * env(n, [(0, -1), (dur * .3, 0), (dur, 0.2)]))
    ph = 2 * np.pi * np.cumsum(f) / SR
    saw = np.zeros(n)
    for k in range(1, 70):
        if f0 * k > 6000:
            break
        saw += np.sin(k * ph + 0.3 * k) / k
    fc = 90 + 900 * env(n, [(0, 0), (dur * .35, 1), (dur * .6, .55), (dur, .1)], 1.5)
    body = svf(saw, fc, 2.2, 'lp')
    sub = np.sin(ph) * 0.9
    noise = svf(pink(n, rng), fc * 1.4, 1.2, 'bp') * 0.25
    y = saturate((body * 0.7 + sub + noise) * e * 0.9, 1.8)
    return widen(y, rng, 0.5, 23)


def glass_rattle(n, rng, intensity):
    """Camların zangırdaması: düzensiz darbeler -> cam/çerçeve rezonansları."""
    inten = np.broadcast_to(intensity, (n,))
    rate = 18 + 30 * inten
    prob = rate / SR * inten
    imp = (rng.random(n) < prob) * rng.standard_normal(n) * (0.4 + inten)
    glass = resbank(imp, [1180, 1730, 2410, 3360, 4890, 6320, 7900], [0.09, 0.07, 0.06, 0.05, 0.04, 0.03, 0.025], [1, .8, .7, .6, .45, .3, .2])
    frame = resbank(imp, [420, 780, 1350], [0.03, 0.025, 0.02], [1, .7, .5]) * 0.6
    flex = lp(brown(n, rng), 140) * inten * 0.4
    y = hp(glass * 0.6 + frame, 250) + flex
    return widen(y, rng, 0.9, 7)


def sphere_hum(dur, rng, f0=233.0, bright=1.0):
    """Küre: cam armonika benzeri, yavaş vuruşlu inharmonik tınılar + nefes alan hava."""
    n = T(dur); t = tt(n)
    y = np.zeros(n)
    for mul, a in [(1, 1), (1.497, .55), (2.01, .45), (2.76, .25), (3.53, .15), (0.5, .5)]:
        f = f0 * mul * (1 + 0.003 * smooth_noise(n, 0.3, rng))
        ph = 2 * np.pi * np.cumsum(f) / SR + rng.random() * 6
        y += np.sin(ph) * a * (0.6 + 0.4 * smooth_noise(n, 0.4, rng))
    air = svf(white(n, rng), 1800 + 900 * smooth_noise(n, 0.2, rng), 4, 'bp') * 0.18 * bright
    sub = np.sin(2 * np.pi * 55 * t) * 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
    y = (y * 0.22 + air + sub * 0.5)
    return widen(lp(y, 7000), rng, 1.0, 13)


def whoosh(dur, rng, f0=300, f1=3000, p0=-0.7, p1=0.7, peak=0.6, q=1.3, tone=0.0):
    """Hava sesi geçişi: süzgeç tarama + zarf + pan hareketi (+ isteğe bağlı tonal öz)."""
    n = T(dur); x = np.linspace(0, 1, n)
    fc = f0 * (f1 / f0) ** (x ** 1.2)
    e = np.where(x < peak, (x / peak) ** 2.2, ((1 - x) / (1 - peak)) ** 1.6)
    y = svf(pink(n, rng), fc, q, 'bp') * 1.4 + svf(white(n, rng), fc * 2.2, 0.8, 'bp') * 0.25
    if tone > 0:
        ph = 2 * np.pi * np.cumsum(fc * 0.25) / SR
        y += np.sin(ph) * tone * 0.5
    y *= e
    return pan(y, p0 + (p1 - p0) * x)


def swarm(dur, rng, n_src=14, base=210):
    """Küçük gemiler: 'kovanına dönen arılar' — çok sayıda titreşen vızıltı, hareketli."""
    n = T(dur); out = np.zeros((2, n)); t = tt(n)
    for i in range(n_src):
        f = base * (0.75 + 0.6 * rng.random()) * (1 + 0.02 * smooth_noise(n, 0.5 + rng.random(), rng))
        ph = 2 * np.pi * np.cumsum(f) / SR
        buzz = np.zeros(n)
        for k in range(1, 12):
            buzz += np.sin(k * ph) / k ** 1.3
        trem = 0.6 + 0.4 * np.sin(2 * np.pi * (17 + 9 * rng.random()) * t + rng.random() * 6)
        a = np.clip(0.5 + 0.7 * smooth_noise(n, 0.25, rng), 0, 1)
        sig = bp(buzz * trem * a, 150, 3500)
        p = np.clip(smooth_noise(n, 0.15, rng), -1, 1)
        out += pan(sig, p) / n_src * 2.5
    return out


def ship_open(dur, rng):
    """Geminin karnı açılıyor: derin metalik inilti + hidrolik tıslama + yükselen enerji uğultusu."""
    n = T(dur)
    exc = brown(n, rng) * env(n, [(0, 0), (0.4, 1), (dur * .7, .6), (dur, 0)])
    modes = 70 * np.array([1, 1.41, 1.93, 2.47, 3.18, 3.9, 4.77, 5.5, 6.9, 8.3])
    groan = resbank(exc, modes, 2.5, [1, .9, .8, .7, .5, .4, .3, .25, .2, .15]) * 0.03
    groan = varispeed(np.concatenate([groan, np.zeros(T(2))]), np.linspace(1.08, 0.93, n + T(2)))[:n]
    hiss = hp(white(n, rng), 3000) * env(n, [(0, 0), (dur * .2, 0), (dur * .3, .25), (dur * .6, .1), (dur, 0)]) * 0.5
    t = tt(n); f = 55 * (1 + 0.5 * env(n, [(0, 0), (dur, 1)]))
    ph = 2 * np.pi * np.cumsum(f) / SR
    hum = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3.01 * ph)) * env(n, [(0, 0), (dur * .4, .3), (dur, 1)]) * 0.35
    boom = impact(rng, size=1.2)[0]
    y = saturate(groan + hum, 1.3) + hiss
    y[:len(boom)] += boom[:n] * 0.8
    return widen(y, rng, 0.8, 19)


# ============================================================ ELEKTRİK
def power_down(rng, dist=1.0):
    """Lamba/röle sönüşü: şebeke vızıltısı düşüp kesilir + röle 'klak' + uzak trafo patlaması."""
    dur = 1.6; n = T(dur); t = tt(n)
    f = 50 * (1 - 0.35 * env(n, [(0, 0), (0.25, 0), (0.9, 1), (dur, 1)]))
    ph = 2 * np.pi * np.cumsum(f) / SR
    hum = sum(np.sin(k * ph) / k for k in range(1, 9)) * env(n, [(0, .5), (0.2, .5), (0.85, 0), (dur, 0)]) * 0.25
    buzz = bp(np.sign(np.sin(2 * ph)), 200, 3000) * env(n, [(0, .12), (0.22, .12), (0.3, 0)]) * 0.5
    relay = click(rng, [180, 340, 1220, 2600], [0.08, 0.05, 0.03, 0.02], [1, .7, .5, .3], n_exc=60, dur=0.3, hp_hz=60)
    y = hum + buzz
    i = T(0.22); y[i:i + len(relay)] += relay[:n - i] * 1.2
    if dist > 1.5:
        y = lp(y, 2500 / dist)
    return y


def transformer_pop(rng):
    n = T(1.2)
    b = lp(burst(T(0.25), rng), 1800) * 1.0
    y = np.zeros(n); y[:len(b)] += b
    y += click(rng, [90, 160], [0.25, 0.2], [1, .6], n_exc=200, dur=1.2, hp_hz=30) * 0.6
    return y


def lights_on(rng, dist=1.0):
    """Sokak lambaları yanıyor: röle 'tak' + balast vızıltısı yükseliyor."""
    dur = 3.0; n = T(dur); t = tt(n)
    relay = click(rng, [160, 300, 1000, 2400], [0.07, 0.05, 0.03, 0.02], [1, .7, .5, .35], n_exc=50, dur=0.3, hp_hz=60)
    ph = 2 * np.pi * 100 * t
    ballast = sum(np.sin(k * ph + k) / k ** 1.2 for k in range(1, 14)) * env(n, [(0, 0), (0.15, 0), (0.6, .35), (1.5, .18), (dur, .12)])
    tick = bp(white(n, rng), 3000, 9000) * (rng.random(n) < 0.002) * 2
    y = bp(ballast, 90, 4000) * 0.4 + tick * env(n, [(0, 0), (0.2, 1), (1, 0)])
    y[:len(relay)] += relay * 1.0
    return lp(y, 9000 / max(1, dist))


def phone_vibrate(dur, rng, on=0.42, off=0.28, surface='table'):
    """Telefon titreşimi: eksantrik motor + yüzey takırtısı (masa) ya da cep boğukluğu."""
    n = T(dur); t = tt(n)
    gate = ((t % (on + off)) < on).astype(float)
    gate = lp(gate, 40)
    f = 168 * (1 + 0.015 * smooth_noise(n, 3, rng))
    ph = 2 * np.pi * np.cumsum(f) / SR
    motor = (np.sin(ph) + 0.45 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)) * gate
    if surface == 'table':
        rattle = resbank((rng.random(n) < 0.01 * gate) * rng.standard_normal(n), [820, 1650, 2900, 4100], [0.02, 0.015, 0.012, 0.01], [1, .7, .5, .3])
        y = bp(motor, 90, 2500) * 0.6 + rattle * 0.5 + bp(motor * rng.standard_normal(n) * 0.2, 1500, 5000) * 0.3
    else:
        y = lp(motor, 450) * 0.8
    return y


def car_alarm(dur, rng):
    """Klasik araba alarmı: yükselip alçalan siren + hızlı vuvuu + kesik bip döngüsü (uzakta)."""
    n = T(dur); t = tt(n)
    seg = (t % 9.0)
    f = np.where(seg < 3, 750 + 650 * (0.5 + 0.5 * np.sin(2 * np.pi * 1.6 * t)),
                 np.where(seg < 6, 600 + 900 * ((t * 5) % 1), 1050 * (((t * 6) % 1) < .5) + 800 * (((t * 6) % 1) >= .5)))
    ph = 2 * np.pi * np.cumsum(f) / SR
    tone = np.sign(np.sin(ph)) * 0.6 + np.sin(ph) * 0.4
    y = bp(tone, 400, 5000)
    y = peq(y, 2200, 1.5, 5)
    return saturate(y * env(n, [(0, 0), (0.05, 1), (dur - 0.3, 1), (dur, 0)]), 1.5) * 0.5


def car_horn(dur, rng, f=420):
    n = T(dur); t = tt(n)
    v = 1 + 0.004 * np.sin(2 * np.pi * 5 * t)
    ph1 = 2 * np.pi * np.cumsum(np.full(n, f) * v) / SR; ph2 = 2 * np.pi * np.cumsum(np.full(n, f * 1.26) * v) / SR
    y = np.tanh(3 * np.sin(ph1)) + np.tanh(3 * np.sin(ph2))
    y = bp(y, 250, 4000); y = peq(y, 1800, 2, 4)
    return y * env(n, [(0, 0), (0.02, 1), (dur - 0.05, 1), (dur, 0)]) * 0.4


# ============================================================ RADYO
def radio_filter(x, tight=1.0):
    """Lambalı radyo hoparlörü: bant sınırlı, 1 kHz civarı burunlu, hafif doygun."""
    y = bp(x, 180 * tight, 4000 / tight, 4)
    y = peq(y, 1100, 1.2, 4); y = peq(y, 350, 1.0, 2)
    return lp(saturate(y * 1.2, 1.4), 4800 / tight, 4)


def radio_static(n, rng, level=1.0, crackle=1.0, hum=1.0, whistle=0.4):
    """AM/kısa dalga cızırtısı: süzülmüş tıslama + çıtırtılar + yavaş sönümlenme (QSB) + heterodin ıslığı + lamba vızıltısı."""
    t = tt(n)
    qsb = 0.65 + 0.35 * smooth_noise(n, 0.35, rng)
    hiss = white(n, rng) * qsb * 0.35 * level
    pr = (rng.random(n) < 14 / SR * crackle)
    imp = pr * rng.standard_normal(n) * (rng.pareto(2.2, n) + 0.3)
    crk = bp(imp, 600, 6000) * 0.9
    pops = lp((rng.random(n) < 1.5 / SR * crackle) * rng.standard_normal(n) * 3, 900)
    f = 900 + 500 * smooth_noise(n, 0.05, rng)
    whis = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.04 * whistle * np.clip(smooth_noise(n, 0.1, rng), 0, 1)
    hm = (np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 150 * t) + 0.25 * np.sin(2 * np.pi * 200 * t)) * 0.025 * hum
    return radio_filter(hiss + crk + pops + whis, 1.0) + hm


def cw_signal(times, rng, f=620, unit=None, drift=0.0, fade=0.0, n=None, clean=False):
    """Mors ses tonu: times = [(baş, süre), ...] — 5 ms yumuşak kenarlı ton, radyo süzgeci."""
    end = max(a + d for a, d in times) + 0.5
    n = n or T(end); t = tt(n)
    gate = np.zeros(n)
    for a, d in times:
        i, j = T(a), T(a + d)
        gate[i:j] = 1
    k = T(0.005)
    w = np.hanning(2 * k); w /= w.sum()
    gate = np.convolve(gate, w, 'same')
    ff = f * (1 + drift * smooth_noise(n, 0.2, rng))
    ph = 2 * np.pi * np.cumsum(ff) / SR
    tone = np.sin(ph) + (0.0 if clean else 0.08 * np.sin(2 * ph))
    qsb = 1 - fade * (0.5 + 0.5 * smooth_noise(n, 0.3, rng))
    return tone * gate * qsb


def morse_key(rng, down=True):
    """Pirinç Mors anahtarı: kontak 'tık' + ahşap taban 'tok'."""
    if down:
        a = click(rng, [2150, 3720, 5900, 8300], [0.035, 0.03, 0.02, 0.015], [1, .8, .5, .3], n_exc=18, dur=0.12, hp_hz=600)
        b = click(rng, [190, 410, 820], [0.05, 0.03, 0.02], [1, .5, .3], n_exc=40, dur=0.12, hp_hz=80) * 0.8
        return a + b
    return click(rng, [2600, 4300, 6800], [0.025, 0.02, 0.015], [1, .6, .3], n_exc=12, dur=0.08, hp_hz=900) * 0.55


_XM = {'Q': '--.-', 'W': '.--', 'X': '-..-', '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.'}


def morse_code_times(text, t0, unit=0.075, jitter=0.0, rng=None, dash=3.0):
    """Metni Mors zamanlamasına çevirir: [(baş, süre)]. jitter: el ritmi sapması."""
    from tl import MORSE
    out = []; t = t0
    for ch in text:
        if ch == ' ':
            t += unit * 4; continue
        code = MORSE.get(ch) or _XM.get(ch, '')
        for sym in code:
            d = unit * (dash if sym == '-' else 1.0)
            if jitter and rng is not None:
                d *= 1 + jitter * (rng.random() - 0.5) * 2
            out.append((t, d)); t += d + unit * (1 + (jitter * (rng.random() - .3) if jitter and rng is not None else 0))
        t += unit * 2
    return out, t


def babble_voice(dur, rng, f0=120, rate=4.5, gender='m', radio=True):
    """Anlaşılmaz konuşma (yayın/telsiz sesi): gırtlak darbeleri -> değişen formantlar, hece zarfı, sürtünmeli ünsüzler."""
    n = T(dur); t = tt(n)
    VOW = {'a': (750, 1250, 2550), 'e': (500, 1800, 2600), 'i': (330, 2200, 3000), 'o': (500, 900, 2500), 'u': (350, 800, 2300), 'ı': (380, 1500, 2500)}
    keys = list(VOW)
    syl = []; x = 0.0
    while x < dur:
        L = (1 / rate) * (0.6 + 0.8 * rng.random()); syl.append((x, L, keys[rng.integers(len(keys))])); x += L
        if rng.random() < 0.12:
            x += 0.15 + 0.3 * rng.random()
    F = np.zeros((3, n)); A = np.zeros(n); FR = np.zeros(n)
    for s, L, v in syl:
        i, j = T(s), min(n, T(s + L))
        for k in range(3):
            F[k, i:j] = VOW[v][k] * (1 + 0.06 * (rng.random() - .5))
        m = j - i
        if m > 4:
            e = np.sin(np.linspace(0, np.pi, m)) ** 0.7
            A[i:j] = e * (0.6 + 0.4 * rng.random())
            if rng.random() < 0.55:
                c = min(m, T(0.05)); FR[i:i + c] = np.linspace(1, 0, c) * (0.5 + rng.random())
    for k in range(3):
        F[k] = lp(F[k], 18)
    A = lp(A, 25); A[A < 0] = 0
    contour = f0 * (1 + 0.12 * smooth_noise(n, 0.6, rng) - 0.08 * ((t % 2.5) / 2.5))
    ph = np.cumsum(contour) / SR
    pulses = np.diff(np.floor(ph), prepend=0) > 0
    glott = lp(pulses.astype(float) * 4.0, 2500, 1) + 0.04 * white(n, rng)
    y = np.zeros(n)
    for k, (bw, g) in enumerate([(80, 1.0), (110, 0.7), (160, 0.35)]):
        y += svf(glott, F[k], F[k].mean() / bw if F[k].mean() > 0 else 5, 'bp') * g
    y *= A
    y += bp(white(n, rng), 3500, 8000) * FR * 0.25
    y = hp(y, 120)
    y /= np.max(np.abs(y)) + 1e-9
    return radio_filter(y, 1.15) if radio else y


def whisper(text_syls, rng, rate=3.2):
    """Fısıltı ('du-yan var mı'): gürültü -> sesli harf formantları, ünsüz patlamaları."""
    VOW = {'a': (750, 1250, 2550), 'u': (350, 800, 2300), 'ı': (380, 1500, 2500), 'e': (500, 1800, 2600), 'i': (330, 2200, 3000), 'o': (500, 900, 2500)}
    out = []
    for syl in text_syls:
        v = [c for c in syl if c in VOW][-1]
        L = 1 / rate * (1.3 if syl.endswith('ı') or syl.endswith('an') else 1.0)
        n = T(L); e = np.sin(np.linspace(0, np.pi, n)) ** 0.6
        src = white(n, rng)
        y = sum(svf(src, np.full(n, VOW[v][k]), VOW[v][k] / bw, 'bp') * g for k, (bw, g) in enumerate([(120, 1), (160, .8), (220, .5)]))
        cons = bp(white(n, rng), 2500, 7000) * np.exp(-np.arange(n) / T(0.03)) * (0.8 if syl[0] in 'dvym' else 0.4)
        out.append((y * e + cons) * (1 if syl[0] != 'm' else 0.7))
        out.append(np.zeros(T(0.03)))
    y = np.concatenate(out)
    return hp(y, 400) / (np.max(np.abs(y)) + 1e-9)


# ============================================================ MEKANİK / KÜÇÜK
def watch_ticks(dur, rng, rate=5.0, gain=1.0):
    """Kurmalı cep saati (18000 vuruş/saat): tik-tak çifti, küçük metal rezonanslar."""
    n = T(dur); y = np.zeros(n)
    tk = click(rng, [4200, 6900, 9300, 2950, 11500], [0.018, 0.015, 0.012, 0.02, 0.01], [1, .8, .6, .5, .3], n_exc=6, dur=0.05, hp_hz=1500)
    tc = click(rng, [3600, 5800, 8100, 2400], [0.02, 0.016, 0.012, 0.022], [1, .7, .5, .5], n_exc=6, dur=0.05, hp_hz=1300) * 0.75
    k = 0; x = 0.0
    while x < dur - 0.05:
        s = tk if k % 2 == 0 else tc
        i = T(x + rng.normal(0, 0.0015)); m = min(len(s), n - i)
        if i >= 0 and m > 0:
            y[i:i + m] += s[:m] * (0.85 + 0.3 * rng.random())
        x += 1 / rate; k += 1
    return y * gain


def watch_wind(rng, clicks=14):
    """Saat kurma: hızlı cırcır tıkları."""
    y = np.zeros(T(0.08 * clicks + 0.2))
    for i in range(clicks):
        c = click(rng, [5200, 7600, 3300], [0.012, 0.01, 0.015], [1, .6, .4], n_exc=4, dur=0.04, hp_hz=2000)
        p = T(i * 0.07 + rng.random() * 0.01)
        y[p:p + len(c)] += c * (0.6 + 0.4 * rng.random())
    return y


def heartbeat(dur, rng, bpm0=72, bpm1=110, gain=1.0):
    """Kalp atışı (lub-dub), giderek hızlanan."""
    n = T(dur); y = np.zeros(n); x = 0.2
    while x < dur - 0.4:
        bpm = bpm0 + (bpm1 - bpm0) * (x / dur)
        for off, f, a in [(0, 52, 1.0), (0.28 * 60 / max(bpm, 60) * 1.0, 64, 0.7)]:
            L = T(0.16); tt_ = np.arange(L) / SR
            fr = f * np.exp(-tt_ * 6)
            s = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-tt_ * 22) * a
            s += lp(rng.standard_normal(L), 160) * np.exp(-tt_ * 40) * 0.3 * a
            i = T(x + off); m = min(L, n - i)
            if m > 0:
                y[i:i + m] += s[:m]
        x += 60 / bpm
    return saturate(y * 1.5 * gain, 1.3)


def footstep(rng, surface='concrete', w=1.0):
    """Tek adım: topuk tok sesi + taban sürtünmesi + yüzey rezonansı."""
    n = T(0.25); y = np.zeros(n)
    heel = lp(burst(T(0.04), rng), 900) * 1.0
    thud = click(rng, [95, 180, 320], [0.05, 0.04, 0.03], [1, .6, .4], n_exc=80, dur=0.2, hp_hz=40)
    tip = bp(burst(T(0.025), rng), 1500, 6000) * 0.35
    y[:len(heel)] += heel * w; y[:len(thud)] += thud * 0.9 * w
    k = T(0.045 + 0.02 * rng.random()); y[k:k + len(tip)] += tip * w
    if surface == 'stairs':
        y += click(rng, [140, 260, 610], [0.07, 0.05, 0.03], [1, .7, .4], n_exc=60, dur=0.25, hp_hz=50) * 0.5 * w
    if rng.random() < 0.35:
        sc = bp(white(T(0.08), rng), 1200, 5000) * np.hanning(T(0.08)) * 0.18
        k = T(0.06); y[k:k + len(sc)] += sc
    return y


def footsteps(dur, rng, rate=2.2, surface='concrete', run=False):
    n = T(dur); y = np.zeros(n); x = 0.05
    while x < dur - 0.25:
        s = footstep(rng, surface, 1.3 if run else 1.0)
        i = T(x); m = min(len(s), n - i); y[i:i + m] += s[:m] * (0.8 + 0.4 * rng.random())
        x += 1 / rate * (1 + 0.08 * rng.normal())
    return y


def door_slam(rng):
    """Çatı kapısı çarpması: gövde güm + panel rezonansı + kilit dili + tıkırtı."""
    n = T(1.6); y = np.zeros(n)
    tt_ = tt(T(0.35)); sub = np.sin(2 * np.pi * np.cumsum(72 * np.exp(-tt_ * 3)) / SR) * np.exp(-tt_ * 12)
    y[:len(sub)] += sub * 1.2
    exc = lp(burst(T(0.03), rng), 2500)
    panel = resbank(np.concatenate([exc, np.zeros(n - len(exc))]), [112, 175, 263, 418, 640, 910], [0.35, 0.3, 0.25, 0.2, 0.12, 0.08], [1, .8, .7, .5, .35, .25]) * 0.12
    y += panel
    latch = click(rng, [2300, 3900, 6100], [0.04, 0.03, 0.02], [1, .7, .4], n_exc=20, dur=0.15, hp_hz=800)
    k = T(0.012); y[k:k + len(latch)] += latch * 0.7
    for d in [0.11, 0.19, 0.24]:
        c = click(rng, [1500, 2700], [0.03, 0.02], [1, .6], n_exc=10, dur=0.08, hp_hz=700) * 0.25
        k = T(d); y[k:k + len(c)] += c
    return saturate(y, 1.2)


def door_creak(dur, rng):
    """Menteşe gıcırtısı: yapış-kay sürtünme darbeleri -> ahşap rezonansları."""
    n = T(dur); x = np.linspace(0, 1, n)
    rate = 40 + 90 * np.abs(smooth_noise(n, 1.2, rng))
    ph = np.cumsum(rate) / SR
    pulses = (np.diff(np.floor(ph), prepend=0) > 0).astype(float) * (0.6 + 0.4 * rng.random(n))
    y = resbank(pulses, [430, 760, 1180, 1720, 2600], [0.04, 0.035, 0.03, 0.02, 0.015], [1, .9, .7, .5, .3])
    return hp(y, 200) * np.sin(np.pi * x) ** 0.5


def switch_click(rng):
    a = click(rng, [3100, 5200, 7400], [0.02, 0.015, 0.01], [1, .7, .4], n_exc=8, dur=0.06, hp_hz=1000)
    b = click(rng, [1800, 2900], [0.02, 0.015], [1, .5], n_exc=12, dur=0.06, hp_hz=600) * 0.5
    return a + b


def knob_click(rng):
    return click(rng, [1400, 2600, 4100], [0.03, 0.025, 0.015], [1, .7, .4], n_exc=14, dur=0.1, hp_hz=500)


def paper(dur, rng):
    """Kâğıt hışırtısı: hızlı rastgele kırışma darbeleri + kayma."""
    n = T(dur)
    cr = (rng.random(n) < 0.004) * rng.standard_normal(n) * (rng.pareto(2.5, n) + 0.2)
    y = bp(cr, 1500, 9000) * 0.8 + bp(white(n, rng), 2000, 7000) * 0.06 * np.abs(smooth_noise(n, 6, rng))
    return y * env(n, [(0, 0), (0.05, 1), (dur - 0.1, 1), (dur, 0)])


def wood_thump(rng, f=150):
    return click(rng, [f, f * 1.7, f * 2.9, f * 4.6], [0.12, 0.09, 0.06, 0.04], [1, .7, .45, .3], n_exc=120, dur=0.5, hp_hz=50)


def shop_bell(rng):
    p = np.array([1, 2.0, 2.76, 3.0, 4.2, 5.4]) * 1760
    y = click(rng, p, [0.9, 0.7, 0.6, 0.5, 0.35, 0.25], [1, .4, .6, .3, .25, .15], n_exc=6, dur=1.5, hp_hz=500)
    y2 = click(rng, p * 1.004, [0.9, 0.7, 0.6, 0.5, 0.35, 0.25], [1, .4, .6, .3, .25, .15], n_exc=6, dur=1.5, hp_hz=500)
    out = np.zeros(T(1.8)); out[:len(y)] += y; out[T(0.12):T(0.12) + len(y2)] += y2 * 0.7
    return out


def glass_clink(rng, small=True):
    p = np.array([2950, 6120, 8900, 4400]) if small else np.array([1700, 3900, 6200])
    return click(rng, p * (0.97 + 0.06 * rng.random()), [0.35, 0.25, 0.15, 0.2][:len(p)], [1, .6, .4, .3][:len(p)], n_exc=5, dur=0.8, hp_hz=800)


def coins(rng, k=5):
    y = np.zeros(T(1.0))
    for i in range(k):
        c = click(rng, np.array([4300, 7100, 9800]) * (0.9 + 0.2 * rng.random()), [0.15, 0.1, 0.07], [1, .7, .4], n_exc=4, dur=0.3, hp_hz=1500)
        p = T(0.05 + i * 0.07 * (1 + rng.random())); m = min(len(c), len(y) - p)
        if m > 0:
            y[p:p + m] += c[:m] * (0.5 + 0.5 * rng.random())
    return y


def ladle_pot(rng):
    p = np.array([430, 1105, 2140, 3390, 4800])
    return click(rng, p, [1.2, 0.9, 0.6, 0.4, 0.3], [1, .7, .5, .3, .2], n_exc=10, dur=2.0, hp_hz=200) * 0.6


def curtain(dur, rng):
    """Perde açma: halkaların rayda metal sürtünmesi + kumaş hışırtısı."""
    n = T(dur); y = np.zeros(n)
    x = 0.0
    while x < dur - 0.1:
        c = click(rng, np.array([2400, 3800, 5600]) * (0.9 + 0.2 * rng.random()), [0.05, 0.04, 0.03], [1, .6, .4], n_exc=10, dur=0.1, hp_hz=900)
        i = T(x); m = min(len(c), n - i); y[i:i + m] += c[:m] * 0.5
        x += 0.05 + 0.08 * rng.random()
    scr = bp(white(n, rng), 1800, 6000) * 0.08 * np.abs(smooth_noise(n, 8, rng))
    cloth = bp(white(n, rng), 300, 3000) * 0.12 * (0.5 + 0.5 * np.abs(smooth_noise(n, 3, rng)))
    return (y + scr + cloth) * env(n, [(0, 0), (0.08, 1), (dur - 0.2, 1), (dur, 0)])


def breath_in(rng, dur=0.6):
    n = T(dur)
    y = svf(white(n, rng), np.linspace(1400, 2600, n), 1.5, 'bp') * env(n, [(0, 0), (dur * .3, 1), (dur * .85, .7), (dur, 0)])
    return y * 0.5


def puff(rng):
    n = T(0.35)
    return lp(white(n, rng), 1200) * env(n, [(0, 0), (0.03, 1), (0.35, 0)], 1.5) * 0.6


def tap_dry(rng):
    """Kuru musluk: vana gıcırtısı + hava guruldaması + tek damla."""
    sq = door_creak(0.35, rng) * 0.6
    n = T(1.6); g = np.zeros(n)
    for i in range(9):
        L = T(0.04 + 0.04 * rng.random()); f = np.linspace(300 + 500 * rng.random(), 900 + 600 * rng.random(), L)
        b = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.linspace(0, np.pi, L))
        p = T(0.4 + i * 0.09 + 0.05 * rng.random()); g[p:p + L] += b * 0.3
    drop = np.zeros(T(0.2)); L = T(0.03); drop[:L] = np.sin(2 * np.pi * np.cumsum(np.linspace(900, 2300, L)) / SR) * np.hanning(L)
    out = np.zeros(n); out[:len(sq)] += sq; out += g; out[T(1.35):T(1.35) + len(drop)] += drop[:n - T(1.35)] * 0.6
    return out


# ============================================================ DOĞA / ORTAM
def wind(n, rng, strength=1.0, gust=1.0, whistle=0.0, lo=180, hi=900):
    gm = np.clip(0.55 + 0.45 * smooth_noise(n, 0.12, rng) * gust, 0.08, 1.4)
    fc = lo + (hi - lo) * np.clip(0.5 + 0.5 * smooth_noise(n, 0.09, rng), 0, 1)
    y = svf(pink(n, rng), fc, 0.9, 'lp') * gm + svf(pink(n, rng), fc * 2.2, 2.5, 'bp') * gm * 0.25
    if whistle > 0:
        wf = 650 + 450 * np.clip(0.5 + 0.5 * smooth_noise(n, 0.07, rng), 0, 1)
        y += svf(white(n, rng), wf, 18, 'bp') * gm ** 2 * whistle * 0.9
    return widen(y * strength * 0.5, rng, 1.0, 9)


def fire(n, rng, size=1.0):
    roar = svf(brown(n, rng), 300 + 150 * smooth_noise(n, 0.5, rng), 0.8, 'lp') * 0.35 * (0.7 + 0.3 * smooth_noise(n, 0.8, rng))
    cr = (rng.random(n) < 22 / SR) * rng.standard_normal(n) * (rng.pareto(2.0, n) + 0.3)
    crk = svf(cr, 1500 + 3500 * np.abs(smooth_noise(n, 5, rng)), 2.0, 'bp') * 0.8
    pops = lp((rng.random(n) < 1.2 / SR) * rng.standard_normal(n) * 4, 2200)
    hiss = hp(white(n, rng), 4000) * 0.01
    return widen((roar + crk + pops * 0.6 + hiss) * size, rng, 0.8, 5)


def rain(n, rng, intensity=1.0):
    drops = (rng.random(n) < 900 / SR * intensity) * rng.standard_normal(n) * (rng.pareto(3, n) + 0.2)
    y = svf(drops, 2500 + 2500 * np.abs(smooth_noise(n, 20, rng)), 1.2, 'bp') * 0.35
    y += hp(pink(n, rng), 500) * 0.1 * intensity
    heavy = (rng.random(n) < 25 / SR * intensity) * rng.standard_normal(n)
    y += resbank(heavy, [900, 1700, 3100], [0.02, 0.015, 0.01], [1, .6, .4]) * 0.4
    return widen(y, rng, 1.0, 7)


def bucket_drips(dur, rng, rate=3.0):
    n = T(dur); y = np.zeros(n); x = 0.1
    while x < dur - 0.1:
        L = T(0.03 + 0.03 * rng.random()); f = np.linspace(700 + 400 * rng.random(), 1800 + 1500 * rng.random(), L)
        b = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-np.arange(L) / (L / 3))
        i = T(x); y[i:i + L] += b * (0.3 + 0.5 * rng.random())
        x += rng.exponential(1 / rate)
    return y


def city_hush(n, rng, traffic=1.0):
    """Uzak şehir uğultusu (trafik/hava): alçak geçirgen kahverengi gürültü, nefes alan."""
    y = lp(brown(n, rng), 260) * 0.6 + bp(pink(n, rng), 300, 1500) * 0.08
    y *= (0.75 + 0.25 * smooth_noise(n, 0.05, rng)) * traffic
    return widen(y, rng, 1.0, 21)


def room_tone(n, rng, level=1.0):
    y = lp(pink(n, rng), 900) * 0.05 + lp(brown(n, rng), 120) * 0.08
    return widen(y * level, rng, 0.6, 5)


# ============================================================ KALABALIK
def walla(dur, rng, n_voices=24, f_lo=95, f_hi=230, rate=4.0, excite=0.0):
    """Kalabalık mırıltısı: çok sayıda bağımsız 'konuşan' sentetik ses (anlaşılmaz), farklı mesafe/pan."""
    n = T(dur); out = np.zeros((2, n))
    for i in range(n_voices):
        g = 'f' if rng.random() < 0.45 else 'm'
        f0 = (f_lo + (f_hi - f_lo) * rng.random()) * (1.6 if g == 'f' else 1.0) * (1 + 0.5 * excite)
        L = dur * (0.5 + 0.5 * rng.random())
        v = babble_voice(L, rng, f0=f0, rate=rate * (0.8 + 0.5 * rng.random()) * (1 + 0.3 * excite), radio=False)
        v = lp(v, 2500 + 2500 * rng.random())
        d = 0.4 + rng.random() * 0.6
        s = T(rng.random() * max(0.01, dur - L))
        seg = pan(v * d, rng.uniform(-0.9, 0.9))
        m = min(seg.shape[1], n - s); out[:, s:s + m] += seg[:, :m]
    return out / np.sqrt(n_voices) * 0.9


def claps(dur, rng, density=60):
    """Alkış: binlerce bağımsız el çırpması (her biri kısa süzülmüş gürültü)."""
    n = T(dur); out = np.zeros((2, n))
    m = int(density * dur)
    for _ in range(m):
        L = T(0.006 + 0.012 * rng.random()); fc = 900 + 1800 * rng.random()
        c = bp(rng.standard_normal(L + 64), fc * 0.6, fc * 1.8)[:L] * np.exp(-np.arange(L) / (L / 4))
        i = T(rng.random() * dur); k = min(L, n - i)
        if k > 0:
            out[:, i:i + k] += pan(c[:k] * (0.3 + 0.7 * rng.random()), rng.uniform(-0.9, 0.9))
    return out * 0.6


def cheer(dur, rng, n_voices=30):
    """Tezahürat: açık ünlülerde sürdürülen bağırışlar + ıslıklar + alkış."""
    n = T(dur); out = np.zeros((2, n)); t = tt(n)
    for i in range(n_voices):
        L = 0.6 + 1.8 * rng.random(); m = T(L)
        f0 = (180 + 220 * rng.random()) * (1.5 if rng.random() < .4 else 1)
        tl = np.arange(m) / SR
        f = f0 * (1 + 0.15 * np.sin(np.pi * tl / L) + 0.01 * np.sin(2 * np.pi * 5.5 * tl))
        ph = np.cumsum(f) / SR
        glott = lp((np.diff(np.floor(ph), prepend=0) > 0).astype(float) * 4, 3000, 1)
        vw = [(800, 1300, 2600), (650, 1800, 2700), (700, 1150, 2500)][rng.integers(3)]
        y = sum(svf(glott, np.full(m, vw[k]), vw[k] / bw, 'bp') * g for k, (bw, g) in enumerate([(90, 1), (120, .7), (170, .4)]))
        y *= np.sin(np.linspace(0, np.pi, m)) ** 0.5
        y = lp(y, 3000 + 2000 * rng.random())
        i = T(rng.random() * max(0.01, dur - L)); k = min(m, n - i)
        out[:, i:i + k] += pan(y[:k] * (0.4 + 0.6 * rng.random()), rng.uniform(-0.9, 0.9))
    for i in range(int(dur / 2.5)):
        L = T(0.6 + 0.5 * rng.random()); tl = np.arange(L) / SR
        f = 1800 + 900 * np.sin(np.pi * tl / tl[-1]) + 40 * np.sin(2 * np.pi * 7 * tl)
        w_ = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.linspace(0, np.pi, L)) * 0.12
        p = T(rng.random() * (dur - 1)); out[:, p:p + L] += pan(w_ + bp(white(L, rng), 1500, 4000) * 0.01, rng.uniform(-.8, .8))
    out /= np.sqrt(n_voices)
    out += claps(dur, rng, 45) * 0.7
    return out


# ============================================================ ARAÇLAR
def engine(dur, rng, rpm_hz=32.0, kind='diesel', load=0.6):
    """Dizel kamyon / traktör motoru: ateşleme darbeleri + mekanik takırtı + egzoz gürültüsü."""
    n = T(dur); t = tt(n)
    f = rpm_hz * (1 + 0.04 * smooth_noise(n, 0.4, rng))
    ph = np.cumsum(f) / SR
    fire_ = (np.diff(np.floor(ph), prepend=0) > 0).astype(float) * (0.7 + 0.6 * rng.random(n))
    body = resbank(fire_, [rpm_hz * 2.1, 120, 230, 410], [0.08, 0.06, 0.05, 0.03], [1, .7, .5, .3]) * 0.2
    exhaust = lp(fire_ * rng.standard_normal(n) * 3 + fire_ * 2, 600) * 0.5
    clatter = bp((np.diff(np.floor(ph * (2 if kind == 'diesel' else 1)), prepend=0) > 0) * rng.standard_normal(n), 1200, 4000) * 0.25 * load
    y = saturate(body + exhaust + clatter, 1.5)
    return y


def jet_pass(dur, rng, peak=0.5, whine=5200):
    """Savaş uçağı geçişi: kükreme + türbin ıslığı, Doppler ve mesafe süzgeci."""
    n = T(dur); x = np.linspace(0, 1, n)
    dist = 0.15 + np.abs(x - peak) * 2.2
    roar = pink(n, rng) * 0.8 + brown(n, rng) * 0.5
    fc = 400 + 5000 / (1 + dist * 4)
    y = svf(roar, fc, 0.7, 'lp')
    v = 1 + 0.25 * np.tanh((peak - x) * 6)
    wh = np.sin(2 * np.pi * np.cumsum(whine * v) / SR) * 0.08 / (1 + dist * 3)
    a = 1 / (0.15 + dist ** 1.6)
    out = (y + wh) * a / a.max()
    return pan(out, np.clip((x - peak) * 2.5, -1, 1))


def uav(dur, rng, f=96):
    """İHA pervane motoru: 'gökte çim biçme makinesi' vızıltısı."""
    n = T(dur); t = tt(n)
    ff = f * (1 + 0.01 * smooth_noise(n, 0.3, rng))
    ph = 2 * np.pi * np.cumsum(ff) / SR
    y = sum(np.sin(k * ph + rng.random()) / k ** 0.9 for k in range(1, 30))
    y = bp(y, 80, 3500) * (0.8 + 0.2 * np.sin(2 * np.pi * 7.3 * t))
    y += bp(pink(n, rng), 200, 2000) * 0.15
    return y * 0.3


def falling_whistle(dur, rng, f0=1600, f1=500):
    n = T(dur); x = np.linspace(0, 1, n)
    fc = f0 * (f1 / f0) ** x
    y = svf(white(n, rng), fc, 14, 'bp') * 1.5 + np.sin(2 * np.pi * np.cumsum(fc) / SR) * 0.05
    return y * env(n, [(0, 0), (dur * .2, .6), (dur * .9, 1), (dur, 0)])


def sheep(rng, f0=300):
    L = T(0.75); tl = np.arange(L) / SR
    f = f0 * (1 + 0.06 * np.sin(2 * np.pi * 7 * tl) + 0.04 * np.sin(np.pi * tl / tl[-1]))
    ph = np.cumsum(f) / SR
    glott = lp((np.diff(np.floor(ph), prepend=0) > 0).astype(float) * 4, 3500, 1)
    F = [(np.linspace(450, 750, L), 90), (np.linspace(1900, 1300, L), 120), (np.full(L, 2700), 180)]
    y = sum(svf(glott, Fk, Fk.mean() / bw, 'bp') * g for (Fk, bw), g in zip(F, [1, .7, .4]))
    am = 0.7 + 0.3 * np.sin(2 * np.pi * 7 * tl)
    return hp(y * am * np.sin(np.linspace(0, np.pi, L)) ** 0.4, 250) * 0.6


# ============================================================ PATLAMALAR / SAVAŞ
def impact(rng, size=1.0):
    """Sinematik darbe (gün başlıkları): alt bas düşüşü + gövde + kısa metalik parıltı."""
    dur = 2.6 * size; n = T(dur); tl = tt(n)
    f = 58 * np.exp(-tl * 1.1) + 26
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tl * 1.6 / size)
    body = lp(white(n, rng), 700) * np.exp(-tl * 9) * 0.8
    crack = hp(white(T(0.02), rng), 1500) * np.hanning(T(0.04))[T(0.02):] * 0.5
    y = sub * 1.1 + body
    y[:len(crack)] += crack
    shimmer = click(rng, np.array([1210, 1830, 2650, 3410]) * 0.9, [1.4, 1.1, 0.8, 0.6], [1, .7, .5, .3], n_exc=20, dur=dur, hp_hz=600) * 0.08
    y[:len(shimmer)] += shimmer[:n]
    return pan(saturate(y, 1.4), 0)


def explosion(rng, size=1.0, dist=1.0):
    """Patlama: çatlak + gövde + alt bas darbe + enkaz çıtırtısı; mesafeyle süzgeç."""
    dur = 3.5 * size; n = T(dur); tl = tt(n)
    crack = hp(white(T(0.012), rng), 800) * 1.5
    body = svf(white(n, rng) * 0.6 + brown(n, rng) * 0.6, 2500 * np.exp(-tl * 2.5) + 150, 0.8, 'lp') * np.exp(-tl * 2.2 / size)
    f = 45 * np.exp(-tl * 2) + 22
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tl * 2.8 / size) * 1.3
    deb = (rng.random(n) < 40 / SR) * rng.standard_normal(n) * np.exp(-tl * 1.5) * (rng.pareto(2.5, n) + .2)
    debris = bp(deb, 1500, 7000) * 0.5
    y = body + sub + debris
    y[:len(crack)] += crack
    y = saturate(y * 1.1, 1.6)
    if dist > 1:
        y = lp(y, 6000 / dist ** 1.2)
        y = np.concatenate([np.zeros(T((dist - 1) * 0.08)), y])
    return widen(y, rng, 0.9, 12)


def cannon(rng, dist=1.0, cal=1.0):
    """Uçaksavar topu atışı: keskin rapor + gövde + mekanik geri tepme/kovan + uzak yankı (yankı veri yolunda)."""
    n = T(1.6); tl = tt(n)
    rep = hp(white(T(0.006), rng), 400) * 2.0
    body = svf(white(n, rng), 1800 * np.exp(-tl * 18) + 120, 0.9, 'lp') * np.exp(-tl * 14) * 1.2
    f = 70 * np.exp(-tl * 9) + 35
    thump = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tl * 10) * 1.3 * cal
    y = body + thump
    y[:len(rep)] += rep
    mech = click(rng, [620, 1380, 2470, 3900], [0.09, 0.07, 0.05, 0.03], [1, .7, .5, .3], n_exc=30, dur=0.25, hp_hz=300) * 0.25
    k = T(0.09); y[k:k + len(mech)] += mech
    casing = click(rng, [2900, 4700, 7100], [0.12, 0.08, 0.05], [1, .6, .3], n_exc=6, dur=0.3, hp_hz=1500) * 0.12
    k = T(0.35 + 0.1 * rng.random()); y[k:k + len(casing)] += casing
    y = saturate(y, 1.8)
    if dist > 1:
        y = lp(y, 7000 / dist)
        y = np.concatenate([np.zeros(T((dist - 1) * 0.12)), y])
    return y


def shell_whistle(dur, rng, up=True):
    n = T(dur); x = np.linspace(0, 1, n)
    fc = (900 * (3.0 ** x)) if up else (2600 * (0.35 ** x))
    y = svf(white(n, rng), fc, 9, 'bp') * 1.2
    return y * env(n, [(0, 0), (0.05, 1), (dur * .7, .6), (dur, 0)]) * 0.5


def flak(rng, dist=2.0):
    return explosion(rng, size=0.45, dist=dist) * 0.8


def metal_scream(dur, rng):
    """'Metalin çığlığı': dev yapının sürtünme çığlığı + inilti + alt bas, yavaş perde düşüşü."""
    n = T(dur); tl = tt(n); x = np.linspace(0, 1, n)
    f = 330 * (0.55 ** x) * (1 + 0.03 * smooth_noise(n, 6, rng) + 0.012 * np.sin(2 * np.pi * 9 * tl))
    ph = 2 * np.pi * np.cumsum(f) / SR
    stick = 0.6 + 0.4 * np.abs(smooth_noise(n, 25, rng))
    sq = sum(np.sin(k * ph + 0.4 * k) * (0.9 ** k) for k in range(1, 18)) * stick
    sq = bp(sq, 150, 6000) * env(n, [(0, 0), (dur * .08, 1), (dur * .5, .8), (dur * .85, .5), (dur, 0)])
    exc = brown(n, rng) * env(n, [(0, .2), (dur * .2, 1), (dur, .3)])
    modes = 48 * np.array([1, 1.37, 1.92, 2.31, 2.88, 3.53, 4.17, 5.02, 5.89, 7.13, 8.4, 9.9])
    groan = resbank(exc, modes, 3.0, np.linspace(1, .2, len(modes))) * 0.02
    groan = varispeed(np.concatenate([groan, np.zeros(T(3))]), np.linspace(1.0, 0.8, n + T(3)))[:n]
    sub = lp(brown(n, rng), 70) * env(n, [(0, .3), (dur * .3, 1), (dur, .4)]) * 0.8
    y = saturate(sq * 0.35 + groan + sub, 1.5)
    return widen(y, rng, 1.0, 15)


def riser(dur, rng):
    n = T(dur); x = np.linspace(0, 1, n)
    y = svf(pink(n, rng), 200 * (40 ** x ** 1.5), 2.0, 'bp') * x ** 2
    f = 110 * (2 ** (x * 1.5)); ph = 2 * np.pi * np.cumsum(f) / SR
    y += (np.sin(ph) + 0.4 * np.sin(2.01 * ph)) * x ** 3 * 0.25
    return widen(y, rng, 1.0, 11)


def energy_surge(dur, rng):
    """Pencere bembeyaz ışıkla doluyor: yükselen enerji + elektrik arkı çıtırtıları + patlama."""
    n = T(dur); x = np.linspace(0, 1, n)
    hum = sphere_hum(dur, rng, f0=196, bright=1.5)
    arc = hp(white(n, rng), 2500) * (rng.random(n) < 0.004 + 0.02 * x) * 3
    arc = lp(np.abs(arc), 9000) * np.sign(rng.standard_normal(n))
    y = hum * (x ** 1.5)[None, :] * 1.5 + pan(bp(arc, 2000, 9000) * x * 0.4, 0)
    b = impact(rng, 1.0)
    k = n - T(0.05)
    out = np.zeros((2, n + b.shape[1])); out[:, :n] += y; out[:, k:k + b.shape[1]] += b * 0.8
    return out
