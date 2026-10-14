"""SFX kanalı: animasyonla senkron ortam yatakları + olay efektleri. Çıktı: out/sfx.wav"""
import os, sys, time
import numpy as np
import soundfile as sf
from lib import *
import sfxkit as K
from tl import cue, scene, END, SHOTS, MORSE_SHOTS, SIG7, D

t_start = time.time()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2024)
N = T(END + 1.0)
M = np.zeros((2, N), np.float32)
BUS = {k: np.zeros((2, N), np.float32) for k in ('room', 'stair', 'out', 'big', 'radio')}
IR = {
    'room': make_ir(0.5, 0.004, 0.025, 10, damp=1.1, width=0.7, seed=1),
    'stair': make_ir(1.8, 0.008, 0.06, 16, damp=0.9, width=0.9, seed=2),
    'out': make_ir(2.4, 0.03, 0.18, 14, damp=1.4, width=1.0, seed=3),
    'big': make_ir(5.0, 0.07, 0.4, 18, damp=1.6, width=1.0, seed=4),
    'radio': make_ir(0.35, 0.002, 0.012, 6, damp=1.0, width=0.5, seed=5),
}
LOG = []


def C(spec, fb=None):
    try:
        return cue(spec)
    except Exception as e:
        if fb is None:
            raise
        print('.. cue', spec, '->', fb); return cue(fb)


def add(snd, t, g=0.0, p=0.0, bus=None, send=0.0, dry=1.0, norm=True, name=''):
    """snd: mono/stereo; t: saniye; g: dB (tepe-normalize sonrası); bus: yankı veri yolu."""
    if t is None:
        return
    s = st(np.asarray(snd, float), p)
    if norm:
        s = s / (np.max(np.abs(s)) + 1e-12)
    s = (s * db(g)).astype(np.float32)
    if dry > 0:
        K.place(M, s * dry, t)
    if bus and send > 0:
        K.place(BUS[bus], s * send, t)
    if name:
        LOG.append((t, name))


def rmsnorm(x, target_db):
    r = np.sqrt(np.mean(np.asarray(x, float) ** 2)) + 1e-12
    return x * (db(target_db) / r)


def bed(fn, t0, t1, g_db, fin=1.5, fout=1.5, bus=None, send=0.0, chunk=40.0, auto=None, name=''):
    """Uzun ortam yatağı: parça parça üretilip çapraz geçişle eklenir. auto=[(t, dB)] kazanç otomasyonu."""
    xf = 2.0; t = t0; first = True
    while t < t1 - 0.01:
        a = t; b = min(t1, t + chunk)
        L = b - a + (xf if b < t1 else 0)
        x = st(fn(T(L)))
        x = rmsnorm(x, g_db)
        x = K.fade(x, fin if first else xf, fout if b >= t1 else xf)
        if auto is not None:
            ts = a + np.arange(x.shape[1]) / SR
            x = x * db(np.interp(ts, [p[0] for p in auto], [p[1] for p in auto]))[None, :]
        add(x, a, 0, norm=False, bus=bus, send=send)
        first = False; t = b
    if name:
        LOG.append((t0, name + ' (yatak)'))


# =================================================================== 1) GENEL GEÇİŞLER
SPACE_OUT = {'street', 'city', 'skyUp', 'roof', 'highway', 'stall', 'pasture', 'mine', 'battle', 'shipCU', 'gulls', 'noshadow', 'jets', 'uav', 'truck', 'drones'}
for s in SHOTS:
    tr, t0 = s.get('tr'), s['s']
    if tr in ('whip', 'whipL'):
        add(K.whoosh(0.55, rng, 500, 4200, 0.8 if tr == 'whip' else -0.8, -0.8 if tr == 'whip' else 0.8, peak=0.55, q=1.1), t0 - 0.3, -19, send=0.15, bus='out')
    elif tr == 'whipV':
        add(K.whoosh(0.6, rng, 1800, 300, 0, 0, peak=0.5, q=1.2), t0 - 0.32, -20, send=0.15, bus='out')
    elif tr == 'zoom':
        add(K.whoosh(0.9, rng, 250, 2600, -0.2, 0.2, peak=0.85, q=1.4, tone=0.15), t0 - 0.72, -21, send=0.2, bus='big')
    elif tr == 'flash':
        add(K.impact(rng, 0.55), t0 - 0.02, -21, send=0.25, bus='big')
    elif tr == 'glitch':
        n = T(0.45); g = (rng.random(n) < 0.5).astype(float); g = np.repeat(g[::T(0.03)], T(0.03))[:n]
        z = K.radio_static(n, rng, 1.2, 3.0, 0.5, 1.5) * np.pad(g, (0, max(0, n - len(g))))[:n]
        add(z, t0 - 0.05, -22, p=rng.uniform(-.3, .3))

# Tipografik vurgu (slam) ve mühür (stamp) yazıları
for c in D['caps']:
    if c['st'] == 'slam':
        add(lp(K.impact(rng, 0.45), 3500), c['s'] - 0.01, -25, send=0.2, bus='big')
    elif c['st'] == 'stamp':
        th = K.wood_thump(rng, 130) + 0.0
        pa = K.paper(0.15, rng)
        y = np.zeros(T(0.6)); y[:len(th)] += th[:len(y)]; y[:len(pa)] += pa * 0.3
        add(y, c['s'] + 0.02, -12, send=0.25, bus='room', name='mühür')

# Bölüm kartları (gün başlıkları): derin darbe
for s in SHOTS:
    if s['R'] == 'chapter':
        add(K.impact(rng, 1.3), s['s'] + 0.05, -9, send=0.35, bus='big', name='gün kartı darbesi')
        add(K.riser(1.6, rng), s['s'] - 1.55, -27, send=0.2, bus='big')

# =================================================================== 2) AÇILIŞ (0–50 s)
add(K.impact(rng, 1.4), C('1:uzaylılar') - 0.03, -6, send=0.45, bus='big', name='açılış darbesi')
add(K.ship_horn(5.5, rng, 43), C('1:uzaylılar') + 0.1, -16, send=0.6, bus='big', name='açılış uğultusu')
bed(lambda n: K.room_tone(n, rng), 3.2, 50.4, -44, fin=0.8, fout=0.3, name='oda sessizliği')
bed(lambda n: K.city_hush(n, rng, 0.8), 9.5, 50.4, -46, fin=2.0, fout=0.5, bus='out', send=0.3)
bed(lambda n: K.wind(n, rng, 0.7, 0.8, 0.0, 150, 600), 12.0, 50.4, -42, fin=3.0, fout=0.5, bus='out', send=0.3, name='gece rüzgârı')
add(K.watch_ticks(1.6, rng, 1.0), C('2:saat') - 0.1, -34, bus='room', send=0.3)
add(K.curtain(1.4, rng), C('4:perdeyi') - 0.1, -17, bus='room', send=0.35, name='perde')
add(K.sphere_hum(6.0, rng, f0=110, bright=0.4), C('5:bütün'), -34, send=0.5, bus='big')
add(K.uav(2.2, rng, 230) * np.linspace(0.3, 1, T(2.2)) * np.linspace(1, 0, T(2.2)) ** 0.5, C('12:vızıldar') - 0.3, -30, p=0.4, bus='out', send=0.4, name='dron vızıltısı')
add(K.paper(0.4, rng) * 0.5, C('13:uzanıyorsun') - 0.2, -28, bus='room', send=0.2)
add(K.switch_click(rng), C('13:video') - 0.1, -26, bus='room', send=0.2)

# =================================================================== 3) UĞULTU + KARARMA (50–93 s)
t_h = C('14:ses') - 0.3
add(K.ship_horn(10.5, rng, 46), t_h, -8, send=0.7, bus='big', name='GEMİNİN UĞULTUSU')
add(K.ship_horn(9.0, rng, 34.5), t_h + 1.2, -15, p=0.2, send=0.8, bus='big')
gr_t = C('15:camlar') - 0.6
n = T(5.5); inten = env(n, [(0, 0), (0.5, 1), (3.0, .8), (5.5, 0)])
add(K.glass_rattle(n, rng, inten), gr_t, -12, bus='room', send=0.3, name='camlar zangırdıyor')
# ışıklar sönüyor: oda lambası, sokak, karşı yaka, köprüler — giderek uzaklaşan
for spec, dist, g, p in [('15:odanın', 1.0, -14, -0.2), ('16', 2.0, -22, 0.5), ('16:karşıyakanın', 3.0, -27, -0.6), ('16:köprülerin', 4.0, -30, 0.3)]:
    t = C(spec); add(K.power_down(rng, dist), t, g, p=p, bus='out' if dist > 1 else 'room', send=0.5 if dist > 1 else 0.3, name='ışık sönüyor ' + spec)
    if dist > 2:
        add(K.transformer_pop(rng), t + 0.25, g - 6, p=p, bus='big', send=0.8, dry=0.3)
add(K.switch_click(rng), C('17:telefonun') + 0.4, -24, bus='room', send=0.2, name='telefon kararıyor')
# geminin uğultusu yatağı başlıyor (bundan sonra gemi gidene kadar sürer — ayrı bölümde)
# köpekler: hepsi aynı anda havlıyor ve aynı saniyede susuyor
t_d0, t_d1 = C('19:köpekler') + 0.6, C('20:susuyor') + 0.05
k = 0
for i in range(70):
    tt0 = t_d0 + rng.random() * (t_d1 - t_d0 - 0.25)
    dist = rng.uniform(1.0, 4.0)
    b = K.gs_note(1, 123, int(rng.integers(52, 66)), 0.2, vel=int(rng.integers(90, 127)))
    b = lp(b, 9000 / dist)
    add(b, tt0, -13 - dist * 4, p=rng.uniform(-1, 1), bus='out', send=0.25 + 0.15 * dist, dry=1.0 / dist ** 0.5)
LOG.append((t_d0, 'köpekler havlıyor'))
LOG.append((t_d1, 'köpekler aynı anda susuyor'))
add(K.heartbeat(3.0, rng, 60, 66), C('20:hepsi') + 0.5, -24, name='sessizlikte kalp')

# =================================================================== 4) GEMİ UĞULTUSU YATAĞI (gemi gidene dek)
t_hum0 = C('14:uğultu', '14:derin') + 2.0
t_hum1 = C('203:kayboluyor', '203') + 0.5
auto = [(t_hum0, -6), (t_hum0 + 6, 0), (95, -4), (118, -6), (120, 0), (205, -2), (215, -5), (300, -3), (312, -9), (390, -2), (436, 0), (448, -4),
        (493, -6), (640, -4), (650, 0), (700, -4), (720, -7), (760, -5), (777, -3), (826, 0), (856, 3), (880, 2), (890, -3), (935, -4), (1050, 0),
        (1100, -2), (1106, 0), (1131, 3), (1143, -12), (1145.5, 2), (1168, 5), (1175, 0), (1205, 2), (t_hum1 - 6, 0), (t_hum1, -30)]
bed(lambda n: K.ship_hum(n, rng, 31.0), t_hum0, t_hum1, -30, fin=5, fout=4, bus='big', send=0.15, auto=auto, name='gemi uğultusu')

# =================================================================== 5) TEZ (93–118 s)
add(K.radio_static(T(2.5), rng, 1.0, 1.5), C('24:radyosu') - 0.2, -30, bus='radio', send=0.4, name='radyo cızırtısı (önizleme)')
add(K.jet_pass(4.0, rng, 0.45), 108.6, -24, bus='out', send=0.4, name='savaş uçağı (görsel)')

# =================================================================== 6) BİRİNCİ GÜN: MAHALLE (118–210 s)
bed(lambda n: K.wind(n, rng, 0.8, 0.9, 0.0, 200, 900), 119.4, 209.8, -38, fin=2, fout=2, bus='out', send=0.2, name='sabah rüzgârı')
bed(lambda n: K.walla(n / SR, rng, 18, rate=3.2), 124.4, 146.0, -40, fin=3, fout=4, bus='out', send=0.5, name='mahalle mırıltısı')
add(K.wood_thump(rng, 220), C('28:tezgahını'), -24, bus='out', send=0.4, name='simit tezgahı')
add(K.glass_clink(rng, False), C('28:tezgahını') + 0.4, -30, bus='out', send=0.4)
add(K.walla(4, rng, 10, rate=3.0), C('33:gündüz'), -40, bus='out', send=0.5)
add(K.riser(3.0, rng), C('35:tüyleri') - 2.4, -28, send=0.3, bus='big', name='tüyler diken')
# ölü arabalar: marş basıp çalışmayan motor
def starter(dur):
    n_ = T(dur); t_ = tt(n_)
    comp = 0.55 + 0.45 * np.sign(np.sin(2 * np.pi * 5.2 * t_)) * np.abs(np.sin(2 * np.pi * 5.2 * t_)) ** 0.5
    ph = 2 * np.pi * np.cumsum(150 * (0.8 + 0.2 * comp)) / SR
    y = sum(np.sin(k_ * ph) / k_ for k_ in range(1, 20)) * comp
    return bp(y, 90, 4000) * env(n_, [(0, 0), (0.05, 1), (dur - 0.3, .8), (dur, 0)])
add(starter(1.8), C('38:arabaların'), -24, p=0.3, bus='out', send=0.4, name='marş basıyor, çalışmıyor')
add(starter(1.3), C('38:arabaların') + 2.3, -28, p=-0.4, bus='out', send=0.5)
add(K.transformer_pop(rng), C('39:fişini') + 0.2, -14, bus='big', send=0.5, name='fişi çekilmiş şehir')
add(K.footsteps(4.0, rng, 1.6), C('39:deden') - 0.3, -22, p=0.3, bus='out', send=0.3, name='dedenin ayak sesleri')

# =================================================================== 7) RIZA DEDE VE RADYO (210–308 s)
bed(lambda n: K.wind(n, rng, 0.6, 0.6, 0.0, 200, 700), 209.8, 307.7, -42, fin=2, fout=2, bus='out', send=0.2)
add(K.wood_thump(rng, 170), C('41:elinde') + 0.5, -22, bus='out', send=0.3, name='ahşap radyo')
t_ak = C('42:aküye')
add(K.click(rng, [2100, 3400, 5200], [0.05, 0.04, 0.03], [1, .7, .4], n_exc=16, dur=0.2), t_ak + 0.3, -20, bus='room', send=0.3, name='akü maşası')
add(K.click(rng, [1900, 3100, 4800], [0.05, 0.04, 0.03], [1, .7, .4], n_exc=16, dur=0.2), t_ak + 0.65, -21, bus='room', send=0.3)
sp = K.radio_static(T(0.35), rng, 0.5, 6, 0, 0); add(sp, t_ak + 0.68, -24)
# lamba ısınması: vızıltı yükselir, cızırtı açılır
n = T(5.0); warm = env(n, [(0, 0), (2.5, .5), (5, 1)])
add(K.radio_static(n, rng, 1.0, 1.2, 2.0) * warm, t_ak + 0.9, -24, bus='radio', send=0.3, name='lamba ısınıyor + cızırtı')
t_br = C('43:cızırtının')
def tune(dur):
    n_ = T(dur); x_ = np.linspace(0, 1, n_)
    f_ = 400 + 2600 * (0.5 + 0.5 * np.sin(2 * np.pi * x_ * 1.3 + 1))
    w_ = np.sin(2 * np.pi * np.cumsum(f_) / SR) * 0.15
    return K.radio_filter(w_ + K.radio_static(n_, rng, 1.0, 1.0) * 0.5)
add(tune(2.0), t_br - 0.6, -25, bus='radio', send=0.3, name='istasyon arama')
# Ankara yayını: anlaşılmaz spiker sesi + cızırtı (anlatımın altında, kısık)
for a, b, gg in [(t_br + 1.2, C('44') - 0.2, -27), (C('46:radyotaki'), C('49') - 0.5, -29)]:
    L = b - a
    v = K.babble_voice(L, rng, f0=118, rate=4.2)
    stc = K.radio_static(T(L), rng, 0.6, 0.8)
    add(K.fade(v * 0.8 + stc * 0.5, 0.4, 0.8), a, gg, bus='radio', send=0.35, name='Ankara yayını (radyo)')
add(K.jet_pass(3.2, rng, 0.5, 4200), C('44') + 0.1, -16, bus='out', send=0.45, name='MiG-25 geçişi')
# EMP darbesi görseli
add(K.energy_surge(1.4, rng), C('45:elektromanyetik') - 1.2, -20, bus='big', send=0.4, name='EMP')
add(K.walla(5, rng, 14, rate=3.4), C('52:kuyumcuların'), -34, bus='out', send=0.4, name='kuyumcu kuyruğu')
add(K.coins(rng, 6), C('53:gram'), -22, bus='room', send=0.25, name='altın şıngırtısı')
add(K.coins(rng, 4), C('53:gram') + 1.1, -25, bus='room', send=0.25)
add(K.gs_note(5, 124, 72, 1.2), C('54:asıyor') - 0.3, -22, p=-0.3, bus='out', send=0.4, name='nazar boncuğu / rüzgâr çanı')
add(K.glass_clink(rng, True), C('54:asıyor') + 0.1, -24, p=-0.3, bus='out', send=0.3)

# =================================================================== 8) İKİNCİ GÜN (308–388 s)
bed(lambda n: K.room_tone(n, rng, 1.0) + K.walla(n / SR, rng, 12, rate=2.5)[:, :n] * 0.4, C('56:ankarada'), C('57:türkiye') - 0.5, -40, fin=1, fout=1.5, bus='room', send=0.4, name='MGK salonu')
t_u = C('65:onlarca')
n = T(6.5); uv = K.uav(6.5, rng, 96) * env(n, [(0, 0), (2.5, 1), (6.5, 1)])
add(uv, t_u - 0.5, -18, p=-0.3, bus='out', send=0.35, name='İHA sürüsü yaklaşıyor')
add(K.uav(6.0, rng, 104) * env(T(6.0), [(0, 0), (3, 1), (6, 1)]), t_u + 0.3, -21, p=0.4, bus='out', send=0.4)
t_fall = C('66:taş') - 0.4
for i in range(6):
    tf = t_fall + i * 0.55 + rng.random() * 0.2
    add(K.falling_whistle(1.6, rng, 1700 - i * 60, 420), tf, -20 - i, p=rng.uniform(-.8, .8), bus='out', send=0.4)
    add(K.explosion(rng, 0.35, 3.5), tf + 1.55, -18 - i, p=rng.uniform(-.8, .8), bus='big', send=0.6, name='İHA düşüyor' if i == 0 else '')
t_j = C('67:savaş')
n = T(6.0); jt = K.jet_pass(6.0, rng, 0.7, 5600)
add(jt, t_j - 0.5, -13, bus='out', send=0.4, name='savaş uçakları kalkıyor')
add(K.jet_pass(5.0, rng, 0.6, 6100), t_j + 0.8, -17, bus='out', send=0.5)
# gösterge alarmı + parazit
t_c = C('67:göstergeleri')
n = T(4.0); tc = tt(n)
caution = np.sin(2 * np.pi * 1050 * tc) * ((tc % 0.5) < 0.22) * 0.5 + np.sin(2 * np.pi * 1310 * tc) * ((tc % 0.5) < 0.22) * 0.3
add(K.radio_filter(caution) * env(n, [(0, 0), (0.05, 1), (3.9, 1), (4, 0)]) + K.radio_static(n, rng, 0.8, 3) * 0.6, t_c, -24, bus='room', send=0.2, name='kokpit uyarısı')
add(K.power_down(rng, 1.0), C('68:ekranlar') + 0.2, -18, bus='room', send=0.2, name='ekranlar kararıyor')
add(K.jet_pass(5.0, rng, 0.25, 5200), C('68:geri') - 0.3, -20, p=0.5, bus='out', send=0.5, name='uçaklar geri dönüyor')

# =================================================================== 9) GÖÇ, ÇAYCI, BAKKAL (388–437 s)
t0, t1 = C('69:yüz'), C('72:ama#2', '73')
bed(lambda n: K.footsteps(n / SR, rng, 7.5) * 0.5 + K.footsteps(n / SR, rng, 5.3) * 0.5, t0, t1, -34, fin=2, fout=2, bus='out', send=0.4, name='yürüyen kalabalık')
bed(lambda n: K.walla(n / SR, rng, 22, rate=3.6), t0, t1, -36, fin=2, fout=2, bus='out', send=0.5)
def wheels(dur):
    n_ = T(dur)
    return bp(brown(n_, rng), 120, 2000) * (0.8 + 0.2 * smooth_noise(n_, 3, rng)) + bp((rng.random(n_) < 18 / SR) * rng.standard_normal(n_), 800, 4000) * 0.8
bed(lambda n: wheels(n / SR), C('70:otoyolun'), t1, -38, fin=1.5, fout=2, bus='out', send=0.3, name='bavul tekerlekleri')
for i in range(5):
    add(K.glass_clink(rng, True), C('71:yolun') + 0.8 + i * 1.3 + rng.random() * 0.4, -24 - i, p=0.4, bus='out', send=0.35)
LOG.append((C('71:yolun'), 'çay bardakları'))
t_k = C('74:dükkanını')
def shutter(dur):
    n_ = T(dur); y_ = np.zeros(n_); x_ = 0.0
    while x_ < dur - 0.1:
        c_ = K.click(rng, np.array([820, 1530, 2650, 3900]) * (0.9 + 0.2 * rng.random()), [0.06, 0.05, 0.04, 0.03], [1, .8, .6, .4], n_exc=20, dur=0.12, hp_hz=300)
        i_ = T(x_); m_ = min(len(c_), n_ - i_); y_[i_:i_ + m_] += c_[:m_] * (0.6 + 0.4 * rng.random())
        x_ += 0.035 + 0.02 * rng.random()
    roll = bp(brown(n_, rng), 100, 1500) * 0.4
    return (y_ + roll) * env(n_, [(0, 0), (0.1, 1), (dur - 0.3, 1), (dur, 0)])
add(shutter(2.2), t_k - 0.2, -14, bus='out', send=0.35, name='kepenk açılıyor')
add(K.shop_bell(rng), t_k + 2.1, -22, bus='room', send=0.3)
add(K.wood_thump(rng, 140), C('75:veresiye') - 0.25, -15, bus='room', send=0.3, name='veresiye defteri')
add(K.paper(1.0, rng), C('75:veresiye') + 0.2, -26, bus='room', send=0.2)
def pencil(dur):
    n_ = T(dur); x_ = tt(n_)
    stroke = np.abs(np.sin(2 * np.pi * 3.4 * x_)) ** 0.6 * (0.6 + 0.4 * smooth_noise(n_, 4, rng))
    return bp(white(n_, rng), 2500, 9000) * stroke * 0.5
add(pencil(1.6), C('76:yazarım') - 0.1, -26, bus='room', send=0.2, name='kurşun kalem')

# =================================================================== 10) İLK DERS: 'DUYAN VAR MI' (437–493 s)
bed(lambda n: K.wind(n, rng, 0.7, 1.0, 0.15, 180, 700), 436.8, 492.6, -40, fin=2, fout=2, bus='out', send=0.25, name='gece rüzgârı')
bed(lambda n: K.radio_static(n, rng, 0.8, 0.9), C('78:dedenin'), C('83:deden') + 0.6, -36, fin=1, fout=0.05, bus='radio', send=0.3, name='radyo cızırtısı')
t_v = C('79:burası')
for k_, (a, L) in enumerate([(t_v - 0.2, 1.4), (t_v + 1.4, 1.2), (t_v + 2.8, 1.2)]):
    add(K.babble_voice(L, rng, f0=128, rate=4.6), a, -26, bus='radio', send=0.35, name='telsizci yayını' if k_ == 0 else '')
    add(K.radio_static(T(0.18), rng, 2.5, 1, 0, 0), a + L, -24)
t_s = C('79:gece')
add(K.whoosh(3.2, rng, 1800, 260, 0.6, 0.0, peak=0.7, q=1.6, tone=0.25), t_s, -18, bus='big', send=0.5, name='küre iniyor')
add(K.sphere_hum(9.0, rng, 233), t_s + 2.2, -22, bus='big', send=0.5)
# taklit ses: aynı kesit, birebir tekrar (nefessiz) + halka modülasyonu
mim = K.babble_voice(1.25, rng, f0=128, rate=4.6)
tm = tt(len(mim)); mim2 = mim * (0.7 + 0.3 * np.sin(2 * np.pi * 31 * tm))
t_m = C('81:hoparlörden')
for i in range(9):
    add(mim2, t_m + i * 1.32, -24 - (i % 3), bus='radio', send=0.4, name='taklit ses (birebir tekrar)' if i == 0 else '')
add(K.knob_click(rng), C('83:deden') + 0.55, -18, bus='room', send=0.3, name='radyo kapatılıyor')
add(K.heartbeat(5.0, rng, 88, 96), C('83:elleri') - 0.2, -25)
add(K.wind(T(5.0), rng, 1.0, 1.2, 0.5), C('84:sabah'), -30, bus='out', send=0.3, name='boş çatıda rüzgâr')
add(K.impact(rng, 0.9), C('86:bulurlar') + 0.05, -14, send=0.5, bus='big', name='"seni bulurlar"')

# =================================================================== 11) ÜÇÜNCÜ GÜN: BOR (493–642 s)
bed(lambda n: K.radio_static(n, rng, 0.7, 0.8), C('86:sabah'), C('89:kırkadan') + 0.5, -37, fin=1, fout=1.5, bus='radio', send=0.3)
add(K.babble_voice(5.0, rng, f0=115, rate=4.3) * 0.8, C('86:beklenmedik') - 0.3, -28, bus='radio', send=0.3, name='sabah haberi (radyo)')
add(K.ship_open(5.0, rng), C('86:gece') - 0.2, -18, bus='big', send=0.5, name='geminin altı açılıyor')
add(K.swarm(6.0, rng, 18, 230), C('87:içinden'), -20, bus='big', send=0.5, name='yüzlerce küçük gemi')
add(K.whoosh(2.4, rng, 2200, 300, 0.0, 0.9, 0.5), C('87:güneye') - 0.4, -18, bus='big', send=0.4)
# mera: rüzgâr, koyun çanları, meleme; toprak yukarı akıyor (ters yağmur)
bed(lambda n: K.wind(n, rng, 1.0, 1.0, 0.1, 250, 1100), C('89:kırkadan') - 0.5, C('94:neden') - 0.5, -36, fin=1.5, fout=2, bus='out', send=0.3, name='Anadolu merası')
def sheep_bells(dur, density=4.0):
    n_ = T(dur); y_ = np.zeros(n_); x_ = 0.0
    while x_ < dur - 0.3:
        f_ = rng.choice([1450, 1620, 1890, 2140, 2380]) * (0.97 + 0.06 * rng.random())
        b_ = K.click(rng, np.array([1, 2.4, 3.9, 5.1]) * f_, [0.25, 0.18, 0.12, 0.08], [1, .5, .3, .2], n_exc=8, dur=0.4, hp_hz=600)
        i_ = T(x_); m_ = min(len(b_), n_ - i_); y_[i_:i_ + m_] += b_[:m_] * (0.3 + 0.7 * rng.random())
        x_ += rng.exponential(1 / density)
    return y_
a_, b_ = C('89:kırkadan'), C('91:koyunlar')
add(K.widen(sheep_bells(b_ - a_), rng, 1.0, 9), a_, -26, bus='out', send=0.4, name='koyun çanları')
for i in range(5):
    add(K.sheep(rng, 270 + 60 * rng.random()), a_ + 0.5 + rng.random() * (b_ - a_ - 1), -27, p=rng.uniform(-.8, .8), bus='out', send=0.5)
add(K.babble_voice(2.4, rng, f0=135, rate=5.0), C('90:sesi') - 0.1, -27, bus='radio', send=0.3, name='çobanın titrek sesi (radyo)')
n = T(5.5); rr_ = K.rain(n, rng, 0.8)[:, ::-1] * env(n, [(0, 0), (2, 1), (5.5, .3)])[None, :]
add(rr_, C('90:toprak') - 0.2, -22, bus='out', send=0.4, name='toprak yukarı akıyor (ters yağmur)')
add(K.sphere_hum(5.0, rng, 147, 0.6), C('91:koyunlar') - 0.2, -24, bus='big', send=0.5, name='koyunlar havalanıyor')
for i in range(4):
    add(K.sheep(rng, 330 + 50 * rng.random()), C('91:koyunlar') + 0.3 + i * 0.4, -25, p=rng.uniform(-.6, .6), bus='out', send=0.6)
add(K.impact(rng, 1.1), C('96:bor') - 0.05, -10, send=0.5, bus='big', name='"BOR"')
add(K.radio_static(T(3.0), rng, 0.6, 0.7), C('95:ama') , -36, bus='radio', send=0.3)
# reaktör/nötron anlatımı: düşük reaktör uğultusu
def reactor(dur):
    n_ = T(dur); t_ = tt(n_)
    y_ = sum(np.sin(2 * np.pi * f_ * t_ + rng.random() * 6) * a for f_, a in [(60, 1), (120, .5), (180, .25), (240, .15)])
    return lp(y_ + bp(pink(n_, rng), 400, 2000) * 0.2, 2500) * env(n_, [(0, 0), (1, 1), (dur - 1, 1), (dur, 0)])
add(reactor(6.0), C('101:nükleer'), -30, bus='big', send=0.3, name='reaktör uğultusu')
add(K.swarm(5.0, rng, 10, 160), C('107:gölgesinde', '108') - 0.5, -26, bus='big', send=0.5)
add(lp(brown(T(5.0), rng), 120) * env(T(5.0), [(0, 0), (1, 1), (4, 1), (5, 0)]), C('108:anadoluyu', '108') , -28, bus='big', send=0.5, name='kazı gümbürtüsü')
t_rv = C('108:radyonun')
n = T(3.0); add(K.radio_static(n, rng, 0.8, 1.0) * env(n, [(0, 1), (1.0, 1), (2.0, .15), (3, .15)]), t_rv - 0.5, -32, bus='radio', send=0.3, name='radyonun sesi kısılıyor')
add(K.knob_click(rng), t_rv + 0.3, -24, bus='room', send=0.2)
add(K.impact(rng, 0.8), C('110:kalmayacak') + 0.1, -16, send=0.5, bus='big')

# =================================================================== 12) DÖRDÜNCÜ GÜN: AÇIKLIK + MORS (642–777 s)
bed(lambda n: K.wind(n, rng, 0.8, 1.0, 0.1, 180, 700), 646.6, 653.7, -36, fin=1, fout=1.5, bus='out', send=0.3, name='balkon gecesi')
add(K.watch_ticks(5.0, rng), C('111:kurmalı') - 0.3, -24, name='cep saati')
add(K.paper(1.1, rng), C('112:kağıt') - 0.1, -22, bus='room', send=0.25, name='kâğıt uzatıyor')
add(K.watch_ticks(5.5, rng), C('113') + 0.2, -28)
add(K.ship_open(5.5, rng), C('115') - 0.3, -20, bus='big', send=0.6, name='açılış (anlatım)')
add(K.swarm(4.0, rng, 12, 240), C('115') + 1.0, -26, bus='big', send=0.5)
n = T(5.5); add(K.radio_static(n, rng, 0.8, 1.0) * env(n, [(0, 1), (1.2, .25), (4, .25), (5.5, 1)]), C('116:açıklık') - 0.3, -32, bus='radio', send=0.3, name='cızırtı azalıyor')
add(K.whoosh(2.0, rng, 300, 1500, 0, 0, 0.8, tone=0.3), C('117:dedenin') - 0.5, -26, bus='big', send=0.4)
add(K.wood_thump(rng, 120), C('119:gardırobun') + 0.4, -16, bus='room', send=0.35, name='kutu indiriliyor')
add(K.puff(rng), C('119:toz') - 0.05, -22, bus='room', send=0.3)
for d_ in (0.0, 0.35):
    add(K.switch_click(rng), C('119:içinde#2', '119:içinde') + d_, -22, bus='room', send=0.25)
LOG.append((C('119:içinde#2', '119:içinde'), 'kutunun kilitleri'))
add(K.curtain(1.3, rng), C('121:perdeleri') - 0.3, -20, bus='room', send=0.3, name='perde kapanıyor')
add(K.puff(rng), C('121:mumu') + 0.3, -18, bus='room', send=0.2, name='mum söndürülüyor')

# ---------------- MORS: görseldeki her tık/tak ile birebir (anahtar + yan ton)
def key_events(seq_list, room='room', g_key=-16, g_tone=-22, f=640, muffled=False, mon=True):
    """seq_list: [(baş, süre)] — anahtar tıkları + verici yan tonu."""
    for a, d_ in seq_list:
        kd = K.morse_key(rng, True); ku = K.morse_key(rng, False)
        if muffled:
            kd = lp(kd, 1500); ku = lp(ku, 1500)
        add(kd, a - 0.004, g_key + rng.normal(0, 1.0), bus=room, send=0.35 if not muffled else 0.6, dry=1.0 if not muffled else 0.5)
        add(ku, a + d_, g_key - 4 + rng.normal(0, 1.0), bus=room, send=0.35 if not muffled else 0.6, dry=1.0 if not muffled else 0.5)
    if mon and seq_list:
        tone = K.cw_signal([(a - seq_list[0][0] + 0.02, d_) for a, d_ in seq_list], rng, f=f)
        tone = K.radio_filter(tone, 1.3)
        if muffled:
            tone = lp(tone, 1200)
        add(tone, seq_list[0][0] - 0.02, g_tone, bus=room, send=0.3 if not muffled else 0.6, dry=1.0 if not muffled else 0.5, norm=True)


def seq_of(shot_i):
    m = [x for x in MORSE_SHOTS if x['i'] == shot_i][0]
    out = []
    for e in (m['seq'] or []):
        if m['s'] - 0.05 <= e['at'] < m['e'] - 0.05:
            out.append((e['at'], 0.3 if e['s'] == '-' else 0.1))
    return out


def human_morse(t0, t1, unit=0.085, jitter=0.18, text='CQ CQ DE TA1RD SOS SOS ISTANBUL K'):
    """Görsel olmayan aralıklarda dedenin el ritmiyle süren Mors gönderimi."""
    out = []; t = t0; k = 0
    while t < t1:
        seg, t = K.morse_code_times(text[k % len(text):] + ' ', t, unit, jitter, rng)
        for a, d_ in seg:
            if a + d_ < t1:
                out.append((a, d_))
        k += 7
    return out


# Sahne 120: anahtar tanıtımı
add(K.morse_key(rng, True), 704.6, -16, bus='room', send=0.3); add(K.morse_key(rng, False), 704.75, -20, bus='room', send=0.3)
# Sahne 122–125: dede gönderiyor (görsel dizilerle birebir; arada süreklilik)
vis = seq_of(317) + seq_of(320) + seq_of(324)
vis.sort()
fill = [e for e in human_morse(C('123:10') - 0.2, 731.7, 0.09) if not (717.8 <= e[0] <= 723.5)]
fill += [e for e in human_morse(732.75, 736.9, 0.09)]
allm = sorted(vis + fill)
allm = [e for e in allm if e[0] < C('125:elini', '125:dede#2') - 0.1]
key_events(allm, 'room', -15, -23, 640)
LOG.append((allm[0][0], 'MORS: dede gönderiyor (görselle senkron)'))
# ışık yaklaşıyor
t_l = C('124:perdenin')
add(K.sphere_hum(16.0, rng, 233) * env(T(16), [(0, 0), (5, .5), (9, 1), (12, .9), (16, 0)])[None, :], t_l - 0.5, -18, bus='big', send=0.5, name='küre yaklaşıyor')
add(K.whoosh(3.5, rng, 600, 200, 0.2, 0.9, 0.3, q=1.4, tone=0.2), C('126:yavaşça') - 0.3, -22, bus='big', send=0.5, name='küre çekiliyor')
# bekleme: cızırtı, sonra Ankara cevap veriyor (CW, sönümlenmeli)
bed(lambda n: K.radio_static(n, rng, 0.9, 1.0), C('126:bekliyorsunuz') - 0.3, 760.5, -35, fin=1, fout=1.2, bus='radio', send=0.3, name='cızırtı — bekleyiş')
add(K.watch_ticks(5.0, rng), C('127:bir') - 0.2, -30)
rx = seq_of(333)
tone = K.cw_signal([(a - rx[0][0] + 0.02, d_) for a, d_ in rx], rng, f=760, drift=0.004, fade=0.35)
add(K.radio_filter(tone, 1.25) + K.radio_static(len(tone), rng, 0.6, 1.0) * 0.4, rx[0][0] - 0.02, -21, bus='radio', send=0.3, name='ANKARA CEVAP VERİYOR (CW)')
rx2, _ = K.morse_code_times('QSL QRV K', 755.6, 0.075)
tone = K.cw_signal([(a - rx2[0][0] + 0.02, d_) for a, d_ in rx2], rng, f=760, drift=0.004, fade=0.35)
add(K.radio_filter(tone, 1.25), rx2[0][0] - 0.02, -24, bus='radio', send=0.3)
add(K.paper(1.2, rng) * 0.6, C('128:dede') + 0.4, -30, bus='room', send=0.2)
for i, d_ in enumerate([0, 0.12, 0.34]):
    add(K.morse_key(rng, True), C('130:saygılarımla') + 0.2 + d_, -20, bus='room', send=0.3)

# =================================================================== 13) BEŞ GÜN: DAYANIŞMA (777–826 s)
add(K.tap_dry(rng), C('132:musluklardan') - 0.1, -18, bus='room', send=0.3, name='musluktan su akmıyor')
t_r = C('133:insanlar')
bed(lambda n: K.rain(n, rng, 0.9), t_r - 0.5, C('133:marketlerdeki') + 0.5, -30, fin=1, fout=1.5, bus='out', send=0.3, name='yağmur')
add(K.bucket_drips(5.0, rng, 6.0), t_r, -26, bus='out', send=0.3)
def slosh(dur):
    n_ = T(dur)
    y_ = svf(white(n_, rng), 300 + 400 * np.abs(smooth_noise(n_, 3, rng)), 2.0, 'bp') * np.abs(smooth_noise(n_, 2.5, rng)) ** 2
    g_ = np.zeros(n_)
    for i_ in range(4):
        L_ = T(0.08); f_ = np.linspace(220, 480, L_)
        b_ = np.sin(2 * np.pi * np.cumsum(f_) / SR) * np.sin(np.linspace(0, np.pi, L_))
        p_ = T(0.2 + i_ * 0.18); g_[p_:p_ + L_] += b_
    return y_ + g_ * 0.5
add(slosh(1.2), C('133:damacanaları') - 0.1, -22, bus='out', send=0.3, name='damacana')
add(K.walla(6, rng, 14, rate=3.0), C('133:kuyruklar'), -34, bus='out', send=0.4)
bed(lambda n: K.fire(n, rng, 1.0), C('134:apartmanların') - 0.3, C('137:diyor') + 1.0, -30, fin=1, fout=2, bus='out', send=0.25, name='odun ateşleri')
add(K.ladle_pot(rng), C('134:kazanda') + 0.2, -18, bus='out', send=0.3, name='kazan + kepçe')
def bubbling(dur):
    n_ = T(dur); y_ = np.zeros(n_); x_ = 0.0
    while x_ < dur - 0.1:
        L_ = T(0.05 + 0.07 * rng.random()); f_ = np.linspace(120 + 80 * rng.random(), 260 + 200 * rng.random(), L_)
        b_ = np.sin(2 * np.pi * np.cumsum(f_) / SR) * np.exp(-np.arange(L_) / (L_ / 3))
        i_ = T(x_); y_[i_:i_ + L_] += b_ * (0.4 + 0.6 * rng.random()); x_ += rng.exponential(0.09)
    return lp(y_, 1500)
add(bubbling(6.0), C('134:mercimek') - 0.3, -24, bus='out', send=0.3, name='çorba fokurduyor')
for i in range(3):
    add(K.switch_click(rng) * 0.5, C('135:telefonuyla') + 0.3 + i * 0.35, -30, bus='out', send=0.2)
def pour(dur):
    n_ = T(dur)
    return svf(white(n_, rng), 500 + 900 * np.linspace(0, 1, n_), 1.5, 'bp') * (0.7 + 0.3 * smooth_noise(n_, 12, rng)) * env(n_, [(0, 0), (0.1, 1), (dur - 0.2, 1), (dur, 0)])
add(pour(1.4), C('136:tabağını') - 0.1, -22, bus='out', send=0.25, name='çorba dökülüyor')
add(K.ladle_pot(rng) * 0.6, C('136:tabağını') + 1.3, -26, bus='out', send=0.3)
add(K.walla(7, rng, 18, rate=2.8), C('138:zor'), -36, bus='out', send=0.5, name='sarılan insanlar')

# =================================================================== 14) ÇATIDA 12:34 (826–936 s)
bed(lambda n: K.wind(n, rng, 0.9, 1.0, 0.1, 200, 800), C('139:ama#2', '139:ama') , 888.0, -37, fin=2, fout=0.3, bus='out', send=0.25, name='çatıda rüzgâr')
rx = seq_of(359)
tone = K.cw_signal([(a - rx[0][0] + 0.02, d_) for a, d_ in rx], rng, f=760, drift=0.004, fade=0.3)
add(K.radio_filter(tone, 1.25) + K.radio_static(len(tone), rng, 0.6) * 0.4, rx[0][0] - 0.02, -22, bus='radio', send=0.3, name='Ankara: yeni mesaj (CW)')
add(K.watch_wind(rng, 12), C('142:yaylar') - 0.2, -22, name='saat kuruluyor')
add(K.watch_ticks(22.0, rng), C('142:öğlen') - 0.4, -22, name='12:34 tik-tak')
add(K.heartbeat(5.5, rng, 84, 118), C('144:kalbin') - 0.3, -16, name='kalp hızlanıyor')
t_o = C('145:ve')
add(K.ship_open(9.0, rng), t_o - 0.2, -12, bus='big', send=0.6, name='GEMİNİN KARNI AÇILIYOR')
add(K.riser(2.5, rng), t_o - 2.4, -24, bus='big', send=0.3)
add(K.swarm(9.0, rng, 24, 210), C('146:küçük') - 0.5, -16, bus='big', send=0.5, name='kovanına dönen arılar')
t_sp = C('149:açıklığın')
add(K.whoosh(4.5, rng, 2600, 300, -0.6, 0.3, 0.75, q=1.5, tone=0.3), t_sp + 0.8, -15, bus='big', send=0.5, name='küre çatıya süzülüyor')
add(K.sphere_hum(6.0, rng, 233) * env(T(6), [(0, 0), (4, 1), (6, 1)])[None, :], t_sp + 1.5, -18, bus='big', send=0.5)
t_run = C('150:merdivenlere') - 0.1
add(K.footsteps(1.0, rng, 4.0, 'concrete', True), t_run, -14, bus='out', send=0.3, name='koşuyorsun')
t_sl = C('150:çarpıyorsun') - 0.35
add(K.door_slam(rng), t_sl, -5, bus='stair', send=0.6, name='KAPI ÇARPIYOR')
add(K.footsteps(1.6, rng, 5.0, 'stairs', True), t_sl + 0.35, -15, bus='stair', send=0.5)
# kapının arkasında: küre ışığı tarıyor (boğuk, sağ-sol), nefes tutma, sızan ışık, fısıltı
n = T(9.0); hum = K.sphere_hum(9.0, rng, 233)
scan = pan(lp(hum.mean(0), 1100), np.sin(np.linspace(0, 2.6 * np.pi, n)) * 0.7)
add(scan, C('151:aralığından') - 0.4, -19, bus='stair', send=0.6, name='ışık çatıyı tarıyor (kapı ardından)')
add(K.breath_in(rng, 0.7), C('151:nefesini') - 0.2, -20, bus='stair', send=0.3, name='nefesini tutuyorsun')
add(K.heartbeat(7.0, rng, 104, 92), C('151:nefesini') + 0.6, -18)
add(K.sphere_hum(9.0, rng, 233, 1.2) * env(T(9), [(0, 0), (3, .8), (6, 1), (8, 1), (9, .7)])[None, :], C('152:ışık') - 0.2, -17, bus='stair', send=0.7, name='ışık basamaklardan süzülüyor')
w = K.whisper(['du', 'yan', 'var', 'mı'], rng, 3.0)
wl = lp(w, 3000)
add(wl, C('154:duyan') - 0.05, -17, bus='stair', send=0.8, name='fısıltı: "duyan var mı"')
add(wl[::-1] * np.linspace(0, 1, len(wl)), C('154:duyan') - 0.05 - len(wl) / SR, -26, bus='stair', send=1.0)
add(K.breath_in(rng, 0.4) * 0.5, C('154:ağzını') + 0.3, -26, bus='stair', send=0.3)
add(K.whoosh(2.5, rng, 700, 180, 0.0, -0.6, 0.25, q=1.4, tone=0.2), C('155:sonra') + 0.3, -20, bus='stair', send=0.6, name='küre geri dönüyor')
add(K.footsteps(4.0, rng, 1.8, 'stairs'), C('156:aşağı') - 0.3, -22, bus='stair', send=0.5, name='aşağı iniş')

# =================================================================== 15) ALTINCI GÜN: PLAN (936–1050 s)
rx = seq_of(406)
tone = K.cw_signal([(a - rx[0][0] + 0.02, d_) for a, d_ in rx], rng, f=740, drift=0.005, fade=0.4)
add(K.radio_filter(tone, 1.25) + K.radio_static(len(tone), rng, 0.6) * 0.4, rx[0][0] - 0.02, -22, bus='radio', send=0.3, name='Ankara: uzun mesaj (CW)')
add(K.paper(1.5, rng), C('159:dede') + 0.3, -28, bus='room', send=0.2)
def metal_clank(g_=1.0):
    return K.click(rng, np.array([310, 690, 1230, 2100, 3300]) * (0.9 + 0.2 * rng.random()), [0.4, 0.3, 0.25, 0.18, 0.12], [1, .8, .6, .4, .3], n_exc=40, dur=0.9, hp_hz=120) * g_
bed(lambda n: K.room_tone(n, rng, 1.5), 951.0, 961.97, -40, fin=0.5, fout=0.5, bus='stair', send=0.6, name='depo')
add(metal_clank(), C('161:tamamen') + 0.3, -16, bus='stair', send=0.6, name='top mekanizması')
def crank(dur):
    n_ = T(dur); y_ = np.zeros(n_); x_ = 0.0
    while x_ < dur - 0.05:
        c_ = K.click(rng, [1700, 2900, 4500], [0.03, 0.02, 0.015], [1, .6, .4], n_exc=10, dur=0.06, hp_hz=700)
        i_ = T(x_); m_ = min(len(c_), n_ - i_); y_[i_:i_ + m_] += c_[:m_]; x_ += 0.07 + 0.01 * rng.random()
    return y_ + bp(white(n_, rng), 400, 2000) * 0.03
add(crank(2.0), C('162:elle') + 0.2, -20, bus='stair', send=0.5, name='el çarkı (nişan)')
add(K.engine(6.0, rng, 34, 'diesel', 0.8), C('163') - 0.3, -18, bus='out', send=0.35, name='eski dizel kamyon')
add(K.engine(4.5, rng, 11, 'tractor', 0.5), C('163:traktörler') - 0.2, -18, p=0.3, bus='out', send=0.35, name='traktör')
add(metal_clank(0.7), C('164:toplardan') + 0.5, -22, bus='stair', send=0.5)
def air_horn(dur):
    n_ = T(dur); t_ = tt(n_)
    y_ = sum(np.tanh(2.5 * np.sin(2 * np.pi * f_ * t_)) for f_ in (233, 277, 349))
    return bp(y_, 180, 4000) * env(n_, [(0, 0), (0.04, 1), (dur - 0.08, 1), (dur, 0)])
add(air_horn(0.9), C('166:ağır') - 0.4, -18, bus='out', send=0.4, name='"Ağır abi" kamyon kornası')
add(K.engine(5.0, rng, 30, 'diesel', 0.7), C('165:onu') , -22, bus='out', send=0.4)
bed(lambda n: K.wind(n, rng, 0.8, 0.8, 0.0, 150, 600) + st(K.engine(n / SR, rng, 13, 'tractor', 0.2) * 0.15), C('166:tarihte') , C('168:ağaçların') + 2.0, -38, fin=1.5, fout=2, bus='out', send=0.4, name='konvoy uzakta')
add(K.switch_click(rng), C('171:bir') + 0.4, -20, bus='room', send=0.2, name='el feneri tıkı')
rx = seq_of(445)
tone = K.cw_signal([(a - rx[0][0] + 0.02, d_) for a, d_ in rx], rng, f=650)
add(K.radio_filter(tone, 1.3), rx[0][0] - 0.02, -23, bus='room', send=0.3, name='uzun yayın (görsel)')
def chain(dur):
    n_ = T(dur); y_ = np.zeros(n_); x_ = 0.0
    while x_ < dur - 0.1:
        c_ = K.click(rng, np.array([3800, 6100, 8400]) * (0.9 + 0.2 * rng.random()), [0.06, 0.05, 0.03], [1, .6, .4], n_exc=4, dur=0.1, hp_hz=2000)
        i_ = T(x_); m_ = min(len(c_), n_ - i_); y_[i_:i_ + m_] += c_[:m_] * rng.random(); x_ += rng.exponential(0.04)
    return y_
add(chain(0.9), C('173:dede') + 1.2, -24, bus='room', send=0.25, name='cep saati zinciri')
def flag(dur):
    n_ = T(dur)
    flap = np.abs(smooth_noise(n_, 7, rng)) ** 1.5
    return bp(white(n_, rng), 300, 4000) * flap * (0.6 + 0.4 * smooth_noise(n_, 0.5, rng))
add(K.widen(flag(5.5), rng, 1.0, 9), C('176:gençken') - 0.2, -27, bus='out', send=0.4, name='bayrak dalgalanıyor')

# =================================================================== 16) ÇATI VE KUŞATMA (1050–1106 s)
t_roof = C('177:çatıya')
bed(lambda n: K.wind(n, rng, 1.0, 1.1, 0.15, 180, 800), t_roof - 0.5, C('185:yedinci') - 0.3, -35, fin=2, fout=0.2, bus='out', send=0.3, name='gece çatı rüzgârı')
t_morse_end = C('185:susuyor')
vis = seq_of(459) + seq_of(470) + seq_of(472) + seq_of(474)
fill = human_morse(t_roof + 2.0, t_morse_end - 0.2, 0.095, text='CQ DE TA1RD QRV QRV ISTANBUL K ')
fill = [e for e in fill if not any(abs(e[0] - v[0]) < 0.35 for v in vis)]
for (a0, a1) in [(1060.28, 1063.22), (1092.25, 1093.68), (1094.7, 1096.23), (1099.44, 1103.17)]:
    fill = [e for e in fill if not (a0 - 0.1 <= e[0] <= a1)]
allm = sorted([e for e in vis + fill if e[0] < t_morse_end - 0.05])
key_events(allm, 'room', -20, -27, 640, muffled=True)
LOG.append((allm[0][0], 'MORS: aşağıdan, perdenin arkasından (boğuk)'))
LOG.append((t_morse_end, 'Mors anahtarının sesi susuyor'))
add(K.watch_ticks(6.0, rng), C('179:bir') - 0.3, -28)
for i, spec in enumerate(['180:gemiden', '180:sonra', '180:sonra#2', '180:üç', '180:üç+0.4']):
    add(K.whoosh(3.2, rng, 2400, 280, rng.uniform(-.9, .9), rng.uniform(-.3, .3), 0.65, q=1.4, tone=0.3), C(spec) - 0.2, -18 - i * 0.5, bus='big', send=0.5, name='küreler ayrılıyor' if i == 0 else '')
add(K.sphere_hum(12.0, rng, 220) * env(T(12), [(0, 0), (4, 1), (12, 1)])[None, :], C('180:hepsi') - 0.5, -18, bus='big', send=0.5)
for i, spec in enumerate(['181:duyan', '182:beşinci', '182:dördüncü', '183']):
    w_ = lp(K.whisper(['du', 'yan', 'var', 'mı'], rng, 3.2), 3200 - i * 450)
    add(w_, C(spec) - 0.05, -18 - i * 1.5, bus='stair', send=0.9, name='merdivende fısıltı ' + spec)
add(K.sphere_hum(10.0, rng, 196, 1.3) * env(T(10), [(0, 0), (3, 1), (10, 1)])[None, :], C('182') - 0.3, -19, bus='stair', send=0.7, name='ışık kat kat iniyor')
add(K.impact(rng, 1.0), C('183') + 0.05, -13, bus='big', send=0.5, name='"dedenin katı"')
add(K.energy_surge(3.2, rng), C('184:penceresi') - 2.4, -9, bus='big', send=0.6, name='PENCERE BEMBEYAZ IŞIKLA DOLUYOR')
add(K.heartbeat(4.0, rng, 70, 60), t_morse_end + 0.6, -22)

# =================================================================== 17) YEDİNCİ GÜN: İŞARET VE ATEŞ (1106–1205 s)
t7 = C('185:sabah')
bed(lambda n: K.wind(n, rng, 0.9, 0.9, 0.1, 200, 700), t7 - 0.2, 1143.15, -36, fin=2, fout=0.25, bus='out', send=0.25, name='şafak rüzgârı')
add(K.watch_ticks(1143.0 - t7, rng), t7, -23, name='cep saati (elde)')
add(K.ship_open(8.0, rng), C('190:geminin') - 0.3, -10, bus='big', send=0.6, name='mavi çizgi beliriyor')
add(K.curtain(0.6, rng) * 0.5, C('190:feneri') - 0.2, -26, bus='out', send=0.2)
for a, L in SIG7:
    add(K.switch_click(rng), a - 0.01, -14, bus='out', send=0.2)
    add(K.switch_click(rng) * 0.7, a + L, -17, bus='out', send=0.2)
LOG.append((SIG7[0][0], 'FENER: kısa-uzun-kısa-uzun (görselle senkron)'))
t_sil = 1143.17
t_fire = 1145.92
add(K.breath_in(rng, 0.5), t_sil + 0.2, -26)
LOG.append((t_sil, 'bir saniyelik sessizlik'))
# BARAJ ATIŞI: üç batarya (Çamlıca sol, Kandilli orta-sağ, Ulus sağ), uzak yankılı
t_bend = C('196:metalin') + 1.0
batteries = [(-0.75, 1.6, 0.5), (0.35, 2.4, 0.42), (0.8, 3.2, 0.55), (-0.3, 4.0, 0.6), (0.1, 5.0, 0.48)]
cnt = 0
for bi, (p_, dist, per) in enumerate(batteries):
    t = t_fire + bi * 0.07 + rng.random() * 0.05
    while t < t_bend:
        fade_k = 1.0 if t < t_bend - 6 else max(0.0, (t_bend - t) / 6)
        if rng.random() < fade_k:
            g_ = -6 - dist * 3.0 + rng.normal(0, 1)
            add(K.cannon(rng, dist, 1.1 if bi < 2 else 0.9), t, g_, p=p_ + rng.normal(0, .05), bus='big', send=0.55, dry=0.9)
            cnt += 1
            if rng.random() < 0.25:
                add(K.shell_whistle(1.4, rng, True), t + 0.05, g_ - 12, p=p_ * 0.5, bus='big', send=0.4)
        t += per * (0.85 + 0.3 * rng.random()) + (0.6 if rng.random() < 0.08 else 0)
LOG.append((t_fire, 'BARAJ ATEŞİ başlıyor (%d atış)' % cnt))
# gökte şarapnel patlamaları
for i in range(26):
    t = t_fire + 1.4 + rng.random() * (t_bend - t_fire - 4)
    add(K.flak(rng, rng.uniform(2.5, 4.5)), t, -18 + rng.normal(0, 2), p=rng.uniform(-.7, .7), bus='big', send=0.6, dry=0.7)
# geminin içinde patlamalar
t_e = C('194:ilk')
add(K.explosion(rng, 2.2, 2.0), t_e - 0.1, -3, p=0.1, bus='big', send=0.6, name='İLK PATLAMA (gemi içi)')
add(K.explosion(rng, 1.8, 2.2), C('194:sonra') - 0.1, -5, p=-0.2, bus='big', send=0.6, name='ikinci patlama')
t_many = C('195:sonra', '195')
for i in range(16):
    add(K.explosion(rng, rng.uniform(0.8, 1.6), rng.uniform(2.0, 3.5)), t_many + i * 0.32 + rng.random() * 0.25, -8 - rng.random() * 5, p=rng.uniform(-.8, .8), bus='big', send=0.6, name='onlarca patlama' if i == 0 else '')
add(K.fire(T(7.0), rng, 1.5) * env(T(7), [(0, 0), (0.3, 1), (5, .7), (7, 0)])[None, :], C('195:geminin') - 0.2, -14, bus='big', send=0.5, name='turuncu ışık fışkırıyor (alev)')
add(K.ship_horn(6.0, rng, 38), C('195:dev') - 0.3, -10, bus='big', send=0.7, name='dev gövde yana yatıyor')
add(K.metal_scream(7.5, rng), C('196:metalin') - 0.6, -2, bus='big', send=0.7, name='METALİN ÇIĞLIĞI')
# telefon titriyor, şehir uyanıyor
add(K.phone_vibrate(3.4, rng, surface='pocket'), C('196:cebinde') + 0.2, -12, name='cebinde telefon titriyor')
add(K.phone_vibrate(2.0, rng, 0.3, 0.2, 'pocket'), C('197:telefonun') + 0.2, -12)
t_l = C('197:sokak')
for i in range(6):
    d_ = 1 + i * 0.6
    add(K.lights_on(rng, d_), t_l + 0.2 + i * 0.35 + rng.random() * 0.2, -16 - i * 2.2, p=rng.uniform(-.8, .8), bus='out', send=0.5, name='sokak lambaları yanıyor' if i == 0 else '')
add(lp(K.car_alarm(16.0, rng), 2800), C('198:bir') + 0.2, -29, p=-0.5, bus='out', send=0.8, dry=0.45, name='araba alarmı')
add(lp(K.car_alarm(12.0, rng), 2000), C('198:sonra') + 0.3, -34, p=0.6, bus='out', send=0.9, dry=0.35)
bed(lambda n: K.city_hush(n, rng, 1.0), C('198:bütün') - 0.5, 1261.0, -36, fin=4, fout=3, bus='out', send=0.3, name='şehir uyanıyor')
for i in range(8):
    add(K.gs_note(1, 123, int(rng.integers(54, 64)), 0.2, vel=100), C('198:bütün') + 1 + rng.random() * 5, -26 - rng.random() * 6, p=rng.uniform(-1, 1), bus='out', send=0.6, dry=0.5)
t_p = C('199:milyon')
for i in range(22):
    add(K.phone_vibrate(1.2 + rng.random(), rng, 0.3 + 0.2 * rng.random(), 0.2, 'table' if rng.random() < .5 else 'pocket'), t_p + rng.random() * 6, -18 - rng.random() * 8, p=rng.uniform(-1, 1), bus='room', send=0.4, name='15 milyon telefon titriyor' if i == 0 else '')
def notif(f0):
    p_ = np.array([1, 2.0, 3.01]) * f0
    return K.click(rng, p_, [0.4, 0.25, 0.15], [1, .4, .2], n_exc=8, dur=0.6, hp_hz=300)
for i in range(30):
    add(notif(rng.choice([880, 988, 1175, 1319, 1568, 784])), t_p + 1.0 + rng.random() * 9, -27 - rng.random() * 8, p=rng.uniform(-1, 1), bus='room', send=0.4)
add(K.phone_vibrate(1.6, rng, 0.2, 0.12, 'table'), C('200:aile') , -18, name='aile grubu bildirimleri')

# =================================================================== 18) ZAFER VE KAYIP (1205–1262 s)
add(K.ship_horn(9.0, rng, 40), C('202') - 0.3, -10, bus='big', send=0.7, name='gemi yaralı hayvan gibi yükseliyor')
n = T(10.0); dep = K.ship_hum(n, rng, 33) * env(n, [(0, 1), (6, .5), (10, 0)])[None, :]
add(lp(dep, 400), C('202:yükseliyor#2', '202') , -14, bus='big', send=0.6)
add(K.explosion(rng, 2.5, 6.0), C('203:kayboluyor') + 0.3, -14, bus='big', send=0.8, name='gözden kayboluyor (uzak gümbürtü)')
add(K.swarm(6.0, rng, 16, 250) * env(T(6), [(0, 1), (6, 0)])[None, :], C('203:anadoludaki') + 0.5, -22, bus='big', send=0.5)
add(K.whoosh(3.0, rng, 500, 3500, -0.5, 0.8, 0.45, q=1.2), C('203:havalanıp', '204') - 0.5, -20, bus='big', send=0.5)
t_c = C('204:bütün')
t_c1 = C('206:ama#2', '206:koşuyorsun')
bed(lambda n: K.cheer(n / SR, rng, 34), t_c, t_c1 + 0.8, -26, fin=1.0, fout=1.4, bus='out', send=0.5, chunk=20, name='KUTLAMA: tezahürat + alkış')
for i in range(12):
    add(K.car_horn(0.25 + 0.6 * rng.random(), rng, rng.choice([380, 420, 470, 520])), t_c + 0.5 + rng.random() * (t_c1 - t_c - 1), -22 - rng.random() * 8, p=rng.uniform(-1, 1), bus='out', send=0.6, dry=0.7)
add(K.gs_note(0, 126, 60, 6.0), t_c + 1.0, -20, bus='out', send=0.4)
def washer(dur):
    n_ = T(dur); t_ = tt(n_)
    water = svf(white(n_, rng), np.full(n_, 900.0), 0.8, 'bp') * env(n_, [(0, 0), (0.4, 1), (dur, .8)]) * 0.6
    motor = sum(np.sin(2 * np.pi * f_ * t_) for f_ in (50, 100, 150)) * 0.05
    y_ = water + motor; c_ = K.switch_click(rng); y_[:len(c_)] += c_
    return y_
add(washer(2.4), C('206:çamaşır') + 0.1, -24, bus='room', send=0.3, name='çamaşır makinesi :)')
add(K.footsteps(3.0, rng, 4.8, 'stairs', True), C('206:koşuyorsun') + 0.1, -13, bus='stair', send=0.5, name='merdivenden koşuyorsun')
add(K.door_creak(1.6, rng), C('207:kapı') - 0.1, -16, bus='room', send=0.4, name='kapı açık')
t_room = C('207:oda')
bed(lambda n: K.wind(n, rng, 1.0, 1.3, 0.9, 200, 900), t_room - 0.5, 1261.3, -30, fin=1.5, fout=2.5, bus='room', send=0.4, name='camı olmayan pencereden rüzgâr')
def cooling(dur):
    n_ = T(dur); y_ = np.zeros(n_); x_ = 0.3
    while x_ < dur - 0.2:
        c_ = K.click(rng, np.array([3100, 5200, 7600]) * (0.85 + 0.3 * rng.random()), [0.05, 0.04, 0.03], [1, .6, .4], n_exc=4, dur=0.1, hp_hz=1500)
        i_ = T(x_); m_ = min(len(c_), n_ - i_); y_[i_:i_ + m_] += c_[:m_] * (0.3 + 0.7 * rng.random()); x_ += rng.exponential(0.8)
    return y_ + K.radio_static(n_, rng, 0.15, 0.2, 2.5, 0) * 0.5
add(cooling(6.0), C('207:tersizin') - 0.2, -26, bus='room', send=0.3, name='telsiz lambaları hâlâ sıcak (soğuyan metal)')
add(K.click(rng, [2400, 3900], [0.02, 0.015], [1, .6], n_exc=8, dur=0.08, hp_hz=1000), C('209:katlanmış') + 0.2, -24, bus='room', send=0.3, name='katlanmış gözlük')

# =================================================================== 19) ONUNCU GÜN (1261–1327 s)
bed(lambda n: K.city_hush(n, rng, 0.9) + K.walla(n / SR, rng, 16, rate=3.0) * 0.3, 1262.2, 1326.8, -38, fin=2, fout=2, bus='out', send=0.3, name='şehir toparlanıyor')
add(K.engine(8.0, rng, 30, 'diesel', 0.7), C('213:anadoludan') - 0.3, -20, p=-0.4, bus='out', send=0.4, name='tır konvoyu')
add(K.engine(7.0, rng, 28, 'diesel', 0.7), C('213:anadoludan') + 1.5, -22, p=0.4, bus='out', send=0.4)
def air_brake():
    n_ = T(0.8)
    return hp(white(n_, rng), 2500) * env(n_, [(0, 0), (0.02, 1), (0.8, 0)], 1.5)
add(air_brake(), C('213:anadoludan') + 3.0, -22, bus='out', send=0.4)
add(K.engine(6.0, rng, 50, 'diesel', 0.3), C('214:jeneratör') - 0.2, -26, bus='out', send=0.3, name='jeneratör')
add(K.coins(rng, 5), C('215:mahallede') + 1.5, -26, bus='room', send=0.3)
add(K.wood_thump(rng, 150), C('216:kapatıyor') + 0.1, -16, bus='room', send=0.3, name='veresiye defteri kapanıyor')
bed(lambda n: K.wind(n, rng, 1.0, 1.0, 0.3, 250, 1100), C('217:madenlerde') - 0.3, C('219:sosyal') , -34, fin=1, fout=1.5, bus='out', send=0.3, name='boş maden çukurları')

# =================================================================== 20) KAPANIŞ: RİTİM (1327–1438 s)
t_rad = C('222:lambalı')
add(K.knob_click(rng), t_rad - 0.3, -18, bus='room', send=0.3, name='radyo açılıyor')
n = T(4.0); add(K.radio_static(n, rng, 1.0, 1.0, 2.0) * env(n, [(0, 0), (3, 1), (4, 1)]), t_rad, -30, bus='radio', send=0.3)
st_auto = [(t_rad + 4, -2), (1338, 0), (1356, -2), (1370, -1), (1388, -2), (1402, -3), (1411, -6), (1430, -6), (END - 0.5, -30)]
bed(lambda n: K.radio_static(n, rng, 1.0, 1.0, 1.0), t_rad + 4, END - 0.3, -36, fin=0.5, fout=2.5, bus='radio', send=0.3, auto=st_auto, name='gece radyo cızırtısı (sona kadar)')
# 'morz' — dedenin ritmi (insan eli: hafif düzensiz)
first = seq_of(590) + seq_of(591)
tone = K.cw_signal([(a - first[0][0] + 0.02, d_) for a, d_ in first], rng, f=640, drift=0.003)
add(K.radio_filter(tone, 1.25), first[0][0] - 0.02, -16, bus='radio', send=0.35, name='MORS — "TIK TAK TIK TIK"')
add(K.heartbeat(2.6, rng, 70, 50), C('224:kalbin') - 0.2, -18, name='kalbin duruyor')
p226 = seq_of(597)
tone = K.cw_signal([(a - p226[0][0] + 0.02, d_) for a, d_ in p226], rng, f=640, drift=0.003)
add(K.radio_filter(tone, 1.25), p226[0][0] - 0.02, -19, bus='radio', send=0.35, name='dedenin ritmi')
# KORKMA İYİYİM: her harf, kâğıtta belirmeden hemen önce (insan ritmi)
dec = [x for x in MORSE_SHOTS if x['i'] == 598][0]
lets = []
for L in dec['lines']:
    for ch, at in zip(L['text'], L['at']):
        lets.append((ch, at))
for ch, at in lets:
    code, _ = K.morse_code_times(ch, 0.0, 0.085, 0.15, rng)
    dur = code[-1][0] + code[-1][1]
    t0_ = at - dur - 0.08
    tone = K.cw_signal([(a + 0.02, d_) for a, d_ in code], rng, f=640, drift=0.003)
    add(K.radio_filter(tone, 1.25), t0_ - 0.02, -17, bus='radio', send=0.35)
LOG.append((lets[0][1], 'KORKMA İYİYİM — harf harf (dedenin eli)'))
p229 = seq_of(602)
tone = K.cw_signal([(a - p229[0][0] + 0.02, d_) for a, d_ in p229], rng, f=640, drift=0.003)
add(K.radio_filter(tone, 1.25), p229[0][0] - 0.02, -19, bus='radio', send=0.35)
# makine ritmi: kusursuz, soğuk, hafif metalik
def cold_cw(times, f=820):
    tone = K.cw_signal([(a - times[0][0] + 0.02, d_) for a, d_ in times], rng, f=f, clean=True)
    tm_ = tt(len(tone))
    ring = tone * (0.75 + 0.25 * np.sin(2 * np.pi * 1.5 * f * tm_))
    return K.radio_filter(ring, 1.2)
mach = seq_of(604)
mach = [(a, 0.3 if d_ > 0.2 else 0.1) for a, d_ in mach]
add(cold_cw(mach), mach[0][0] - 0.02, -16, bus='radio', send=0.25, name='MAKİNE RİTMİ (soğuk, kusursuz)')
dec = [x for x in MORSE_SHOTS if x['i'] == 609][0]
lets = []
for L in dec['lines']:
    for ch, at in zip(L['text'], L['at']):
        lets.append((ch, at))
for ch, at in lets:
    code, _ = K.morse_code_times(ch, 0.0, 0.07, 0.0, None)
    dur = code[-1][0] + code[-1][1]
    add(cold_cw([(a + 0.02, d_) for a, d_ in code]), at - dur - 0.06, -15, bus='radio', send=0.25)
LOG.append((lets[0][1], 'GERİ DÖNECEZ — makine ritmi'))
add(K.impact(rng, 1.0), C('233:z', '233') + 0.2, -12, bus='big', send=0.5)
add(K.phone_vibrate(3.5, rng, 0.45, 0.3, 'table'), C('235:telefonun') - 0.1, -12, name='masadaki telefon titriyor')
add(K.phone_vibrate(1.4, rng, 0.45, 0.3, 'table'), C('236') + 0.2, -15)
w = K.whisper(['du', 'yan', 'var', 'mı'], rng, 3.0)
add(lp(w, 4000), C('236:duyan') - 0.05, -14, bus='stair', send=0.9, name='SON: "duyan var mı"')
add(lp(w, 4000)[::-1] * np.linspace(0, 1, len(w)), C('236:duyan') - 0.05 - len(w) / SR, -22, bus='stair', send=1.0)
add(K.impact(rng, 1.2), C('236:duyan') + 0.9, -9, bus='big', send=0.6)

# =================================================================== MİKS
print('olaylar yerleşti %.1fs' % (time.time() - t_start))
for k, b in BUS.items():
    wet = convolve(b.astype(np.float64), IR[k])[:, :N]
    M += (wet * {'room': 0.55, 'stair': 0.75, 'out': 0.6, 'big': 0.7, 'radio': 0.4}[k]).astype(np.float32)
    print('yankı', k, '%.1fs' % (time.time() - t_start))
del BUS
x = hp(M.astype(np.float64), 22, 2)
np.save(os.path.join(OUT, 'sfx_raw.npy'), x.astype(np.float32))
with open(os.path.join(OUT, 'sfx_log.txt'), 'w') as f:
    for t, nm in sorted(LOG):
        if nm:
            f.write('%02d:%05.2f  %s\n' % (int(t // 60), t % 60, nm))
print('bitti %.1fs' % (time.time() - t_start))
