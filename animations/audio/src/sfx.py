"""SFX track: location ambience beds + events synced to every shot of the animation."""
import math, time
import numpy as np
from sfxlib import *
from tl import T, TOTAL, SHOTS, shots

A = Bus(TOTAL)      # dry
RM = Bus(TOTAL)     # small room send
HL = Bus(TOTAL)     # big space (tunnel / cave / hall) send
OUT = Bus(TOTAL)    # outdoor send

def jhash(n):
    v = math.sin(n * 127.1 + 311.7) * 43758.5453123
    return v - math.floor(v)

def put(sig, t, g=-12, p=0.0, room=0.0, hall=0.0, out=0.0):
    if sig is None or t >= TOTAL or t < -5: return
    G = db(g)
    A.add(sig, t, G, p)
    if room: RM.add(sig, t, G * room, p)
    if hall: HL.add(sig, t, G * hall, p)
    if out: OUT.add(sig, t, G * out, p)

def bed(t0, t1, sig, g=-20, room=0.0, hall=0.0, out=0.0, fi=0.8, fo=0.8):
    d = t1 - t0
    if d <= 0.05: return
    s = sig[:, :int(d * SR)]
    s = s * env_ar(s.shape[1], min(fi, d / 3), min(fo, d / 3), 1.0)
    put(s, t0, g, 0, room, hall, out)

# ------------------------------------------------------------------ transitions (mirror of the HTML)
TRS = ['punch', 'glitch', 'flash', 'whip', 'punch', 'zoom', 'glitch', 'cut', 'punch', 'whip']
def transition_sfx(i, s):
    tr = s['p'].get('tr') or TRS[int(jhash(i * 1.37) * len(TRS))]
    t = s['t0']
    if tr in ('punch', 'zoom'):
        put(whoosh(0.35, 250, 1800, 1.0, rev=True), t - 0.33, -26, rnd(-.3, .3))
        put(lp(one('bdrum_soft'), 400)[:, :int(.4 * SR)], t, -27)
    elif tr == 'whip':
        put(whoosh(0.32, 500, 5000, 1.0), t - 0.16, -22, 0)
    elif tr == 'flash':
        put(whoosh(0.5, 1200, 7000, 0.8, rev=True), t - 0.48, -27)
        put(one('marktree')[:, :int(1.5 * SR)] * env_ar(int(1.5 * SR), 0.0, 1.2), t, -34, 0.2, out=0.3)
    elif tr == 'glitch':
        st = static(0.28, 1.0, 2.0) * ((np.arange(int(.28 * SR)) // int(SR / 28)) % 2)[None, :]
        put(st, t, -30, rnd(-.3, .3))
    elif tr == 'redflash':
        put(lp(one('bdrum'), 900)[:, :int(.8 * SR)], t, -18); put(whoosh(0.4, 200, 1200, 1, rev=True), t - 0.38, -24)

# ------------------------------------------------------------------ ambience map
def ambience():
    tE = T('gözlerini açıyorsun'); tD1 = T('birinci gün'); tFB = T('hani her sabah'); tB = T('ama bu sabah tek')
    bed(tE, tFB, room_tone(tFB - tE + 1, 50, 1.0), -26, room=0.2)
    bed(tE, tFB, city_far(tFB - tE + 1), -36)
    bed(tFB, tB, birds(tB - tFB + 1, 1.2) + wind(tB - tFB + 1, 0.2, 700), -22, out=0.3)
    bed(tB, tD1 + 1, room_tone(tD1 - tB + 2, 50), -26, room=0.2)
    # Day 1-3 apartment + city chaos outside
    tM = T('market kuyruğu'); tY = T('ama asıl panik'); tOg = T('öğleden sonra kapındaki'); tD2 = T('ikinci gün')
    bed(tD1, tM, room_tone(tM - tD1 + 1, 50), -27, room=0.2)
    bed(tD1, tM, city_far(tM - tD1 + 1, 1.3), -35)
    bed(tM, tY, city_far(tY - tM + 1, 2.0) + crowd_walla(tY - tM + 1, 16, 0.6), -32, out=0.3)
    bed(tY, tOg, city_far(tOg - tY + 1, 2.5), -32, out=0.3)
    for k in range(int((tOg - tY) / 2.2)):
        put(engine(2.0, rnd(700, 900), rnd(700, 950), 4, 1.0), tY + k * 2.2 + rnd(0, 1), -40, rnd(-.8, .8), out=0.2)
    bed(tOg, tD2, room_tone(tD2 - tOg + 1, 50), -25, room=0.2)
    tAk = T('akşamüstü telefonun'); tBal = T('çünkü o gece balkona'); tD3 = T('üçüncü gün')
    bed(tD2, tBal, room_tone(tBal - tD2 + 1, 50), -27, room=0.2)
    bed(tD2, tBal, city_far(tBal - tD2 + 1, 1.0), -33)
    bed(tBal, tD3, wind(tD3 - tBal + 1, 0.5, 450) + city_far(tD3 - tBal + 1, 1.5), -28, out=0.3)
    tTem = T('15 temmuz'); tIse = T('ama işe gidip'); tOrd = T('ama ordu bir adım'); tGec = T('ve o gece ilk kez'); tEl = T('elektrikler gidiyor'); tD4 = T('birinci gün', after=tEl)
    bed(tD3, tTem, room_tone(tTem - tD3 + 1, 50), -27, room=0.2)
    bed(tTem, tIse, wind(tIse - tTem + 1, 0.6, 500) + waves(tIse - tTem + 1, 0.5), -28, out=0.3)
    bed(tIse, tOrd, waves(tOrd - tIse + 1, 1.0) + wind(tOrd - tIse + 1, 0.4, 600), -25, out=0.3)
    bed(tOrd, tGec, wind(tGec - tOrd + 1, 0.8, 520), -26, out=0.3)
    bed(tGec, tEl + 0.4, city_far(tEl - tGec + 1.5, 1.2) + wind(tEl - tGec + 1.5, 0.3, 400), -26, out=0.3)
    for k in range(int((tEl - tGec) * 0.8)):
        put(gunshot(rnd(0.75, 1.0)), tGec + rnd(0, tEl - tGec), -30, rnd(-.9, .9), out=0.4)
    # Day 4 blackout: the silence
    tEz = T('sabah ezanı okunmadı'); tOgs = T('öğleden sonra kapın çalıyor'); tDon = T('ama toplantıdan dönerken'); tD5 = T('beşinci gün')
    bed(tD4, tEz, wind(tEz - tD4 + 1, 0.35, 380), -27, out=0.4, fi=3)
    for k in range(4): put(distant(one('siren') * 0.7, 0.85), tD4 + 6 + k * 6 + rnd(0, 2), -28, rnd(-.8, .8), out=0.6)
    bed(tEz, tOgs, wind(tOgs - tEz + 1, 0.3, 420) + birds(tOgs - tEz + 1, 0.25, 0.6), -27, out=0.4)
    bed(tOgs, tDon, room_tone(tDon - tOgs + 1, 50, 0.8) + fire(tDon - tOgs + 1, 0.15), -27, room=0.3)
    bed(tDon, tD5, room_tone(tD5 - tDon + 1, 50) + wind(tD5 - tDon + 1, 0.15, 900), -25, hall=0.3)
    # Day 5
    tAma = T('ama önce aklına'); tTit = T('titrek bir el'); tNot = T('notu cebine'); tKos = T('koşuyorsun'); tKar = T('karanlıkta nefes'); tYani = T('yani türkiye nin sokaklarındaki'); tBak = T('bakkaldan su'); tOgu = T('o günden sonra'); tVg = T('ve gece mum ışığında'); tD6 = T('altıncı gün')
    bed(tD5, tAma, room_tone(tAma - tD5 + 1, 50), -26, room=0.2)
    bed(tAma, tTit, wind(tTit - tAma + 1, 0.5, 1100), -27, room=0.4)
    bed(tTit, tNot, room_tone(tNot - tTit + 1, 50, 0.7), -28, room=0.3)
    bed(tNot, tKos, wind(tKos - tNot + 1, 0.8, 500) + city_far(tKos - tNot + 1, 0.6), -22, out=0.4)
    for k in range(5): put(one('crow') * 0.8, tNot + 2 + k * 4.3 + rnd(0, 2), -24, rnd(-.8, .8), out=0.6)
    bed(tKar, tYani, room_tone(tYani - tKar + 1, 50, 0.6), -28, room=0.3)
    bed(tBak, tVg, wind(tVg - tBak + 1, 0.5, 600), -26, out=0.3)
    bed(tVg, tD6 + 1, room_tone(tD6 - tVg + 2, 50, 0.8) + fire(tD6 - tVg + 2, 0.1), -27, room=0.3)
    # Day 6 radio night -> Cappadocia narration
    tKap = T('kapadokyada yerin altında'); tKb = T('karar belli'); tD7 = T('yedinci gün')
    bed(tD6, tKap, room_tone(tKap - tD6 + 1, 50, 0.8), -28, room=0.3)
    bed(tKap, tKb, wind(tKb - tKap + 1, 0.35, 650) + drips(tKb - tKap + 1, 0.4, 0.8), -27, hall=0.6)
    bed(tKb, tD7, room_tone(tD7 - tKb + 1, 50, 0.8), -28, room=0.3)
    # Day 7 garage -> road
    tUc = T('üçüncü denemede'); tSim = T('şimdi asıl sorun'); tTem2 = T('temde kocaeli'); tOto = T('otoyola hiç'); tTor = T('torpidoda kemal'); tKoy = T('köy yollarından'); tSah = T('şahin in en büyük'); tBoz = T('bozulursa da'); tAmo = T('ama önünde anadolu'); tD8 = T('sekizinci gün')
    bed(tD7, tUc, room_tone(tUc - tD7 + 1, 60, 0.9) + drips(tUc - tD7 + 1, 0.5, 0.7), -25, hall=0.5)
    bed(tUc + 0.5, tTem2, engine(tTem2 - tUc + 1, 1700, 1500, 4, 0.9, seed=3), -26, room=0.1)
    bed(tTem2, tOto, wind(tOto - tTem2 + 1, 0.5, 500) + city_far(tOto - tTem2 + 1, 2.0), -29, out=0.4)
    for k in range(int((tOto - tTem2) * 1.2)): put(horn(rnd(.25, .8), (rnd(380, 450), rnd(470, 540))), tTem2 + rnd(0, tOto - tTem2), -32, rnd(-.9, .9), out=0.5)
    car = engine(tAmo - tOto + 2, 2200, 2400, 4, 1.0, rough=0.05, seed=4)
    bed(tOto, tD8 + 0.2, lp(car, 900)[:, :int((tD8 - tOto + 1) * SR)], -27)
    bed(tOto, tAmo, lp(noise(tAmo - tOto + 1, 'pink'), 700) * 0.5, -32)
    bed(tKoy, tSah, birds(tSah - tKoy + 1, 1.0), -24, out=0.4)
    bed(tSah, tBoz, crickets(tBoz - tSah + 1, 6), -26, out=0.3)
    bed(tBoz, tAmo, birds(tAmo - tBoz + 1, 0.6), -26, out=0.3)
    bed(tAmo, tD8 + 0.5, wind(tD8 - tAmo + 1.5, 1.0, 380), -22, out=0.4)
    # Day 8 tunnel
    tGy = T('gece yarısı bolu'); tVto = T('ve tam o anda'); tKon = T('kontak şahin çalışıyor'); tTc = T('tünelden çıktığında'); tD9 = T('dokuzuncu gün')
    bed(tD8, tGy, wind(tGy - tD8 + 1, 0.8, 360), -23, out=0.4)
    eng = engine(tVto - tGy + 4, 850, 820, 4, 1.0, seed=5)
    bed(tGy, tVto + 1.2, eng, -29, hall=0.5)
    bed(tGy, tKon, lp(noise(tKon - tGy + 1, 'brown'), 120) * 0.8 + drips(tKon - tGy + 1, 0.35), -27, hall=0.8)
    bed(tKon, tTc, engine(tTc - tKon + 1, 2500, 4200, 4, 1.1, seed=6), -24, hall=0.4)
    bed(tTc, tD9, birds(tD9 - tTc + 1, 1.0) + lp(engine(tD9 - tTc + 1, 2000, 2000, 4, 1.0, seed=7), 700), -26, out=0.4)
    # Day 9 steppe
    tBen = T('benzin bitmek'); tBir = T('bir akaryakıt istasyonu'); tD10 = T('onuncu gün')
    bed(tD9, tBen, wind(tBen - tD9 + 1, 0.6, 450) + lp(engine(tBen - tD9 + 1, 2100, 2100, 4, 0.8, seed=8), 700), -26, out=0.3)
    bed(tBir, tD10, wind(tD10 - tBir + 1, 0.7, 520) + crickets(tD10 - tBir + 1, 4, 0.6), -25, out=0.4)
    # Day 10 village
    tSk = T('sonra seni köy kahvesine'); tIs = T('işte burada türkiye'); tD11 = T('on birinci gün')
    bed(tD10, tSk, wind(tSk - tD10 + 1, 0.45, 600) + birds(tSk - tD10 + 1, 0.8), -25, out=0.4)
    for k in range(int((tSk - tD10) / 4)): put(one('nepal') * 0.5, tD10 + 1 + k * 4 + rnd(0, 2), -34, rnd(-.8, .8), out=0.6)
    bed(tSk, tIs, room_tone(tIs - tSk + 1, 50, 0.6) + fire(tIs - tSk + 1, 0.3) + crowd_walla(tIs - tSk + 1, 6, 0.4), -25, room=0.4)
    bed(tIs, tD11, wind(tD11 - tIs + 1, 0.4, 600) + birds(tD11 - tIs + 1, 0.6), -28, out=0.4)
    # Day 11 arrival
    tKoy2 = T('köyüne giriyorsun'); tNin = T('ninenin evine koşuyorsun'); tKil = T('kilere giriyorsun'); tKap3 = T('kapağı açıyorsun'); tSon = T('sonunda kayaya'); tD30 = T('otuzuncu gün')
    bed(tD11, tKoy2, wind(tKoy2 - tD11 + 1, 0.4, 600) + birds(tKoy2 - tD11 + 1, 0.9), -25, out=0.4)
    bed(tKoy2, tNin, wind(tNin - tKoy2 + 1, 1.0, 480), -21, out=0.5)
    put(creak(1.6, 15, 40, (300, 520, 900)), tKoy2 + 3, -28, 0.5, out=0.5)
    bed(tNin, tKil, room_tone(tKil - tNin + 1, 50, 0.7) + wind(tKil - tNin + 1, 0.25, 900), -26, room=0.4)
    bed(tKil, tKap3, room_tone(tKap3 - tKil + 1, 50, 0.6), -28, room=0.4)
    bed(tKap3, tSon, lp(noise(tSon - tKap3 + 1, 'brown'), 150) + drips(tSon - tKap3 + 1, 0.6), -25, hall=0.9)
    # underground life
    tKar2 = T('karadenizde insanlar'); tRb = T('radyoya bağlanan'); tDog = T('doğu nun dağ köyleri'); tVo = T('ve o akşam yukarıdan'); tAm = T('ama ses havalandırma'); tIst3 = T('istanbulda o ses kesildiğinde'); tSim3 = T('şimdi bir şeyin'); tCeb = T('cebinden kemal'); tOsk = T('o sabah kapıyı açsaydın'); tPek = T('peki sen o sabah')
    hall_bed = crowd_walla(tKar2 - tSon + 1, 18, 0.5) + drips(tKar2 - tSon + 1, 0.4) + bp(noise(tKar2 - tSon + 1, 'pink'), 2000, 6000) * 0.02
    bed(tSon, tKar2, hall_bed, -27, hall=0.8)
    bed(tKar2, tDog, wind(tDog - tKar2 + 1, 0.9, 500), -23, out=0.5)
    for k in range(4): put(one('cowbell') * 0.4, tKar2 + 1 + k * 2.5 + rnd(0, 1), -34, rnd(-.8, .8), out=0.6)
    bed(tDog, tVo, crowd_walla(tVo - tDog + 1, 10, 0.4) + drips(tVo - tDog + 1, 0.3), -29, hall=0.8)
    bed(tVo, tAm, wind(tAm - tVo + 1, 0.45, 420), -28, out=0.6)
    bed(tAm, tIst3, wind(tIst3 - tAm + 1, 0.5, 300) * 0.8, -27, hall=0.9)
    bed(tIst3, tSim3, wind(tSim3 - tIst3 + 1, 0.6, 350) + city_far(tSim3 - tIst3 + 1, 0.4), -26, out=0.5)
    bed(tSim3, tCeb, birds(tCeb - tSim3 + 1, 1.0) + wind(tCeb - tSim3 + 1, 0.2, 700), -27, out=0.5)
    bed(tCeb, tOsk, lp(noise(tOsk - tCeb + 1, 'brown'), 150) + drips(tOsk - tCeb + 1, 0.3), -28, hall=0.8)
    bed(tOsk, tPek, room_tone(tPek - tOsk + 1, 50), -27, room=0.3)

# ------------------------------------------------------------------ events per shot
def door_pound_hits(d, p):
    a = []
    for i in range(int(math.ceil(d / .4))):
        h = .12 + i * .4 + jhash(i) * .06
        if (not p.get('stopAt')) or h < d * p['stopAt']: a.append(h)
    return a

def side_pound_hits(d):
    out = []; n = 0
    while (n + .5) / 2.4 < d: out.append((n + .5) / 2.4); n += 1
    return out

def ev(i, s):
    t0, t1, tpl, p = s['t0'], s['t1'], s['tpl'], s['p']; d = t1 - t0; m = p.get('m'); K = p.get('_k') or []
    if tpl == 'hookMap':
        put(one('siren', 2), t0 + 0.05, -16, -0.3, out=0.8)
        put(lp(noise(2.0, 'brown'), 200) * env_ar(int(2 * SR), 1.5, 0.3), t0, -18)
    elif tpl == 'horde':
        put(horde(d + 1.5, 14), t0, -14, 0, out=0.5)
    elif tpl == 'eyeOpen':
        put(breath(1.2, 1.0, 0.6), t0 + 0.2, -22); put(lp(noise(0.5, 'pink'), 600) * env_ar(int(.5 * SR), .2, .3), t0 + 1.25, -34)
    elif tpl == 'clock':
        f = p.get('flipAt', .8)
        for k in range(int(d)): put(shift(one('woodblock'), 14)[:, :int(.1 * SR)], t0 + k + 0.1, -32, -0.4, room=0.3)
        put(click(1.0, 2200), t0 + f, -24)
        bz = 0.0
        while bz < d:
            if math.sin(bz * 5) > 0: put(vibrate(0.3), t0 + bz, -24, 0.4, room=0.3)
            bz += 0.32
    elif tpl == 'notifs':
        notes = [79, 84, 76, 81, 88, 79, 83, 86]
        for k in range(8):
            if k * .22 < d: put(chimes([notes[k], notes[k] + 5], 1.0, 'marimba', .07), t0 + k * .22, -24, rnd(-.3, .3), room=0.2)
        for k in range(int(d / .9)): put(vibrate(0.4), t0 + k * .9, -26, 0, room=0.2)
    elif tpl == 'badge':
        tt = 0.0; k = 0
        while tt < d * 0.75:
            put(chimes([84 + (k % 5) * 2], 0.6, 'glock'), t0 + tt, -32, rnd(-.6, .6), room=0.2)
            tt += max(0.05, 0.3 * (1 - tt / (d * .75))); k += 1
        put(chimes([84, 91], 1.0, 'glock', .08), t0 + d * .75, -24)
    elif tpl == 'notThese':
        put(lp(vibrate(0.4), 600), t0 + 0.2, -30)
        put(lp(door_slam(0.6), 500), t0 + d * 0.6, -26, 0.3, room=0.6)
    elif tpl == 'door':
        if m == 'hits':
            for h in K: put(door_slam(1.0), t0 + h, -6, 0.05, room=0.6)
        elif m == 'pound':
            for h in door_pound_hits(d, p): put(door_slam(0.8), t0 + h, -11, 0.05, room=0.6)
        elif m == 'calm':
            for h in K: put(knock_wood(0.6), t0 + h, -14, 0, room=0.5)
        elif m == 'dark':
            put(creak(0.9, 12, 30, (250, 480)), t0 + 0.4, -30, -0.4, room=0.5)
    elif tpl == 'hinge':
        for h in door_pound_hits(d, {}):
            put(door_slam(0.7), t0 + h, -12, 0, room=0.5)
            put(metal_hit(0.25, 8, 0.4), t0 + h + 0.01, -22, 0.2)
            put(grains(0.25, 200, (0.001, 0.003), (2000, 8000)), t0 + h, -26)
    elif tpl == 'hall':
        for h in door_pound_hits(d, {}): put(lp(door_slam(0.7), 900), t0 + h, -18, 0.6, room=0.6)
        st = steps(d, 1.6, 'bare', 1.0); put(st, t0, -26, -0.2, room=0.3)
        put(creak(0.6, 18, 35, (300, 600)), t0 + d * 0.5, -32, 0, room=0.4)
    elif tpl == 'peepEye':
        put(breath(d, 0.8, 0.5), t0, -30)
        put(lp(door_slam(0.6), 700), t0 + 0.4, -22, 0.5, room=0.5)
    elif tpl == 'peep':
        if m == 'pound':
            for h in door_pound_hits(d, {}): put(door_slam(0.9), t0 + h, -10 if not p.get('fb') else -16, 0, room=0.6)
        elif m == 'warm':
            for h in door_pound_hits(d, {})[::2]: put(lp(door_slam(0.7), 1200), t0 + h, -18, 0, room=0.6)
            put(lp(growl(1.5, -10, 1.0), 700), t0 + 0.5, -26, room=0.5)
        else:
            put(lp(breath(d, 1.0, 0.7), 700), t0, -30, room=0.4)
            for k in range(int(d * 4)):
                if jhash(int((t0 + k / 4) * 14)) > .85: put(click(0.4, 6000), t0 + k / 4, -40)
    elif tpl == 'garageFB':
        for k in range(int(d / 1.1)): put(metal_hit(0.3, 14, 0.4), t0 + 0.4 + k * 1.1 + rnd(0, .3), -30, 0.3, room=0.4)
        melody_whistle(t0 + 0.3, d)
    elif tpl == 'baklavaFB':
        put(creak(0.7, 15, 30, (300, 600)), t0, -32, room=0.4)
        put(chimes([88, 93], 0.5, 'glock', .1), t0 + 0.5, -34, 0.3)
    elif tpl == 'face':
        fx = p.get('fx') or []
        if 'growl' in fx: put(growl(d, -12), t0 + 0.1, -12, 0, room=0.4)
        if 'stain' in fx: put(lp(breath(d, 0.8, 0.5), 900), t0, -30)
        if p.get('ch') == 'sen' and p.get('expr') in ('sh', 'sc'): put(breath(0.6, 1.2, 1.5), t0 + 0.05, -24)
    elif tpl == 'doorSides':
        if m in ('pound', 'sitPhone', 'away', 'reveal'):
            for h in side_pound_hits(d):
                put(door_slam(0.85) if m != 'reveal' else lp(door_slam(0.75), 1500), t0 + h, -9 if m == 'pound' else -12, -0.15, room=0.6)
            if m == 'reveal': put(lp(growl(min(d, 2.5), -8, 1.0), 800), t0 + 0.3, -28)
            if m == 'away': put(steps(d, 1.6, 'bare', 0.8), t0 + 0.2, -30, 0.3)
            if m == 'sitPhone': put(click(0.6, 3000), t0 + 0.6, -30, 0.3)
        elif m == 'listen':
            put(breath(d, 0.7, 0.5), t0, -32)
        elif m == 'tik':
            for h in K: put(knock_wood(0.55), t0 + h, -13, -0.2, room=0.5)
    elif tpl == 'decision':
        put(chimes([74, 81], 0.6, 'marimba', .12), t0 + .35, -28, 0.3)
    elif tpl == 'crack':
        g = one('glass')[:, :int(0.4 * SR)]; put(lp(g, 5000), t0 + 0.15, -16, 0, room=0.3)
    elif tpl == 'mystery':
        tt = 0.25
        while tt < d: put(lp(door_slam(0.6), 700), t0 + tt, -20, 0, room=0.5); tt += 0.5
    elif tpl == 'shatter':
        b = p.get('brk', .9); put(glass(1.0, 3), t0 + b, -7, 0, room=0.4); put(metal_hit(0.4, -5, 1.0), t0 + b, -20)
    elif tpl == 'tmap':
        tmap_sfx(t0, d, p)
    elif tpl == 'title':
        st = p.get('st') or 'slam'
        if st == 'slam': put(lp(one('bdrum_soft'), 700)[:, :int(.6 * SR)], t0 + 0.05, -20); put(whoosh(0.3, 300, 2500, 1, rev=True), t0 - 0.25, -28)
        if p.get('bgk') == 'static': put(static(d, 1.0, 1.5), t0, -16)
        if p.get('icon') == 'dog': put(lp(growl(min(d, 2), -16, 1, dog=True), 700), t0 + 0.4, -24)
        if p.get('icon') == 'key': put(keys_jingle(), t0 + 0.2, -22)
        if p.get('icon') == 'money': put(paper(0.5), t0 + 0.2, -26)
        if p.get('icon') == 'thermo': put(breath(d, 1.0, 1.4), t0, -28)
        if p.get('icon') == 'clock':
            for k in range(int(d / .25)): put(shift(one('woodblock'), 14)[:, :int(.08 * SR)], t0 + k * .25, -30)
        if p.get('icon') == 'stove': put(fire(d, 0.3), t0, -30)
        if p.get('icon') == 'gauge': put(tone([880, 1320], 0.35, 0.3, None, False), t0 + 0.4, -26)
        if 'KIYAMET' in (p.get('txt') or ''): put(explosion_far(1.0), t0 + 0.2, -16, out=0.5)
    elif tpl == 'tiles':
        for k, it in enumerate(p.get('items') or [{}] * len(K)):
            at = K[k] if k < len(K) else k * .4
            put(lp(one('bdrum_soft'), 900)[:, :int(.35 * SR)], t0 + at, -24); put(whoosh(0.25, 600, 4000, 1, rev=True), t0 + at - 0.22, -30)
        # crossed-out items get a marker slash
        if p.get('k') and any(w in ('ordu', 'trafik', 'hoparlör') for w in p['k']):
            for at in K: put(whoosh(0.18, 3000, 9000), t0 + at + 0.35, -26)
    elif tpl == 'chapter':
        put(whoosh(0.6, 200, 2500, 1, rev=True), t0 - 0.55, -22)
        put(static(0.25, 1.0, 2.0), t0, -30)
    elif tpl == 'chat':
        ty = p.get('typing') or ''
        nmsg = 7 if p.get('fire') else 3 if 'Hatice' in ty else 2
        iv = p.get('iv', .35)
        for k in range(nmsg):
            if k * iv < d: put(chimes([86 if k % 2 else 81], 0.5, 'marimba'), t0 + k * iv, -30, 0.2, room=0.2)
        if p.get('fire'): put(fire(d, 0.5), t0, -32)
    elif tpl == 'talker':
        put(lp(one('bdrum_soft'), 700)[:, :int(.3 * SR)], t0 + 0.15, -32)
    elif tpl == 'limon':
        put(bubbles_boil(d + 0.5), t0, -22, room=0.3); put(bp(noise(d, 'pink'), 1500, 6000) * 0.2, t0, -30)
    elif tpl == 'hospital':
        sp = np.zeros((2, int(d * SR)), np.float32)
        al = tone([1050], d, 0.25, (0.25, 0.25), False); sp += al[:, :sp.shape[1]] * 0.3
        sp += crowd_walla(d, 10, 1.2)[:, :sp.shape[1]]
        sp += steps(d, 5.0, 'stone', 0.7, run=True)[:, :sp.shape[1]]
        si = distant(one('siren', 1), 0.3)[:, :sp.shape[1]]; sp[:, :si.shape[1]] += si * 0.4
        put(phone_speaker(sp), t0, -16)
    elif tpl == 'tv':
        if m == 'on': put(thunk(1.0, 70), t0, -22); put(bp(noise(0.3, 'white'), 4000, 9000) * env_ar(int(.3 * SR), 0, .25), t0, -34)
        if m == 'flip':
            for k in range(int(d / .42) + 1): put(static(0.08, 1.0), t0 + k * .42, -26); put(click(0.6, 2000), t0 + k * .42, -30)
        if m == 'crack': put(static(0.8, 1.0, 2.0), t0, -22); put(lp(one('glass')[:, :int(.4 * SR)], 4000), t0 + 0.25, -20)
        if m in ('news', 'ticker', 'on'): put(phone_speaker(chimes([62, 69, 74], 1.0, 'tbells', .12)), t0 + 0.1, -26)
    elif tpl == 'airport':
        if m == 'land': put(jet(d + 1.0, 1.0), t0 - 0.3, -16, 0, out=0.4); put(lp(one('bdrum'), 300), t0 + d * 0.8, -20)
        else:
            put(crowd_walla(d, 12, 1.0), t0, -28, hall=0.6); put(chimes([76, 72], 0.7, 'marimba', .45), t0 + 0.3, -26, hall=0.6)
            for k in range(int(d / 1.2) + 1): put(lp(breath(0.25, 1.5, 3), 1500), t0 + k * 1.2 + 0.05, -28)
    elif tpl == 'istanbul':
        if m == 'spread':
            for k in range(8):
                at = .2 + k * (d - .6) / 8 + .3; put(chimes([74 + k], 0.5, 'glock'), t0 + at, -30, rnd(-.6, .6), out=0.4)
        if m == 'split':
            put(lp(noise(2.5, 'brown'), 160) * env_ar(int(2.5 * SR), .5, 1.5), t0 + .2, -12); put(one('brick') * 0.6, t0 + .4, -22, out=0.3)
        if m == 'ring': put(whoosh(d * .6, 200, 900, 1), t0, -28)
        if m == 'lost':
            for k in range(3): put(explosion_far(0.8), t0 + rnd(0, d), -24, rnd(-.8, .8), out=0.6)
            put(distant(shift(inst('tuba').render(36, 2.5, 0.6), 0), 0.6), t0 + 0.5, -24, 0.5, out=0.8)   # ship horn
    elif tpl == 'rules':
        put(whoosh(0.3, 500, 4000, 1, rev=True), t0 - 0.25, -28)
    elif tpl == 'morph':
        for k in range(6): put(shift(one('claves'), rnd(-6, 4))[:, :int(.2 * SR)], t0 + 0.3 + k * rnd(0.15, 0.5), -24, rnd(-.4, .4))
        put(growl(d, -10), t0 + d * 0.5, -18)
    elif tpl == 'soundPull':
        ring = chimes([76, 72, 79, 76, 72, 79], 1.0, 'marimba', .14)
        for k in range(int(d / 1.4) + 1): put(phone_speaker(ring), t0 + k * 1.4, -24, out=0.4)
        put(horde(d, 8), t0, -22, out=0.5)
    elif tpl == 'phoneUI':
        if m == 'call':
            for k, f in enumerate([(697, 1209), (697, 1209), (941, 1336)]): put(tone(f, 0.12, 0.3, None, True), t0 + 0.05 + k * 0.18, -26)
            put(tone([425], d - .7, 0.3, (0.25, 0.25), True), t0 + 0.7, -24)
        if m == 'alert':
            put(tone([853, 960], min(d, 2.4), 0.45, None, True), t0, -14)
            for k in range(int(d / .6)): put(vibrate(0.45), t0 + k * .6, -24)
        if m == 'nosig':
            for k, f in enumerate([950, 1400, 1800]): put(tone([f], 0.33, 0.3, None, True), t0 + 0.8 + k * 0.34, -24)
        if m == 'vib':
            for k in range(int(d / .9) + 1): put(vibrate(0.45), t0 + k * .9, -18, room=0.3)
        if m in ('apps', 'vpn'):
            for k in range(int(d / 0.6)): put(click(0.5, 3000), t0 + 0.3 + k * 0.6 + rnd(0, .2), -32)
    elif tpl == 'window':
        put(city_far(d, 1.5), t0, -28); put(crowd_walla(d, 10, 0.5), t0, -34)
    elif tpl == 'market':
        if m == 'queue': put(crowd_walla(d, 18, 1.0), t0, -24, out=0.3)
        if m == 'grab':
            for k, at in enumerate(K): put(lp(one('framedrum'), 2500)[:, :int(.3 * SR)], t0 + at, -20, rnd(-.4, .4)); put(paper(0.25), t0 + at, -28)
            put(steps(d, 4, 'stone', 0.8, run=True), t0, -30)
        if m == 'bulgur':
            put(steps(d, 1.7, 'stone', 1.0), t0, -26)
            for k in range(int(d / .7)): put(paper(0.3, 0.8), t0 + k * .7, -30)
    elif tpl == 'announce':
        put(paper(0.5), t0, -24); put(lp(one('bdrum_soft'), 1200)[:, :int(.4 * SR)], t0 + .7, -14); put(shift(one('woodblock'), -10)[:, :int(.3 * SR)], t0 + .7, -20)
    elif tpl == 'kids':
        if m == 'freeze': put(whoosh(0.6, 3000, 200, 1), t0, -22)
        else:
            for k in range(14): put(one('claves')[:, :int(.15 * SR)] * 0.5, t0 + rnd(0, .4), -30, rnd(-.8, .8))
            bell = np.zeros((2, int(1.4 * SR)), np.float32)
            for k in range(28): h = one('triangle')[:, :int(.2 * SR)]; i2 = int(k * 0.05 * SR); bell[:, i2:i2 + h.shape[1]] += h[:, :bell.shape[1] - i2]
            put(bell, t0, -28, 0.5, out=0.5)
    elif tpl == 'traffic':
        put(lp(noise(d, 'brown'), 500) * 1.5, t0, -30)
        for k in range(int(d * 1.5)): put(horn(rnd(.2, .6), (rnd(380, 460), rnd(470, 560))), t0 + rnd(0, d), -30, rnd(-.9, .9), out=0.5)
    elif tpl == 'exodus':
        put(engine(d, 900, 950, 4, 1.0), t0, -26)
        for w in range(int(d * 3)):
            for k in range(8):
                if jhash(k + w) > .6 and not jhash(k + w - 1) > .6 and rnd() < 0.5:
                    put(horn(rnd(.25, .6), (rnd(380, 460), rnd(470, 560))), t0 + w / 3, -24, ((k * 330 % 2200) / 1100) - 1, out=0.4)
    elif tpl == 'quiz':
        put(chimes([72, 76], 0.5, 'marimba', .15), t0 + .2, -30)
    elif tpl == 'stats':
        st = p.get('st', 0)
        if st == 0:
            for k in range(14): put(click(0.4, 2500), t0 + k * 0.085, -32)
        if st == 2:
            for k in range(6): put(whoosh(0.4, 800, 4000, 0.7), t0 + .6 + k * .2, -32, 0.6)
    elif tpl == 'xray':
        t_ = T_(d); hum = mono2(np.sin(2 * np.pi * 120 * t_) * 0.1 + np.sin(2 * np.pi * 240 * t_) * 0.05)
        put(hum, t0, -30); put(tone([1400], 0.15, 0.3, None, False), t0 + d * 0.55, -28)
    elif tpl == 'pandemic':
        put(birds(d, 0.6), t0, -30, out=0.4)
    elif tpl == 'police':
        put(one('whistle', 0), t0 + 0.3, -22, -0.4, out=0.4)
        put(horde(d * 0.6, 4), t0 + d * 0.4, -26, 0.5)
    elif tpl == 'pipe':
        t_ = T_(d); put(mono2(np.sin(2 * np.pi * 90 * t_) * 0.08) + bp(noise(d), 800, 2400) * 0.1, t0, -28)
        put(whoosh(1.0, 2000, 300, 1), t0 + 0.3, -26)
    elif tpl == 'crash':
        t_ = T_(d); f = 300 * np.exp(-t_ * 2); put(mono2(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t_ * 1.5) * 0.3), t0, -22)
        put(static(d, 0.8, 3), t0, -26)
    elif tpl == 'cappaFam':
        put(birds(d, 1.2), t0, -26, out=0.4)
        for k in range(2): put(one('nepal') * 0.4, t0 + .5 + k * 1.3, -32, rnd(-.8, .8), out=0.5)
    elif tpl == 'voice':
        cut = p.get('cut', .3)
        put(static(d, 0.5, 1.0) * 0.6, t0, -30)
        for w in range(int(d * 6)):
            if jhash(w * 3 + int(t0 * 6)) < cut: put(static(0.12, 1.0, 3.0), t0 + w / 6, -28)
    elif tpl == 'flatline':
        put(static(0.4, 1.0, 2.0), t0, -24)
    elif tpl == 'kiler':
        if m == 'mystery': put(wind(d, 0.4, 1200), t0, -30, room=0.4); put(creak(1.0, 10, 25, (250, 500)), t0 + 0.6, -30, room=0.5)
        if m == 'memory': put(steps(d * 0.6, 3, 'bare', 0.5), t0, -32)
        if m == 'locked': put(one('chain', 1)[:, :int(d * SR)], t0 + 0.1, -24, room=0.4); put(metal_hit(0.4, 10, 0.6), t0 + 0.4, -26)
        if m == 'reveal': put(creak(1.8, 12, 45, (280, 520, 950)), t0, -18, room=0.5); put(whoosh(1.5, 150, 700, 1, rev=True), t0 + 0.5, -26, hall=0.6)
    elif tpl == 'balcony':
        if m == 'step': put(creak(0.5, 30, 60, (500, 900)), t0 + .1, -30); put(steps(1.2, 1.6, 'stone', 0.7), t0 + 0.3, -30)
        if m == 'out':
            for k in range(7): put(lp(one('bdrum_soft'), 200)[:, :int(.6 * SR)] * 0.6, t0 + 0.4 + k * (d / 7.5), -26, rnd(-.8, .8), out=0.7)
    elif tpl == 'bridgeMil':
        put(engine(d, 650, 650, 6, 1.0, 0.1, diesel=True), t0, -24, out=0.3)
        put(steps(d, 1.2, 'stone', 0.6), t0 + 0.3, -32, out=0.3)
    elif tpl == 'barricade':
        put(one('chain', 0)[:, :int(d * SR)] * 0.5, t0 + 0.2, -30, out=0.4)
        put(engine(d, 640, 640, 6, 0.8, 0.1, diesel=True), t0, -30, out=0.3)
        if p.get('you'): put(metal_hit(0.5, 6, 0.8), t0 + 0.2, -24, out=0.3)
    elif tpl == 'transit':
        if m == 'train':
            w = shift(one('wineglass'), 7)[:, :int(2 * SR)]; put(w * env_ar(w.shape[1], .2, 1.0)[None, :], t0, -24, hall=0.6)
            put(hp(noise(1.5, 'white'), 2000) * env_ar(int(1.5 * SR), .05, 1.2) * 0.3, t0 + 1.2, -26, hall=0.4)
        else:
            put(starter(1.0, False, 0.6), t0, -30); put(metal_hit(0.8, -4, 1.0), t0 + 0.8, -20, out=0.4)
    elif tpl == 'family': put(waves(d, 1.0), t0, -24, out=0.3)
    elif tpl == 'boats':
        put(engine(d, 300, 310, 1, 1.0, 0.04, diesel=True), t0, -26, out=0.4)
        put(creak(1.2, 10, 20, (200, 380)), t0 + 0.5, -30, out=0.4); put(paper(0.4), t0 + 1.0, -30)
    elif tpl == 'cage':
        put(metal_hit(1.0, -12, 2.0), t0 + .65, -10, out=0.6); put(one('anvil') * 0.5, t0 + .65, -18, out=0.5); put(one('brake') * 0.6, t0 + .7, -20, out=0.5)
        put(whoosh(0.7, 200, 1200, 1, rev=True), t0, -24)
    elif tpl == 'gunfire':
        per = .9 if p.get('many') else 1.6; n = 24 if p.get('many') else 3
        for k in range(n):
            ph0 = jhash(k * 3) * per; x = (0.05 + jhash(k * 7.1) * 0.9) * 2 - 1
            q = 0
            while True:
                tt = q * per - ph0
                if tt >= d: break
                if tt >= 0: put(gunshot(0.35 + 0.5 * abs(x) if not p.get('many') else rnd(.4, .9)), t0 + tt, -12 if not p.get('many') else -18, x * 0.8, out=0.5)
                q += 1
    elif tpl == 'wedding':
        wedding(t0, d)
    elif tpl == 'gunDown':
        tt = (math.asin(.85) % (2 * math.pi)) / 9
        k = 0
        while tt + k * (2 * math.pi / 9) < d: put(gunshot(0.2), t0 + tt + k * (2 * math.pi / 9), -12, -0.6, out=0.4); k += 1
        put(horde(d, 6), t0, -22, 0.3, out=0.4)
    elif tpl == 'soundMap':
        for k in range(5): put(gunshot(0.6), t0 + rnd(0, d), -20, rnd(-.3, .3), out=0.4)
        put(horde(d, 10) * np.linspace(0.3, 1, int(d * SR))[None, :], t0, -20, out=0.5)
    elif tpl == 'nightFF':
        for k in range(int(d * 2)):
            tt = rnd(0, d)
            if rnd() > tt / d: put(gunshot(rnd(.6, 1.0)), t0 + tt, -24, rnd(-.9, .9), out=0.5)
        for k in range(int(d / .12)): put(shift(one('woodblock'), 16)[:, :int(.05 * SR)], t0 + k * .12, -38)
    elif tpl == 'bulb':
        pop = p.get('pop', 1.0); t_ = T_(pop)
        hum = mono2((np.sin(2 * np.pi * 100 * t_) * 0.1 + np.sin(2 * np.pi * 200 * t_) * 0.06) * (0.5 + 0.5 * (np.random.rand(len(t_)) > 0.02)))
        put(hum + grains(pop, 40, (0.001, 0.004), (2000, 9000)), t0, -26)
        put(lp(one('glass')[:, :int(.3 * SR)], 6000), t0 + pop, -16); put(spark(0.4), t0 + pop, -20)
    elif tpl == 'darkCity': put(wind(d, 0.3, 380), t0, -30, out=0.5)
    elif tpl == 'streetDark':
        put(wind(d, 0.5, 600), t0, -28, out=0.4); put(paper(0.6, 0.6), t0 + d * .5, -32, out=0.4)
    elif tpl == 'sirenDogs':
        put(distant(one('siren', 3), 0.8), t0 + 0.2, -24, 0.6, out=0.8); put(distant(one('siren_v', 1), 0.9), t0 + d * .5, -28, -0.6, out=0.8)
    elif tpl == 'minaret':
        put(wind(d, 0.35 if m == 'silent' else 0.25, 450), t0, -30, out=0.5)
    elif tpl == 'calendar':
        r = p.get('rate', 9)
        for k in range(int(d * r)): put(paper(0.12, 0.9), t0 + k / r, -26, rnd(-.3, .3))
    elif tpl == 'meeting':
        if m == 'gather':
            for k in range(6): put(creak(0.4, 30, 55, (400, 800)), t0 + rnd(0, d), -34, rnd(-.8, .8), room=0.5)
        if m == 'throat': pass
        if m == 'stare': put(crickets(d, 3, 1.4), t0, -24, room=0.3)
    elif tpl == 'agendaViz':
        if m == 'cars':
            for k in range(3): put(engine(0.9, 900, 600, 4, 0.8), t0 + .2 + k * .3, -30, (k - 1) * .6, out=0.3); put(thunk(0.8, 80), t0 + .9 + k * .3, -26, (k - 1) * .6)
        else: put(bp(noise(d, 'pink'), 300, 1500) * 0.4 + grains(d, 30, (.005, .02), (300, 1500)), t0, -30, room=0.4)
    elif tpl == 'ledger': put(paper(0.5), t0, -26); put(pencil(1.2), t0 + 0.6, -24)
    elif tpl == 'recep': put(metal_hit(0.35, 9, 0.6), t0 + 0.5, -24, room=0.4)
    elif tpl == 'stairs':
        if m == 'up': put(steps(d, 1.5, 'stone', 0.8), t0, -26, hall=0.5)
        if m == 'door': put(creak(1.4, 10, 22, (260, 480, 820)), t0 + 0.3, -26, room=0.5)
    elif tpl == 'emptyHome':
        if m == 'empty': put(creak(1.1, 12, 30, (260, 480)), t0, -28, room=0.6)
        if m == 'cold': put(wind(d, 0.6, 1300), t0, -26, room=0.5)
        if m == 'note': put(paper(0.5), t0 + 0.6, -26)
        if m == 'items' and len(K) >= 2: put(keys_jingle(), t0 + K[0], -20); put(thunk(0.8, 110), t0 + K[1], -22)
    elif tpl == 'water':
        if m == 'tap':
            put(creak(0.5, 40, 70, (900, 1600)), t0 + 0.1, -26)
            put(bp(noise(0.6, 'pink'), 400, 3000) * env_ar(int(.6 * SR), .05, .4) * 0.4, t0 + 0.5, -28)
            put(drips_one(), t0 + d * 0.6, -20, room=0.4)
        if m == 'hydro':
            t_ = T_(d); f = 50 * np.exp(-t_ * 0.8); put(mono2(np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.2 * np.exp(-t_ * 0.7)), t0, -22)
        if m == 'tank': put(metal_hit(0.8, -10, 1.6), t0 + 0.3, -18, hall=0.5)
        if m == 'bottle': put(grains(0.6, 200, (.003, .015), (1000, 7000)), t0 + 0.3, -26)
    elif tpl == 'note':
        r = p.get('rate', 22); lines = p.get('lines') or ['']; cur = p.get('cur', len(lines) - 1)
        L = len(lines[cur]) if cur < len(lines) else 10
        if r < 50: put(pencil(min(d, L / r)), t0, -18)
        else: put(paper(0.6), t0 + 0.1, -26)
        if p.get('sign'): put(pencil(0.7, 1.2), t0 + 0.3, -18)
    elif tpl == 'fbMarket':
        put(growl(1.2, -9), t0 + 0.2, -14); put(crowd_walla(d, 8, 1.0), t0, -28)
        put(shift(one('claves'), -8)[:, :int(.2 * SR)], t0 + d * .4, -18)
    elif tpl == 'garage':
        garage_sfx(t0, d, m, p)
    elif tpl == 'howto':
        for h in K: put(metal_hit(0.9, 4, 0.8), t0 + h, -14, room=0.3); put(spark(0.5), t0 + h, -18)
    elif tpl == 'street':
        if m == 'pharmacy':
            t_ = T_(d); put(mono2(np.sin(2 * np.pi * 120 * t_) * 0.06) + grains(d, 15, (0.001, 0.003), (3000, 9000)), t0, -28, out=0.3)
        if p.get('you'): put(steps(d, 1.6, 'stone', 0.8), t0, -28, out=0.3)
        if m in ('wide', 'cars'): put(paper(0.8, 0.5), t0 + rnd(0, d), -34, out=0.4)
        if m == 'corner': put(breath(d, 0.4, 2.2, panting=True), t0, -34)
    elif tpl == 'karabas':
        if m == 'growl': put(growl(d, -16, 1.0, dog=True), t0, -12, -0.2, room=0.3)
        else: put(breath(d, 0.8, 2.5, panting=True), t0, -26)
    elif tpl == 'alley':
        for k, at in enumerate(K):
            if at < d: put(zgroan(), t0 + at + 0.1, -14, (k - 1) * 0.5, out=0.5)
    elif tpl == 'chase':
        put(steps(d, 5.2, 'stone', 1.2, run=True), t0, -18, -0.1, out=0.3)
        put(steps(d, 7.5, 'bare', 0.5, run=True), t0, -26, 0.3)
        put(breath(d, 1.4, 2.6, panting=True), t0, -24)
        put(horde(d, 5), t0, -18, -0.6, out=0.4)
    elif tpl == 'shutter':
        sl = K[0] if K else 1.2
        ch = one('chain', 2); put(shift(ch, 3)[:, :int(min(sl, 3) * SR)], t0 + max(0, sl - 0.5), -16, out=0.4)
        put(metal_hit(1.0, -6, 1.6), t0 + sl, -6, out=0.5); put(one('brake') * 0.7, t0 + sl, -12, out=0.4); put(lp(one('bdrum'), 800), t0 + sl, -12)
        put(steps(min(sl, 2), 5, 'stone', 1.0, run=True), t0, -24)
    elif tpl == 'darkShop':
        put(breath(d, 1.0, 1.6 if m == 'wait' else 1.0, panting=(m == 'wait')), t0, -24)
        put(lp(horde(d, 3), 600), t0, -30, room=0.3)
        if m == 'radar': put(lp(growl(d * .7, -18, 1, dog=True), 600), t0 + .2, -24)
    elif tpl == 'ignore':
        put(steps(d, 1.2, 'stone', 0.6), t0, -30, out=0.3); put(zgroan(), t0 + .3, -18, -0.3, out=0.4)
    elif tpl == 'walkTogether':
        put(steps(d, 1.7, 'stone', 0.6), t0, -30, out=0.3)
    elif tpl == 'radio':
        radio_sfx(t0, d, m, p)
    elif tpl == 'ugx':
        if m == 'reveal' or m == 'danger': put(lp(noise(d, 'brown'), 120) * 1.2, t0, -22, hall=0.6)
        if m == 'danger':
            for k in range(6): put(horse_step(), t0 + k * 0.25, -28, -0.5 + k * 0.2, out=0.4)
        if m == 'air': put(wind(d, 0.6, 900), t0, -24, hall=0.7)
        if m == 'labels':
            for at in K: put(whoosh(0.25, 800, 4000, 0.8, rev=True), t0 + at - .22, -32)
        if m == 'animals':
            for k in range(3): put(one('nepal') * 0.5, t0 + 0.4 + k * 1.1, -32, rnd(-.6, .6), hall=0.5)
    elif tpl == 'dive':
        put(lp(noise(d, 'brown'), 150) * 1.4 * np.linspace(0.3, 1, int(d * SR))[None, :], t0, -18, hall=0.6)
        put(one('brick', 0) * 0.5, t0 + d * 0.3, -26, hall=0.6)
    elif tpl == 'wall1963':
        if m == 'break':
            for h in K:
                put(lp(one('bdrum'), 1500), t0 + h, -12, room=0.4); put(shift(one('snare'), -6)[:, :int(.4 * SR)], t0 + h, -16)
                put(grains(1.0, 120, (0.002, 0.02), (300, 6000), (0.2, 1.0)) * np.exp(-np.linspace(0, 3, SR)), t0 + h + 0.03, -18, room=0.4)
        else: put(wind(d, 0.5, 300), t0, -24, hall=0.8)
    elif tpl == 'stone':
        k = min(d * .75, 4.0)
        put(lp(speed(one('brick', 1), 0.6), 1500)[:, :int(k * SR)], t0, -14, hall=0.5)
        put(lp(noise(k, 'brown'), 120) * 1.5, t0, -18, hall=0.4); put(lp(one('bdrum'), 400), t0 + k, -12, hall=0.6)
    elif tpl == 'counter':
        for k in range(int(d * 0.7 / 0.06)): put(click(0.3, 2800), t0 + k * 0.06, -36)
    elif tpl == 'ancient':
        for k in range(int(d / .25)): put(horse_step(), t0 + k * .25, -26, np.sin(k) * .6, out=0.4)
        put(fire(d, 0.4), t0, -30)
    elif tpl == 'glovebox':
        if m == 'open': put(click(1.0, 1500), t0, -24); put(paper(0.7), t0 + 0.3, -22)
        else: put(tone([660, 440], 0.4, 0.3, (0.2, 0.0), False), t0 + 0.3, -28)
    elif tpl == 'drive':
        if p.get('bumpy'):
            for k in range(int(d * 2)): put(lp(one('bdrum_soft'), 200)[:, :int(.3 * SR)], t0 + rnd(0, d), -28)
    elif tpl == 'mechanic':
        for k in range(4): put(metal_hit(0.35, 12, 0.4), t0 + 0.3 + k * 0.5, -28, 0.4)
    elif tpl == 'mountain': put(wind(d, 1.0, 330), t0, -22, out=0.4)
    elif tpl == 'tunnelEntry':
        if not p.get('km'): put(lp(one('bdrum_soft'), 300), t0 + 0.2, -30)
    elif tpl == 'tunnel':
        tunnel_sfx(t0, d, m, p)
    elif tpl == 'tunnelSide':
        if m == 'flash': put(click(1.0, 1800), t0 + 0.1, -22, hall=0.5); put(thunk(0.8, 140), t0 + 0.8, -24, hall=0.6); put(creak(0.8, 25, 50, (400, 900)), t0 + 1.5, -26, hall=0.6)
        if m == 'door':
            put(creak(d * 0.7, 8, 30, (220, 420, 780)), t0, -14, hall=0.8); put(hp(noise(0.8, 'white'), 2500) * env_ar(int(.8 * SR), .05, .6) * 0.3, t0 + 0.2, -26, hall=0.6)
            put(horde(d, 4), t0 + d * 0.3, -20, -0.5, hall=0.8)
        if m == 'hit':
            for h in K: put(metal_hit(1.0, 3, 0.9), t0 + h, -12, hall=0.6); put(spark(0.5), t0 + h, -16, hall=0.4)
    elif tpl == 'squeeze':
        put(one('chain', 0)[:, :int(d * SR)] * 0.7, t0, -16, 0.4, hall=0.5)
        put(lp(speed(one('brick', 0), 1.6), 6000)[:, :int(d * SR)], t0, -18, 0.5, hall=0.4)
        put(spark(d, 2.0), t0, -20, 0.5); put(horde(d, 6), t0, -22, -0.5, hall=0.6)
    elif tpl == 'sunrise':
        put(birds(d, 1.5), t0, -26, out=0.4)
    elif tpl == 'dogShoulder':
        put(breath(1.6, 0.8, 0.5), t0 + 1.0, -28)
    elif tpl == 'density':
        for at in K: put(chimes([76], 0.5, 'marimba'), t0 + at, -30)
    elif tpl == 'sim':
        if m == 'dense':
            for k in range(int(d * 8)): put(click(0.4, rnd(1500, 4000)), t0 + d * (k / (d * 8)) ** 0.7, -34, rnd(-.8, .8))
        else: put(steps(d, 0.9, 'stone', 0.5), t0, -32, out=0.4); put(wind(d, 0.4, 500), t0, -30)
    elif tpl == 'station':
        if m == 'dark': put(creak(1.4, 6, 14, (200, 360)), t0 + 0.5, -26, out=0.5)
        if m == 'siphon':
            gl = grains(d, 25, (0.01, 0.04), (200, 1200), (0.3, 1.0), decay=False); put(gl, t0, -22)
            put(bp(noise(d, 'pink'), 400, 2500) * 0.15, t0, -30)
    elif tpl == 'blockade':
        if m == 'wide': put(engine(d, 520, 520, 2, 1.0, 0.06, diesel=True), t0, -22, out=0.4)
        if m == 'dur':
            put(engine(d, 520, 520, 2, 0.8, 0.06, diesel=True), t0, -26, out=0.4)
            for k in range(2): put(metal_hit(0.6, 8, 0.25), t0 + 0.6 + k * 0.18, -18, out=0.4); put(metal_hit(0.5, 4, 0.25), t0 + 0.75 + k * 0.18, -20, out=0.4)
    elif tpl == 'inspect':
        put(steps(d * 0.5, 1.4, 'stone', 0.6), t0, -30, out=0.3)
        if len(K) > 3: put(lp(one('bdrum_soft'), 1500)[:, :int(.4 * SR)], t0 + K[3], -18)
    elif tpl == 'kahve':
        if m == 'room':
            for k in range(int(d * 1.5)): put(tea_clink(), t0 + rnd(0, d), -28, rnd(-.8, .8), room=0.4)
            dice(t0 + d * 0.4)
        else:
            put(thunk(0.8, 160), t0 + 0.1, -22, room=0.3)
            put(tea_clink(), t0 + 0.6, -24, room=0.3)
    elif tpl == 'kilerFull':
        for at in K: put(jar_clink(), t0 + at, -22, rnd(-.5, .5), room=0.4)
    elif tpl == 'compare':
        put(wind(d * 0.5, 0.6, 700), t0 + d * 0.5, -30, 0.6)
    elif tpl == 'scale':
        put(creak(1.2, 20, 45, (500, 1100)), t0 + .3, -26); put(metal_hit(0.4, 14, 0.4), t0 + 1.3, -26)
    elif tpl == 'gift':
        put(breath(1.5, 0.7, 3, panting=True), t0 + 0.6, -28); put(paper(0.3), t0 + 0.3, -32)
    elif tpl == 'cappaT':
        if m == 'reveal': put(lp(engine(d, 1600, 1400, 4, 0.8), 800), t0, -30, out=0.3)
        if m == 'walk': put(steps(d, 1.6, 'stone', 0.7), t0, -28, out=0.4)
    elif tpl == 'nineHouse':
        for k in range(int(d)): put(shift(one('woodblock'), 7 if k % 2 else 5)[:, :int(.12 * SR)], t0 + 0.2 + k, -30, 0.5, room=0.5)
    elif tpl == 'hatch':
        if m == 'shelves': put(steps(min(d, 2), 1.5, 'stone', 0.6), t0, -30, room=0.4); put(jar_clink(), t0 + d * 0.6, -28, room=0.4)
        if m == 'open': put(creak(1.5, 10, 40, (240, 460, 880)), t0, -18, room=0.5); put(lp(one('bdrum'), 500), t0 + 1.5, -18, hall=0.5); put(wind(d, 0.6, 300), t0 + 0.8, -26, hall=0.8)
    elif tpl == 'descend':
        put(steps(d, 1.6, 'stone', 0.9), t0, -22, hall=0.9)
    elif tpl == 'ughall':
        if m in ('crowd', 'reveal'):
            put(steps(d, 6, 'bare', 0.4, run=True), t0, -34, 0.4, hall=0.6)
        if m == 'reunion':
            k0 = K[0] if K else d * .6
            put(steps(min(k0, d), 4.5, 'stone', 0.9, run=True), t0, -24, hall=0.6)
    elif tpl == 'nineStory': put(fire(d, 0.4), t0, -28, room=0.4)
    elif tpl == 'yayla':
        put(one('brick', 1)[:, :int(min(d, 3) * SR)] * 0.5, t0 + 0.5, -28, out=0.5); put(zgroan(), t0 + d * 0.5, -26, out=0.5)
    elif tpl == 'ezanUG':
        put(wind(d, 0.6, 700), t0, -26, hall=0.9)
    elif tpl == 'alt':
        if m == 'door': put(lp(door_slam(0.6), 900), t0 + 0.3, -24, room=0.6)
        if m == 'zombie': put(zgroan(), t0 + 0.4, -18, 0.3, room=0.5); put(zgroan(), t0 + 1.4, -20, 0.6, room=0.5)
    elif tpl == 'poll':
        for k in range(int(d / 0.8)): put(click(0.6, 2500), t0 + 0.6 + k * 0.83, -28)
    elif tpl == 'comment':
        for k in range(int(d * 12)):
            if 0.3 < k / 12: put(click(0.4, rnd(1800, 3200)), t0 + k / 12 + rnd(0, .03), -32)
    elif tpl == 'subscribe':
        c = (K[0] + .3) if K else 1.4
        put(click(1.0, 2200), t0 + c, -18); put(chimes([88, 93], 1.0, 'glock', .05), t0 + c + 0.15, -20, 0.4)
    elif tpl == 'finale':
        put(horde(d, 8) * 0.8, t0 + 1.0, -22, out=0.6)

# ------------------------------------------------------------------ helpers for the big cases
def keys_jingle():
    out = np.zeros((2, int(0.9 * SR)), np.float32)
    for k in range(9):
        h = inst('glock').render(int(rnd(96, 104)), 0.2, 0.5)[:, :int(0.2 * SR)]
        i = int(rnd(0, 0.6) * SR); out[:, i:i + h.shape[1]] += hp(h, 3000) * rnd(0.2, 0.6)
    return out
def h_len(): return int(0.25 * SR)

def tea_clink():
    h = inst('glock').render(int(rnd(98, 103)), 0.15, 0.3)[:, :int(0.25 * SR)]
    return hp(h, 2500) * env_ar(h.shape[1], 0.0, 0.2) * 0.6

def jar_clink():
    a = inst('glock').render(int(rnd(88, 95)), 0.3, 0.5)[:, :int(.35 * SR)] * 0.5
    b = lp(one('bdrum_soft'), 900)[:, :int(.35 * SR)] * 0.5
    return a + b[:, :a.shape[1]]

def dice(t):
    for k in range(9): put(shift(one('claves'), rnd(4, 9))[:, :int(.08 * SR)], t + k * rnd(0.03, 0.09), -30, 0.4, room=0.4)

def drips_one():
    tn = np.arange(int(0.15 * SR)) / SR; f = 1400 * (1 + 0.8 * tn / 0.15)
    return mono2(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tn * 35) * 0.4)

def horse_step():
    return lp(shift(one('claves'), -12)[:, :int(.15 * SR)], 1800) * 0.8

def melody_whistle(t, d):
    """Kemal whistling while fixing his car (real flute, small room)"""
    tune = [('A5', .4), ('D6', .4), ('C6', .4), ('A5', .8), ('G5', .4), ('A5', .4), ('F5', .8)]
    tt = t
    for n, du in tune:
        if tt - t > d - 0.4: break
        y = inst('flute').render(midi(n), du * 0.95, 0.4, 0.03, 0.15); put(y, tt, -26, 0.3, room=0.5); tt += du

def wedding(t0, d):
    e = 60 / 120
    for k in range(int(d / e)):
        put(lp(one('bdrum'), 2500), t0 + k * e, -14 if k % 2 == 0 else -18, -0.5, out=0.4)
        put(shift(one('snare'), 3)[:, :int(.2 * SR)], t0 + k * e + e / 2, -24, -0.5, out=0.4)
    zurna = [('D5', 1), ('Eb5', .5), ('F#5', .5), ('G5', 1), ('A5', 1), ('G5', .5), ('F#5', .5), ('Eb5', 1), ('D5', 2)]
    tt = t0
    for n, du in zurna * 2:
        if tt > t0 + d: break
        y = inst('oboe').render(midi(n), du * e * 0.95, 0.9, 0.01, 0.1); y = np.tanh(y * 3) * 0.5
        put(y, tt, -20, 0.4, out=0.4); tt += du * e
    k = 0
    while True:
        tt = (math.asin(.6) + 2 * math.pi * k) / 8
        if tt >= d: break
        put(gunshot(0.3), t0 + tt, -16, 0.6, out=0.5); k += 1

def tmap_sfx(t0, d, p):
    m = p.get('m')
    if m in ('cars', 'spread'): put(lp(noise(d, 'brown'), 200) * 0.8, t0, -26); put(whoosh(d * 0.8, 200, 1500, 0.6), t0, -30)
    if m == 'phones':
        for k in range(int(d * 10)): put(chimes([int(rnd(84, 100))], 0.25, 'marimba'), t0 + d * 0.7 * (k / (d * 10)) ** 0.6, -36, rnd(-.9, .9))
    if m == 'route':
        put(pencil(min(d * 0.7, 3), 0.6), t0 + 0.2, -30)
        put(lp(one('bdrum_soft'), 900)[:, :int(.3 * SR)], t0 + d * 0.72, -26)
    if m == 'dogs':
        for k in range(14): put(chimes([int(rnd(74, 86))], 0.25, 'marimba'), t0 + rnd(0, d * 0.7), -34, rnd(-.9, .9))
    if m == 'ankara': put(helicopter(d, 1.0), t0, -28, out=0.4)
    if m == 'zpop':
        for k in range(22): put(lp(shift(one('darbuka'), rnd(-2, 4)), 3000)[:, :int(.2 * SR)], t0 + k * 0.07, -30, rnd(-.8, .8))
    if m == 'status':
        for at in (p.get('_k') or []): put(chimes([81, 88], 0.4, 'marimba', .1), t0 + at, -28)
    if m == 'pins':
        for k in range(12): put(lp(one('bdrum_soft'), 1200)[:, :int(.2 * SR)], t0 + k * 0.12, -32, rnd(-.8, .8))
    if m == 'safe':
        put(chimes([74, 81], 0.4, 'marimba', .2), t0 + 0.3, -32)

def garage_sfx(t0, d, m, p):
    if m == 'enter': put(steps(d, 1.3, 'stone', 0.8), t0, -24, hall=0.6); put(click(1.0, 1600), t0 + 0.15, -24, hall=0.4)
    if m == 'shine': put(one('marktree', 2)[:, :int(2 * SR)] * env_ar(int(2 * SR), 0, 1.5), t0 + 0.2, -30)
    if m == 'someone': put(lp(breath(d, 0.8, 0.4), 900), t0, -28, hall=0.5)
    if m == 'kemalZ': put(zgroan(), t0 + 0.3, -14, 0.4, hall=0.6)
    if m == 'stand': put(lp(zgroan(), 1200), t0 + d * 0.5, -22, 0.4, hall=0.6)
    if m == 'turn': put(steps(d, 0.9, 'stone', 0.4), t0, -30, 0.4, hall=0.5); put(zgroan(), t0 + d * 0.6, -16, 0.4, hall=0.6)
    if m == 'jump':
        put(steps(d * 0.6, 5, 'stone', 1.0, run=True), t0, -22, hall=0.5)
        put(thunk(1.0, 75), t0 + d * 0.62, -16, hall=0.4); put(metal_hit(0.3, 10, 0.2), t0 + d * 0.62, -24)

def radio_sfx(t0, d, m, p):
    if m in ('static', 'scan'): put(radio_tune(d) if m == 'scan' else static(d, 1.0, 1.2), t0, -20, 0.2)
    if m == 'lock': put(static(d, 0.4) * np.linspace(1, 0.2, int(d * SR))[None, :], t0, -24, 0.2); put(tone([700], 0.6, 0.25, None, True), t0 + 0.3, -28)
    if m == 'loop': put(static(d, 0.25), t0, -28, 0.2)
    if m == 'cast':
        put(static(d, 0.2), t0, -30, 0.2)
        if p.get('txt', '').startswith('BU BİR'): put(tone([853, 960], 1.0, 0.35, None, True), t0, -22)

def tunnel_sfx(t0, d, m, p):
    if m == 'memory':
        for k in range(4): put(lp(whoosh(1.2, 300, 1200, 1.0), 2500), t0 + k * 0.9, -24, hall=0.4)
    if m == 'beams': put(thunk(0.8, 120), t0 + 0.05, -24, hall=0.6)
    if m == 'cars':
        for k in range(int(d * 2)): put(grains(0.15, 200, (.002, .01), (800, 5000)), t0 + rnd(0, d), -32, hall=0.5)
    if m == 'busMove':
        put(lp(horde(d, 8), 1200), t0, -20, 0, hall=0.8)
        for k in range(int(d * 2)): put(lp(knock_wood(0.5, -4), 1500), t0 + rnd(0, d), -28, rnd(-.4, .4), hall=0.6)
    if m == 'stall':
        put(starter(1.2, False, 0.5), t0 + 0.1, -26, hall=0.4)
        e = engine(2.0, 800, 200, 4, 1.0, 0.2) * np.linspace(1, 0, int(2.0 * SR))[None, :]; put(e, t0, -22, hall=0.6)
        put(thunk(0.6, 100), t0 + 1.7, -26, hall=0.6)
    if m == 'dark':
        put(breath(d, 0.8, 0.6), t0, -30, hall=0.5)
    if m == 'hands':
        for k in range(int(d * 5)):
            tt = rnd(0, d); s = lp(knock_wood(rnd(.5, 1.0), -2), 2500) * 0.8
            sl = bp(noise(0.05, 'white'), 800, 3000) * np.exp(-np.linspace(0, 8, int(.05 * SR)))[None, :]; s[:, :sl.shape[1]] += sl * 0.5
            put(s, t0 + tt, -20, rnd(-.6, .6), hall=0.8)
        put(lp(horde(d, 6), 900), t0, -22, hall=0.8); put(breath(d, 1.0, 1.4), t0, -26)

# ------------------------------------------------------------------ car / engine events not tied to one shot
def car_events():
    # Kemal's Şahin: failed cranks, then start
    for s in shots('carIn'):
        t0, d, m = s['t0'], s['t1'] - s['t0'], s['p'].get('m')
        if m == 'fail':
            k = 0.0
            while k < d:
                put(click(0.8, 1500), t0 + k, -22); put(starter(0.55, False, 1.0), t0 + k + 0.05, -16, room=0.4); k += 1.1
        if m == 'hands':
            for h in s['p'].get('_k') or []: put(door_slam(0.8), t0 + h, -9, -0.4, room=0.4); put(lp(one('glass')[:, :int(.12 * SR)], 3000) * 0.3, t0 + h, -26)
        if m == 'start':
            put(click(0.8, 1500), t0, -22); put(starter(0.6, True, 1.0), t0 + 0.05, -16)
            put(engine(min(d, 3.0), 1200, 3800, 4, 1.4, 0.06) * env_ar(int(min(d, 3.0) * SR), 0.05, 0.8), t0 + 0.55, -12, room=0.3)

def render():
    ambience()
    for i, s in enumerate(SHOTS):
        transition_sfx(i, s)
        try:
            ev(i, s)
        except Exception as e:
            print('EV ERR', i, s['tpl'], s['p'].get('m'), repr(e))
    car_events()

def master():
    import pedalboard as pb
    print('reverbs...', flush=True)
    ir_room = make_ir(0.7, 0.008, 6500, 14, 0.7, seed=21)
    ir_hall = make_ir(3.6, 0.03, 4000, 10, 1.0, seed=22)
    ir_out = make_ir(1.4, 0.045, 7000, 6, 1.0, seed=23)
    mix = A.x + convolve(RM.x, ir_room) * 0.5 + convolve(HL.x, ir_hall) * 0.5 + convolve(OUT.x, ir_out) * 0.4
    mix = hp(mix, 25, 2)
    board = pb.Pedalboard([pb.Compressor(threshold_db=-16, ratio=2.5, attack_ms=5, release_ms=120), pb.Limiter(threshold_db=-3.0, release_ms=120)])
    return board(mix, SR) * db(-1.2)

if __name__ == '__main__':
    t0 = time.time(); render(); print('events %.0fs' % (time.time() - t0), flush=True)
    out = master(); np.save('sfx_raw.npy', out.astype(np.float32)); print('done %.0fs' % (time.time() - t0))
