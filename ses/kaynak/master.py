"""Mastering: müzik (salon yankısı + ses için EQ boşluğu + yumuşak kompresyon) ve SFX; LUFS hedefi, tepe sınırlama, MP3."""
import os, sys, subprocess
import numpy as np, soundfile as sf, pyloudnorm as pyln
from lib import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
meter = pyln.Meter(SR)
which = sys.argv[1:] or ['music', 'sfx']


def finish(x, target_lufs, name, ceil=-1.0):
    lu = meter.integrated_loudness(x.T)
    x = x * db(target_lufs - lu)
    x = limiter(x, ceil, 3.0, 120.0)
    lu2 = meter.integrated_loudness(x.T)
    x = x * db(target_lufs - lu2)
    x = limiter(x, ceil, 3.0, 120.0)
    pk = 20 * np.log10(np.max(np.abs(x)))
    print(name, 'LUFS %.2f -> %.2f, tepe %.2f dBFS' % (lu, meter.integrated_loudness(x.T), pk))
    wav = os.path.join(OUT, name + '.wav')
    sf.write(wav, x.T.astype(np.float32), SR, subtype='PCM_24')
    mp3 = os.path.join(OUT, name + '.mp3')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav, '-codec:a', 'libmp3lame', '-b:a', '320k', '-ar', '48000', mp3], check=True)
    return x


if 'music' in which:
    m = np.load(os.path.join(OUT, 'music_dry.npy')).astype(np.float64)
    n = m.shape[1]
    hall = make_ir(2.6, 0.022, 0.08, 14, damp=1.2, width=1.0, seed=11)
    wet = convolve(m, hall)[:, :n]
    x = m * 0.82 + wet * 0.30
    x = hp(x, 32, 2)
    x = peq(x, 2600, 0.8, -2.5)     # anlatıcı sesine yer aç
    x = peq(x, 220, 0.9, 1.0)       # sıcaklık
    x = shelf(x, 9000, -1.5, True)
    x = compress(x, thr_db=-24, ratio=2.2, att_ms=40, rel_ms=500)
    # sona: kuyruğu yumuşak kıs
    end_n = T(1438.4)
    x = x[:, :end_n]
    x[:, -T(1.5):] *= np.linspace(1, 0, T(1.5)) ** 2
    finish(x, -24.0, 'uzayli_istilasi_muzik')

if 'sfx' in which:
    s = np.load(os.path.join(OUT, 'sfx_raw.npy')).astype(np.float64)
    s = s[:, :T(1438.4)]
    s = hp(s, 24, 2)
    s[:, -T(1.0):] *= np.linspace(1, 0, T(1.0)) ** 2
    finish(s, -25.0, 'uzayli_istilasi_sfx')
