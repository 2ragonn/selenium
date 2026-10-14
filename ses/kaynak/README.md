# Ses üretim kaynakları

Bu klasördeki betikler `ses/` altındaki müzik ve SFX kanallarını üretir.

- `timeline.json`: animasyondan çıkarılan sahne, kelime, çekim ve Mors zamanlamaları.
- `sfx_track.py`: senkron SFX kanalını üretir; tarifler `sfxkit.py` içinde, DSP çekirdeği `lib.py` içinde.
- `music.py` + `mscore.py`: partisyonu MIDI olarak yazar. Bu MIDI fluidsynth ile MuseScore General soundfont'unda çalınır.
- `master.py`: yankı, EQ ve kompresyon uygular, LUFS hedefine getirir ve MP3'e çevirir.

Gereksinimler: `apt install fluidsynth musescore-general-soundfont-lossless`, `pip install numpy scipy numba soundfile mido pyloudnorm`. GeneralUser GS soundfont'u `assets/GeneralUser-GS.sf2` olarak koyun.

Çalıştırma sırası:

```
python3 sfx_track.py
python3 music.py
fluidsynth -ni -R 0 -C 0 -g 0.6 -r 48000 -O float -o synth.midi-bank-select=gs -F out/music_dry.wav MuseScore_General_Full.sf2 out/music.mid
python3 master.py
```

`master.py` çalışmadan önce `out/music_dry.wav` dosyası `out/music_dry.npy` olarak kaydedilmelidir.
