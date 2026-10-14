# Audio for `turkiye-zombi-salgini.html`

Two separate stems (MP3, 320 kbps, 48 kHz stereo), both **1158.08 s** long and locked to the animation timeline (start them at 0:00 together with the animation / voice-over).

| File | What it is | Loudness |
|---|---|---|
| `turkiye-zombi-salgini_sfx.mp3` | Background SFX: room tones and ambiences for every scene, Foley (doors, steps, keys, paper), vehicles, dogs, crowds, zombies, radio/phone, weather, plus soft transition accents on cuts | −24 LUFS integrated, true peak ≤ −2 dBTP |
| `turkiye-zombi-salgini_music.mp3` | Background score: hybrid orchestral (strings, piano, brass, timpani, harp) with Anatolian colour (ney, bağlama, darbuka, 9/8 and 7/16 folk grooves) | −20 LUFS integrated, true peak ≤ −2 dBTP |

## Mixing under the voice-over

- Put the voice-over around −16 LUFS, then start with music at about −8 dB and SFX at about −4 dB relative to the files as delivered, and adjust to taste.
- The SFX track never drops out to digital silence: a quiet room-tone/air underlay runs under every act, so cuts between scene ambiences stay seamless.
- 1097–1110 s (the ezan moment) is left silent in the music on purpose.

## How it was made

Everything is generated offline from code. No commercial sound libraries and no copyrighted recordings are used.

- **Sound effects** are physically modelled rather than built from beeps and plain oscillators. Impacts use band-shaped noise with per-band decay. Creaks are stick-slip friction pulse trains. Engines are built from individual combustion events. Voices and animals use a glottal source with formants, and barks and growls are shaped by classifier-searched spectro-temporal envelopes. Rooms use convolution reverbs.
- **Music** is performed with the CC0 **VSCO-2 Community Edition** orchestral samples by Versilian Studios (credit appreciated, not required), plus a modelled bağlama and frame drums.
- **Realism check:** every 6-second window of both stems was run through an AudioSet sound classifier (YAMNet). Sounds that came back as "synthesizer", "siren", "baby cry" and so on were redesigned until the windows read as the intended real-world sources.
