# Audio for `turkiye-zombi-salgini.html`

Two separate stems, both exactly as long as the animation (1161.08 s), starting at the animation's t = 0:

| File | What it is | Level |
|---|---|---|
| `turkiye-zombi-salgini-muzik.mp3` | Orchestral background score (strings, brass, woodwinds, Steinway piano, harp, timpani, gongs, darbuka/frame drum). Follows the story: main theme, Kemal's piano theme, a Hicaz (D–E♭–F♯–G–A–B♭–C) theme for Cappadocia/underground, a hope theme for the reunion, impacts on every day card. | −18 LUFS, peak −1.2 dBFS |
| `turkiye-zombi-salgini-sfx.mp3` | Ambience beds per location + ~600 events synced to the animation's shots (door pounding on the "GÜM" hits, gunshots locked to the muzzle flashes, glass, shutter, car cranking/starting, tunnel, village, underground). | −24 LUFS, peak −1.2 dBFS |

Both leave headroom for the voice-over; the music also has a gentle cut around 2.8 kHz so the voice sits on top.

## Sources (all public domain / CC0 — no attribution required)
- **VSCO-2 Community Edition** (Versilian Studios) — CC0
- **Versilian Community Sample Library (VCSL)** — CC0
- **Sonic Pi bundled samples** (from freesound.org, CC0)

Everything else (wind, water, fire, creaks, engines, footsteps, radio static, phone tones) is physically modelled noise-based foley generated in `src/`.

## Regenerating
`src/` holds the generator. It expects the sample libraries next to it under `libs/`
(`git clone https://github.com/sgossner/VSCO-2-CE libs/VSCO-2-CE`, `git clone https://github.com/sgossner/VCSL libs/VCSL`)
and the Sonic Pi samples under `sp/`. Then: `python3 music.py` and `python3 sfx.py` (needs numpy, scipy, soundfile, pedalboard).
`timeline.json` is the shot/word timeline exported from the HTML, which is what keeps the audio in sync.
