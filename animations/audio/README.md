# Nazar Boncuğu Kahramanı: background music and SFX

There are two separate stems for `../nazar-boncugu-kahramani.html`. Each one is exactly
**17:17.13 (1037.13 s)** long, the same as the animation and the voice-over, and both start at 0:00.

| File | What it is |
|---|---|
| `nazar-boncugu-muzik.mp3` | Background music only (320 kbps, 48 kHz, −21 LUFS) |
| `nazar-boncugu-sfx.mp3` | Background SFX only (320 kbps, 48 kHz, −21 LUFS) |
| `onizleme-muzik-sfx-birlikte.mp3` | A preview with music and SFX mixed together, for checking the balance only (no voice) |

## No synthesizers

Every note and every effect is a **real acoustic recording**: orchestral and folk instrument
samples, real glass breaking, metal, water, fire, chains, wood knocks, slapsticks, whistles,
claps and crowds. They were edited, layered, pitched and reverberated. The only electronic
sound is the 3-beep Turkish phone busy tone at 1:16 (`DÜT DÜT DÜT`). That tone is electronic
in real life too, and it is filtered like a phone speaker.

### Music
The music follows the same scene timeline as the animation. A darbuka plays the maqsum rhythm,
and kanun-style zither, hicaz-mode melodies, strings, woodwinds, brass, harp, glockenspiel and
choir carry the themes. The main themes are a comic "Murat" theme, a hero theme, a heaven theme,
a pastoral theme and a sad theme. Stingers land on the big beats: the truck, the mirror, `PAT!`,
the victory and the day cards.

### SFX
The effect timings were taken from the animation code itself, by rendering every frame and
logging when each comic word (`ŞANGIRR!`, `BAM!`, `TAK!`, `CIZZZ!`, `PAT!`…), flash, shockwave
and pop-up appears. The median offset between an on-screen word and its sound is about 6 ms.

| Time | Moment | Sound |
|---|---|---|
| 0:01.75 | `KÜT!` the demon king falls | body fall, armour, dust |
| 0:09.66 | `PAT!` + screen crack | slapstick, glass crack |
| 0:19.2 | clock rewinds | vinyl rewind, ratchet winding, ticks |
| 0:21.62 / 0:56.42 | mirror cracks and shatters | layered real glass breaks + tinkles |
| 0:33.2 | `ÇIIIN!` ear ringing | hand chime + rubbed glass |
| 0:47.99 | `DIRRR!` alarm clock | real bells re-struck at 19 Hz |
| 1:10.3 | `DIRRR DIRRR!` mom calls | phone vibration + bell ring |
| 1:32.62 / 1:34.02 | `DÜÜÜT!` + `BAM!` bread truck | toy train whistle + trombones + tuba (horn), didgeridoo engine, impact stack, falling bread |
| 1:41.3 → | heaven | air, glass, wind chimes |
| 2:20.57 | `ÇAT!` mouse trap | slapstick + metal clank |
| 2:57.05 | `POF!` portal closes | reverse whoosh, smoke puff |
| every `DURUM PENCERESİ` | game status window | glockenspiel + hand chime motif |
| 4:34.77 | `PAT!` book slam | stamp |
| 5:23.9 → 5:27.09 | `ÇIT` → `KRRRK…` → `TUZ BUZ!` crystal ball | crack, creak, big shatter, then the hall goes silent |
| 6:28–6:36 | 14 glasses of tea, then the toilet | glass clinks, burp, flushes |
| 6:50.7 | `SAKAL YANIYOR!` | fire crackle, running |
| 6:56.87 | the open scissors land | metal clank |
| 7:03–7:13 | `KAVGA!` rat brawl | slapstick, wood, guiro scratches, growls |
| 7:30 / 7:43 | the child whistles at night | ocarina whistle tune |
| 7:54.9 | the lanterns go out one by one | five "fft" puffs, then the wind stops |
| 8:01.07 | the shadow demon rises | heartbeat, bowed cymbal, didgeridoo roar |
| 8:38.91 | `CIZZZ!` molten lead hits the water | real fuse sizzle, flare, bubbles, steam |
| 8:40.03 | `AAAAAAH!` the demon is sucked into the bowl | screech, suction, "gloop" |
| 9:18.61 | `YENİ YETENEK AÇILDI!` | level-up glockenspiel, bell tree, gong |
| 9:40.16 | royal guards knock | heavy door knocks |
| 9:56.59 | `ZİNDAN!` | chains, cell door |
| 10:31.65 | `SARAY ÇALKALANDI!` | crowd, applause, bells |
| 11:24.97 | the bead reflects the beam | spring zap, glass ting |
| 11:36.63 | `ÇIT!` the bead cracks | small glass crack |
| 12:00–12:12 | glassworks montage | furnace, anvil on the music's beat, glass tinks |
| 12:40.4 | the scout's horse | coconut-style gallop, `KÜT!` fall |
| 12:55.53 | the tea glass drops | glass shatter |
| 13:00.95 | the army stands up on the right foot | 60 layered stomps |
| 13:20.4–13:24.3 | the giant black cat walks | eight giant footsteps |
| 14:18.03 | `ŞLAP!` face-first into the mud | mud splash, slapstick, armour |
| 14:23.9 → 14:35.9 | the black wave rolls in | timpani, bass drum and didgeridoo rumble |
| 14:35.92 / 14:36.62 / 14:37.32 | `TAK! TAK! TAK!` | 10,000 shields: about 70 real wood knocks each |
| 14:43.95 | `ŞIIING!` sword drawn | metal scrape and ring |
| 15:05.2–15:08.1 | Mom's slipper ×5 | whoosh + slap |
| 15:14–15:18.47 | the slipper spins in the air | rotating whooshes |
| 15:18.47 | **`PAT!`** | the biggest slap + bell "bonk" |
| 15:26.37 | the demon king falls | huge body fall, then silence and one distant crow |
| 15:31.51 | `HURRAAAA!` | crowd, screams, applause, armour |
| 15:44–15:46.7 | the cat shrinks, shrinks, shrinks | descending slide whistles + puffs |
| 16:38.93 | `BZZZT!` the phone vibrates | vibration buzz |
| 17:13.03 | `ÇIIIN!` | ear ringing (callback) |

The 173 scene changes have their own whooshes, matched to the transition type:
- wipes sweep across the stereo field
- slices get a quick cabasa "shk-shk"
- iris transitions get a dark woosh
- the 11 blue nazar transitions get a cymbal swell into a glass ting

## Mixing suggestion

1. Voice-over around −16 to −14 LUFS.
2. Music fader around **−9 dB**. Raise it in the gaps between sentences if you like.
3. SFX fader around **−5 dB**. The big hits are meant to poke through, and the ambience sits low.
4. Keep everything aligned at 0:00. Both stems already contain the dramatic pauses
   (5:27 "Salon buz kesti", 12:55–12:58, 15:26–15:31 "Le sessizlik").

## Credits / licences

Everything is CC0 (public domain) except a few crowd, thunder and flush sounds, which are
**CC BY 4.0**. If you use these files, put this line in the video description:

> Some crowd, scream, thunder, pop and toilet-flush sounds: OpenRCT2 OpenSoundEffects
> (github.com/OpenRCT2/OpenSoundEffects): crowd by Ted John, others by Karst van Galen Last.
> CC BY 4.0, edited.

CC0 sources (no credit needed): VSCO 2 Community Edition and the Versilian Community Sample
Library (Versilian Studios), the Sonic Pi sample pack, and these CC0 recordings from the Luanti
`minetest_game` repository:
- water: Yuval
- lava bubbles: Halion
- fuse: theneedle.tv
- flare: frankelmedico
- furnace: iankath
- smoke: Ferk
- metal: qubodup, mypantsfelldown and Ogrebane
- twigs: blukotek
- chopping: Sheyvan
- gravel: lolamadeus
- footsteps: worthahep88 and Ryding
- gate creak: mhtaylor67
