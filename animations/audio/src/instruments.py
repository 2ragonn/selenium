"""Instrument definitions (all CC0 recordings: VSCO-2 CE, VCSL, Sonic Pi/freesound CC0)."""
import glob, os
from core import *

def V(*p): return os.path.join(VSCO, *p)
def C(*p): return os.path.join(VCSL, *p)

_inst = {}
def inst(name):
    if name in _inst: return _inst[name]
    d = {
        'vln':      lambda: Sampler(V('Strings', 'Violin Section', 'susVib', '*.wav'), 'vln', release=0.9),
        'vla':      lambda: Sampler(V('Strings', 'Viola Section', 'susvib', '*.wav'), 'vla', release=0.9),
        'vc':       lambda: Sampler(V('Strings', 'Cello Section', 'susvib', '*.wav'), 'vc', release=0.9),
        'cb':       lambda: Sampler(V('Strings', 'Solo Contrabass', 'SusNV', '*.wav'), 'cb', release=1.0),
        'vln_trem': lambda: Sampler(V('Strings', 'Violin Section', 'Trem', '*.wav'), 'vln_trem', release=0.7),
        'vla_trem': lambda: Sampler(V('Strings', 'Viola Section', 'trem', '*.wav'), 'vla_trem', release=0.7),
        'vc_trem':  lambda: Sampler(V('Strings', 'Cello Section', 'trem', '*.wav'), 'vc_trem', release=0.7),
        'cb_trem':  lambda: Sampler(V('Strings', 'Solo Contrabass', 'Trem', '*.wav'), 'cb_trem', release=0.8),
        'vln_pizz': lambda: Sampler(V('Strings', 'Violin Section', 'Pizz', '*.wav'), 'vln_pizz', release=0.3, loop=False),
        'vc_pizz':  lambda: Sampler(V('Strings', 'Cello Section', 'pizzT', '*.wav'), 'vc_pizz', octave_fix=12, release=0.4, loop=False),
        'cb_pizz':  lambda: Sampler(V('Strings', 'Solo Contrabass', 'Pizz', '*.wav'), 'cb_pizz', release=0.5, loop=False),
        'vln_spic': lambda: Sampler(V('Strings', 'Violin Section', 'Spic', '*.wav'), 'vln_spic', release=0.15, loop=False),
        'vc_spic':  lambda: Sampler(V('Strings', 'Cello Section', 'spic', '*.wav'), 'vc_spic', release=0.15, loop=False),
        'solo_vln': lambda: Sampler(V('Strings', 'Solo Violin', 'Arco Vib', '*.wav'), 'solo_vln', release=0.6),
        'harp':     lambda: Sampler(V('Strings', 'Harp', '*.wav'), 'harp', release=1.5, loop=False),
        'horn':     lambda: Sampler(V('Brass', 'F Horn', 'sus', '*.wav'), 'horn', release=0.7),
        'tuba':     lambda: Sampler(V('Brass', 'Tuba', 'sus', '*.wav'), 'tuba', release=0.7),
        'tbn':      lambda: Sampler(V('Brass', 'Tenor Trombone', 'sus', '*.wav'), 'tbn', release=0.6),
        'tbn_fall': lambda: Sampler(V('Brass', 'OldTrombone', 'Fall', '*.wav'), 'tbn_fall', release=0.4, loop=False),
        'tbn_buzz': lambda: Sampler(V('Brass', 'OldTrombone', 'Buzz', '*.wav'), 'tbn_buzz', release=0.4, loop=False),
        'flute':    lambda: Sampler(V('Woodwinds', 'Flute', 'susNV', '*.wav'), 'flute', release=0.5),
        'flute_v':  lambda: Sampler(V('Woodwinds', 'Flute', 'expvib', '*.wav'), 'flute_v', release=0.5),
        'clar':     lambda: Sampler(V('Woodwinds', 'Clarinet', 'susLong', '*.wav'), 'clar', release=0.5),
        'oboe':     lambda: Sampler(V('Woodwinds', 'Oboe', 'Vib', '*.wav'), 'oboe', release=0.4),
        'bsn':      lambda: Sampler(V('Woodwinds', 'Bassoon', 'sus', '*.wav'), 'bsn', octave_fix=12, release=0.4),
        'bsn_stac': lambda: Sampler(V('Woodwinds', 'Bassoon', 'stac', '*.wav'), 'bsn_stac', release=0.2, loop=False),
        'piano':    lambda: Sampler(C('Chordophones', 'Zithers', 'Grand Piano, Steinway B', 'Sus', 'JHPiano_Sus_Close_*.wav'), 'piano', octave_fix=0, release=1.2, loop=False),
        'glock':    lambda: Sampler(V('Percussion', 'Glock', '*.wav'), 'glock', octave_fix=12, release=1.0, loop=False),
        'marimba':  lambda: Sampler(V('Percussion', 'Marimba', '*.wav'), 'marimba', release=0.6, loop=False),
        'tbells':   lambda: Sampler(C('Idiophones', 'Struck Idiophones', 'Tubular Bells 1', '*.wav'), 'tbells', release=2.5, loop=False),
    }
    _inst[name] = d[name]()
    return _inst[name]

def files(pattern):
    fs = sorted(glob.glob(pattern, recursive=True))
    if not fs: raise RuntimeError('no files ' + pattern)
    return fs

# one-shot pools ---------------------------------------------------------------
POOLS = {}
def pool(name):
    if name in POOLS: return POOLS[name]
    P = {
        'bdrum':     lambda: files(V('Percussion', 'BDrumNewhit_v[5-7]*.wav')),
        'bdrum_soft':lambda: files(V('Percussion', 'BDrumNewhit_v[1-3]*.wav')),
        'bd_vcsl':   lambda: files(C('Membranophones', 'Struck Membranophones', 'Bass Drum 2', 'bassdrum_hit_*.wav')),
        'bd_roll':   lambda: files(C('Membranophones', 'Struck Membranophones', 'Bass Drum 2', 'bassdrum_roll_*.wav')),
        'timp_hit':  lambda: [f for f in files(V('Percussion', 'Timpani', 'Timpani*Hit*.wav'))],
        'timp_roll': lambda: files(V('Percussion', 'Timpani', 'Rolls', 'Timpani*Roll*.wav')),
        'gong':      lambda: files(V('Percussion', 'gongHit*.wav')),
        'gong_scrape': lambda: files(V('Percussion', 'gongscrape*.wav')) + files(C('Idiophones', 'Struck Idiophones', 'Gong 1', '*scrape*.wav')),
        'gong_big':  lambda: files(C('Idiophones', 'Struck Idiophones', 'Gong 1', 'gong_*f*.wav')),
        'sus_cymb':  lambda: files(V('Percussion', 'susCymb*.wav')),
        'cresc':     lambda: [f for f in files(V('Percussion', 'susCymb*.wav')) + files(C('Idiophones', 'Struck Idiophones', 'Suspended Cymbal 1', '*.wav')) if 'cresc' in f.lower()],
        'cymb_bow':  lambda: [f for f in files(V('Percussion', 'susCymb*.wav')) if 'bow' in f.lower()],
        'cymbal':    lambda: files(V('Percussion', 'cymbal*.wav')),
        'snare':     lambda: files(V('Percussion', 'Snare*.wav')),
        'v1drums':   lambda: files(V('VSCO 1 Percussion', 'drums*', '*.wav')) if glob.glob(V('VSCO 1 Percussion', 'drums*', '*.wav')) else files(V('VSCO 1 Percussion', '**', '*.wav')),
        'anvil':     lambda: files(V('Percussion', 'Anvil*.wav')),
        'brake':     lambda: files(V('Percussion', 'BrakeDrum*.wav')),
        'ratchet':   lambda: files(V('Percussion', 'Ratchet*.wav')),
        'claves':    lambda: files(V('Percussion', 'Claves*.wav')),
        'logdrum':   lambda: files(V('Percussion', 'LogDrum*.wav')),
        'tamb':      lambda: files(V('Percussion', 'Tamb*.wav')),
        'triangle':  lambda: files(V('Percussion', 'Triangle*.wav')),
        'belltree':  lambda: files(V('Percussion', 'BellTree*.wav')),
        'cowbell':   lambda: files(V('Percussion', 'Cowbell*.wav')),
        'glass':     lambda: files(V('Miscellania Raw', 'Misc 1', 'glass_break*.wav')),
        'metal':     lambda: files(V('Miscellania Raw', 'Misc 1', 'metal_hit*.wav')),
        'siren_v':   lambda: files(V('Miscellania Raw', 'Misc 1', 'siren*.wav')),
        'chain':     lambda: files(V('Miscellania Raw', 'Misc 1', 'chain*.wav')),
        'brick':     lambda: files(V('Miscellania Raw', 'Misc 1', 'brick_scrape*.wav')),
        'bubbles':   lambda: files(V('Miscellania Raw', 'Misc 1', 'bubbles*.wav')),
        'ambience1': lambda: files(V('Miscellania Raw', 'Misc 1', 'ambience1.wav')),
        'alien':     lambda: files(V('Miscellania Raw', 'Misc 1', 'alien*.wav')),
        'timpsnare': lambda: files(V('Miscellania Raw', 'Misc 1', 'timp*snare*.wav')),
        'siren':     lambda: files(C('Aerophones', 'Free Aerophones', 'Siren', '*.wav')),
        'didge':     lambda: files(C('Aerophones', 'Lip Aerophones', 'Didgeridoo', '*.wav')),
        'darbuka':   lambda: files(C('Membranophones', 'Struck Membranophones', 'Darbuka', '*.wav')),
        'framedrum': lambda: files(C('Membranophones', 'Struck Membranophones', 'Frame Drum', '*.wav')),
        'ocean':     lambda: files(C('Membranophones', 'Other Membranophones', 'Ocean Drum', '*.wav')),
        'wineglass': lambda: files(C('Idiophones', 'Friction Idiophones', 'Wine Glasses', 'Sustains', '*', '*.wav')),
        'woodblock': lambda: files(C('Idiophones', 'Struck Idiophones', 'Woodblock', '*.wav')),
        'slapstick': lambda: files(C('Idiophones', 'Struck Idiophones', 'Slapstick', '*.wav')),
        'marktree':  lambda: files(C('Idiophones', 'Struck Idiophones', 'Mark Trees', '**', '*.wav')),
        'vibraslap': lambda: files(C('Idiophones', 'Struck Idiophones', 'Vibraslap', '**', '*.wav')),
        'flexatone': lambda: files(C('Idiophones', 'Struck Idiophones', 'Flexatone', '*.wav')),
        'nepal':     lambda: files(C('Idiophones', 'Struck Idiophones', 'Hand Bells, Nepalese', '*.wav')),
        'vcsl_anvil':lambda: files(C('Idiophones', 'Struck Idiophones', 'Anvil', '*.wav')),
        'vcsl_gong': lambda: files(C('Idiophones', 'Struck Idiophones', 'Gong 1', '*.wav')),
        'vcsl_cymb': lambda: files(C('Idiophones', 'Struck Idiophones', 'Suspended Cymbal 1', '*.wav')),
        'cineboom':  lambda: [os.path.join(SP, 'misc_cineboom.flac')],
        'impact':    lambda: [os.path.join(SP, 'perc_impact1.flac'), os.path.join(SP, 'perc_impact2.flac')],
        'darkwoosh': lambda: [os.path.join(SP, 'ambi_dark_woosh.flac')],
        'swoosh':    lambda: [os.path.join(SP, 'perc_swoosh.flac'), os.path.join(SP, 'ambi_swoosh.flac')],
        'swash':     lambda: [os.path.join(SP, 'perc_swash.flac')],
        'haunted':   lambda: [os.path.join(SP, 'ambi_haunted_hum.flac')],
        'drone':     lambda: [os.path.join(SP, 'ambi_drone.flac'), os.path.join(SP, 'ambi_lunar_land.flac')],
        'glassrub':  lambda: [os.path.join(SP, 'ambi_glass_rub.flac')],
        'crow':      lambda: [os.path.join(SP, 'misc_crow.flac')],
        'door_sp':   lambda: [os.path.join(SP, 'perc_door.flac')],
        'vinyl':     lambda: [os.path.join(SP, 'vinyl_hiss.flac')],
        'whistle':   lambda: files(C('Aerophones', 'Edge-blown Aerophones', 'Ball Whistle', '*.wav')),
        'snareroll': lambda: [os.path.join(SP, 'drum_roll.flac')],
    }
    POOLS[name] = P[name]()
    return POOLS[name]

def one(name, i=None):
    fs = pool(name)
    f = fs[int(RNG.integers(len(fs)))] if i is None else fs[i % len(fs)]
    return load(f)
