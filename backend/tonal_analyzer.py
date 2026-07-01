"""
Tonal Key Center Analyzer — Krumhansl-Schmuckler Algorithm
Identifies the musical key signature (e.g. E Minor, G# Minor, D Minor)
and provides diatonic scale masks to prevent accidental note hallucinations.
"""
from typing import List, Tuple, Dict, Any, Set
import numpy as np

PITCH_CLASS_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Kessler Key Profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Diatonic Scale Degrees (semitones from root)
MAJOR_INTERVALS = {0, 2, 4, 5, 7, 9, 11}
NATURAL_MINOR_INTERVALS = {0, 2, 3, 5, 7, 8, 10}
HARMONIC_MINOR_INTERVALS = {0, 2, 3, 5, 7, 8, 11}
MELODIC_MINOR_INTERVALS = {0, 2, 3, 5, 7, 9, 11}


def detect_key_signature(notes: List[Any]) -> Dict[str, Any]:
    """
    Computes weighted pitch-class distribution and correlates with 24 major/minor keys.
    Returns: {
        'root': 'E',
        'mode': 'minor',
        'key_name': 'E Minor',
        'scale_pitch_classes': {4, 6, 7, 9, 11, 0, 2},
        'confidence': 0.84
    }
    """
    if not notes:
        return {
            'root': 'C',
            'mode': 'major',
            'key_name': 'C Major',
            'scale_pitch_classes': {0, 2, 4, 5, 7, 9, 11},
            'confidence': 0.0
        }

    # Step 1: Compute weighted chroma distribution (duration * velocity)
    chroma = np.zeros(12, dtype=np.float64)
    for n in notes:
        if isinstance(n, dict):
            pitch = n.get('midi') if n.get('midi') is not None else n.get('pitch', 60)
            start = n.get('time', n.get('start', 0.0))
            end = n.get('end', start + n.get('duration', 0.5))
            vel = n.get('velocity', 64)
        else:
            pitch = n.pitch
            start = n.start
            end = n.end
            vel = n.velocity
        if isinstance(vel, float) and vel <= 1.0:
            vel = int(vel * 127)

        dur = max(0.05, end - start)
        pc = int(pitch) % 12
        chroma[pc] += dur * (vel / 127.0)

    total_energy = np.sum(chroma)
    if total_energy > 0:
        chroma /= total_energy

    # Step 2: Correlate with 24 keys
    best_corr = -2.0
    best_key = ('C', 'major')

    for root_pc in range(12):
        # Rotate profiles so the profile's root (index 0) shifts to index root_pc
        rot_major = np.roll(MAJOR_PROFILE, root_pc)
        rot_minor = np.roll(MINOR_PROFILE, root_pc)

        # Pearson correlation
        corr_maj = np.corrcoef(chroma, rot_major)[0, 1]
        corr_min = np.corrcoef(chroma, rot_minor)[0, 1]

        if not np.isnan(corr_maj) and corr_maj > best_corr:
            best_corr = corr_maj
            best_key = (PITCH_CLASS_NAMES[root_pc], 'major')

        if not np.isnan(corr_min) and corr_min > best_corr:
            best_corr = corr_min
            best_key = (PITCH_CLASS_NAMES[root_pc], 'minor')

    root, mode = best_key
    root_pc = PITCH_CLASS_NAMES.index(root)

    # Step 3: Determine valid scale pitch classes (including harmonic/melodic minor inflections)
    if mode == 'major':
        scale_pcs = set((root_pc + interval) % 12 for interval in MAJOR_INTERVALS)
    else:
        # For minor keys, union natural, harmonic, and melodic minor forms
        scale_pcs = set((root_pc + interval) % 12 for interval in NATURAL_MINOR_INTERVALS | HARMONIC_MINOR_INTERVALS | MELODIC_MINOR_INTERVALS)

    STANDARD_KEY_NAMES = {
        ('A#', 'major'): 'Bb',
        ('D#', 'major'): 'Eb',
        ('G#', 'major'): 'Ab',
        ('C#', 'major'): 'Db',
        ('A#', 'minor'): 'Bb',
        ('D#', 'minor'): 'Eb',
    }
    std_root = STANDARD_KEY_NAMES.get((root, mode), root)
    key_display = f"{std_root} {mode.capitalize()}"
    conf = round(max(0.0, min(1.0, float(best_corr))), 3) if not np.isnan(best_corr) and best_corr > -1.0 else 0.0

    return {
        'root': std_root,
        'mode': mode,
        'key_name': key_display,
        'scale_pitch_classes': scale_pcs,
        'confidence': conf
    }
