"""
Musical Rhythm Quantizer & Chord Aligner

Converts raw continuous analog transcription timestamps into clean,
learnable musical notes with:
  1. Synchronous Chord Alignment (snapping near-simultaneous chord strikes to exact same time)
  2. Beat Grid Quantization (1/16, 1/8, or 1/4 note snapping based on detected BPM)
  3. Duration Cohesion (standardizing ragged note tails to musical note lengths)
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pretty_midi


def estimate_tempo_bpm(audio: np.ndarray, sr: int = 16000, default_bpm: float = 120.0) -> float:
    """Estimates musical tempo (BPM) from audio waveform."""
    try:
        import librosa
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        bpm = float(tempo[0] if isinstance(tempo, (np.ndarray, list)) else tempo)
        if 40.0 <= bpm <= 240.0:
            return round(bpm, 1)
        return default_bpm
    except Exception:
        return default_bpm


def align_chord_onsets(notes: List[pretty_midi.Note], tolerance: float = 0.025) -> List[pretty_midi.Note]:
    """
    Snaps notes struck nearly simultaneously (within tolerance) to the exact same onset time.
    Eliminates tilted/crooked waterfall visualizer chords so they display as clean polyphonic bars.
    Preserves fast melodic runs: does not snap consecutive scalar steps (<= 2 semitones).
    """
    if not notes:
        return notes

    sorted_notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    aligned = []
    i = 0
    n = len(sorted_notes)

    while i < n:
        cluster = [sorted_notes[i]]
        j = i + 1
        while j < n and (sorted_notes[j].start - sorted_notes[i].start) <= tolerance:
            # Check if this note is a rapid scalar step of the previous note in the cluster
            is_scalar_step = any(abs(sorted_notes[j].pitch - member.pitch) <= 2 and (sorted_notes[j].start > member.start) for member in cluster)
            if is_scalar_step:
                # Do not snap fast scalar runs into block chords
                break
            cluster.append(sorted_notes[j])
            j += 1

        # Use median onset time of the cluster for robustness against outliers
        mean_start = round(float(np.median([note.start for note in cluster])), 4)
        for note in cluster:
            dur = max(0.05, note.end - note.start)
            note.start = mean_start
            note.end = max(note.start + 0.05, mean_start + dur)
            aligned.append(note)
        i = j

    return aligned


def quantize_note_rhythm(
    notes: List[pretty_midi.Note],
    bpm: float = 120.0,
    subdivision: str = "1/16",
    snap_strength: float = 0.60
) -> List[pretty_midi.Note]:
    """
    Musical grid quantization:
    Snaps raw continuous note onsets towards musical grid subdivisions while
    preserving human rubato dynamics.
    
    Args:
        notes: List of PrettyMIDI Note objects.
        bpm: Song tempo in beats per minute.
        subdivision: "1/16", "1/8", or "1/4".
        snap_strength: 0.0 = no snap, 1.0 = hard robotic snap, 0.6 = natural musical snap.
    """
    if not notes or bpm <= 0:
        return notes

    # Beat duration in seconds (quarter note)
    beat_sec = 60.0 / bpm
    if subdivision == "1/16":
        grid_sec = beat_sec / 4.0
    elif subdivision == "1/8":
        grid_sec = beat_sec / 2.0
    else:  # 1/4
        grid_sec = beat_sec

    # Step 1: Align chords first
    aligned_notes = align_chord_onsets(notes, tolerance=0.035)

    # Step 2: Soft musical grid snap
    for note in aligned_notes:
        grid_idx = round(note.start / grid_sec)
        target_grid_time = grid_idx * grid_sec
        delta = target_grid_time - note.start

        # Only snap if within 60% of a grid interval to avoid pulling notes across beats
        if abs(delta) <= (grid_sec * 0.60):
            old_dur = max(0.05, note.end - note.start)
            note.start = round(note.start + (delta * snap_strength), 4)
            # Duration snapping to nearest half-grid
            dur_grids = max(1, round(old_dur / (grid_sec * 0.5)))
            target_dur = dur_grids * (grid_sec * 0.5)
            new_dur = old_dur + ((target_dur - old_dur) * snap_strength)
            note.end = round(note.start + max(0.06, new_dur), 4)

    return aligned_notes
