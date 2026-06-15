"""
Symbolic Pianistic Voicing Solver — Non-Destructive Cluster & Sympathetic Resonance Pruner

Addresses sympathetic string resonance under sustain pedal:
When a pianist plays a chord with sustain pedal, sympathetic resonance can cause
acoustic energy at adjacent pitch bins.

Non-Destructive & Musically Accurate Principles:
  1. Never delete authentic chromatic runs or dissonant cluster chords simply because
     they contain consecutive semitones. Authentic piano literature (Chopin, Bach, jazz, blues)
     frequently contains chromatic runs and seconds.
  2. Dynamic piece-adaptive register separation: uses median pitch rather than a rigid
     middle-C (60) split.
  3. Never delete acoustic notes solely because a cluster span exceeds 16 semitones.
     Wide-spread chords are played by two hands or arpeggiated rolled chords.
  4. Only prune notes when a single hand physically exceeds 5 concurrent notes,
     pruning the candidate with the lowest acoustic velocity and chord affinity.
  5. Supports diagnostic mode: records pruned or flagged candidates for user inspection
     without destructive deletion.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
import statistics
import pretty_midi


# Maximum notes a single hand can physically strike simultaneously
MAX_NOTES_PER_HAND = 5

# Cluster onset tolerance (same as chord alignment)
CLUSTER_TOLERANCE = 0.035

# Chord tone intervals relative to root (in semitones)
CHORD_TONE_PRIORITY = {
    0: 10.0,   # Root
    7: 8.0,    # Perfect 5th
    4: 7.5,    # Major 3rd
    3: 7.5,    # Minor 3rd
    11: 6.0,   # Major 7th
    10: 6.0,   # Minor 7th (dominant 7th)
    2: 5.0,    # Major 2nd / 9th
    5: 4.5,    # Perfect 4th / 11th
    9: 4.5,    # Major 6th / 13th
    8: 4.0,    # Minor 6th / augmented 5th
    6: 3.5,    # Tritone / diminished 5th
    1: 3.0,    # Minor 2nd
}


def _compute_split_point(notes: List[pretty_midi.Note]) -> int:
    """Computes dynamic piece split point based on median pitch."""
    if not notes:
        return 60
    median_p = statistics.median(n.pitch for n in notes)
    split = int(round(median_p * 0.25 + 60 * 0.75))
    return max(48, min(72, split))


def _score_note_in_context(
    note: pretty_midi.Note,
    chord_root_pc: Optional[int],
    cluster_velocities: List[int],
) -> float:
    """
    Compute a retention score for a note within a simultaneous cluster.
    Higher score = more likely to be an intentional, struck note.
    """
    score = 0.0

    # 1. Velocity contribution — louder notes are more likely real strikes
    v_min = min(cluster_velocities) if cluster_velocities else 40
    v_max = max(cluster_velocities) if cluster_velocities else 100
    v_range = max(1, v_max - v_min)
    velocity_norm = (note.velocity - v_min) / v_range
    score += velocity_norm * 6.0

    # 2. Chord tone affinity — if we know the chord root, prefer chord tones
    if chord_root_pc is not None:
        interval = (note.pitch % 12 - chord_root_pc) % 12
        score += CHORD_TONE_PRIORITY.get(interval, 3.0)
    else:
        score += 4.0

    return score


def prune_cluster_voicings(
    notes: List[pretty_midi.Note],
    split_point: int = 60,
    chord_root_pc: Optional[int] = None,
    max_per_hand: int = MAX_NOTES_PER_HAND,
) -> Tuple[List[pretty_midi.Note], List[pretty_midi.Note]]:
    """
    Prunes notes only when a single hand cluster exceeds physical finger count (5 notes).
    Does NOT delete authentic chromatic runs or wide-spanning two-hand chords.
    
    Returns:
        (kept_notes, pruned_notes)
    """
    if not notes:
        return notes, []

    # Dynamic split into Left Hand and Right Hand candidate groups
    lh_notes = [n for n in notes if n.pitch < split_point]
    rh_notes = [n for n in notes if n.pitch >= split_point]

    kept: List[pretty_midi.Note] = []
    pruned: List[pretty_midi.Note] = []

    for hand_notes in [lh_notes, rh_notes]:
        if not hand_notes:
            continue

        # If playable within human hand fingers, preserve 100% of notes
        if len(hand_notes) <= max_per_hand:
            kept.extend(hand_notes)
            continue

        # If a single hand has > 5 concurrent strikes, rank by velocity and musical context
        velocities = [n.velocity for n in hand_notes]
        scored = [
            (_score_note_in_context(n, chord_root_pc, velocities), n)
            for n in hand_notes
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Keep top max_per_hand notes, prune excess
        kept_hand = [n for _, n in scored[:max_per_hand]]
        pruned_hand = [n for _, n in scored[max_per_hand:]]

        kept.extend(kept_hand)
        pruned.extend(pruned_hand)

    return kept, pruned


def apply_voicing_solver(
    notes: List[pretty_midi.Note],
    tolerance: float = CLUSTER_TOLERANCE,
    diagnostic_mode: bool = False
) -> Tuple[List[pretty_midi.Note], int]:
    """
    Main entry point: groups notes into onset clusters and applies the
    voicing solver to prune physically impossible simultaneous clusters.

    Returns:
        (output_notes, total_pruned_count)
    """
    if not notes:
        return notes, 0

    sorted_notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    split_point = _compute_split_point(sorted_notes)

    # Group into onset clusters
    clusters: List[List[pretty_midi.Note]] = []
    current_cluster = [sorted_notes[0]]

    for n in sorted_notes[1:]:
        if (n.start - current_cluster[0].start) <= tolerance:
            current_cluster.append(n)
        else:
            clusters.append(current_cluster)
            current_cluster = [n]
    clusters.append(current_cluster)

    total_pruned = 0
    result: List[pretty_midi.Note] = []

    for cluster in clusters:
        # A cluster with 5 or fewer total notes is physically playable by two hands
        if len(cluster) <= MAX_NOTES_PER_HAND:
            result.extend(cluster)
            continue

        pitches = [n.pitch for n in cluster]
        bass_pc = min(pitches) % 12

        kept, pruned_notes = prune_cluster_voicings(
            cluster,
            split_point=split_point,
            chord_root_pc=bass_pc,
            max_per_hand=MAX_NOTES_PER_HAND
        )
        total_pruned += len(pruned_notes)

        if diagnostic_mode:
            # Keep all notes in diagnostic mode
            result.extend(cluster)
        else:
            result.extend(kept)

    return result, total_pruned
