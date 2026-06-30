"""
Pedagogical Piano Fingering Engine (Dynamic Programming / Ergonomic Viterbi Solver)

Replaces naive greedy heuristic with a principled ergonomic solver:
  1. Hand Biomechanics & Anatomy:
     - 5 fingers per hand: 1=Thumb, 2=Index, 3=Middle, 4=Ring, 5=Pinky.
     - Natural resting hand span and interval stretch limits.
  2. Textbook Classical Chords:
     - RH Root Triad (e.g. C-E-G): strictly 1-3-5 (never 1-2-5).
     - LH Root Triad (e.g. C-E-G): strictly 5-3-1 (never 5-4-1).
     - RH 1st Inversion (e.g. E-G-C): 1-2-5.
     - LH 2nd Inversion (e.g. G-C-E): 5-2-1.
     - 4-note and 5-note voicings strictly assign distinct fingers.
  3. Dynamic Programming for Melodies & Polyphony:
     - Evaluates transition costs (hand shift distance, thumb-under, finger-over,
       repeated finger penalty, black-key thumb penalty).
     - Minimizes global biomechanical tension across the musical phrase.
"""

from typing import List, Dict, Any, Tuple, Optional
import math

BLACK_KEYS = {1, 3, 6, 8, 10}  # C#, D#, F#, G#, A# (mod 12)

# Anatomical resting spans (in semitones) between adjacent fingers [f_i, f_{i+1}]
# e.g., (1, 2): thumb-index resting span is ~3-4 semitones; max comfort is ~6-7.
RESTING_SPANS_RH = {
    (1, 2): 3.5,
    (2, 3): 2.0,
    (3, 4): 1.5,
    (4, 5): 1.5,
    (1, 3): 5.5,
    (2, 4): 3.5,
    (3, 5): 3.0,
    (1, 4): 7.0,
    (2, 5): 5.0,
    (1, 5): 8.5,
}


def _is_black_key(midi_pitch: int) -> bool:
    return (midi_pitch % 12) in BLACK_KEYS


def _solve_chord_fingering_rh(pitches: List[int]) -> List[int]:
    """
    Computes ergonomic, textbook fingering for Right Hand chords.
    Pitches are sorted ascending: p_0 < p_1 < ... < p_{k-1}.
    Fingers must be strictly increasing: 1 <= f_0 < f_1 < ... < f_{k-1} <= 5.
    """
    k = len(pitches)
    if k == 0:
        return []
    if k == 1:
        # Single note: thumb/pinky preference depends on register, default 1 or 2
        p = pitches[0]
        if _is_black_key(p):
            return [2 if p < 72 else 3]
        return [1 if p < 72 else 5]

    span = pitches[-1] - pitches[0]

    if k == 2:
        # Two-note dyad
        p0, p1 = pitches[0], pitches[1]
        diff = p1 - p0
        if diff <= 2:
            return [1, 2] if not _is_black_key(p0) else [2, 3]
        elif diff <= 5:  # Third or fourth
            if _is_black_key(p0) and not _is_black_key(p1):
                return [2, 4]
            return [1, 3] if diff <= 4 else [1, 4]
        elif diff <= 7:  # Fifth
            return [1, 4] if diff == 6 else [1, 5]
        else:  # Sixth, octave, or wider
            return [1, 5]

    if k == 3:
        # Triad voicing: analyze interval structure
        i1 = pitches[1] - pitches[0]  # lower interval
        i2 = pitches[2] - pitches[1]  # upper interval

        if span <= 12:
            # Standard close-position triads
            if i1 in (3, 4) and i2 in (3, 4):
                # Root position triad (e.g. C-E-G, 4+3 or 3+4): strictly [1, 3, 5]
                return [1, 3, 5]
            elif i1 in (3, 4) and i2 >= 5:
                # First inversion (e.g. E-G-C, 3+5): textbook [1, 2, 5]
                return [1, 2, 5]
            elif i1 >= 5 and i2 in (3, 4):
                # Second inversion (e.g. G-C-E, 5+4): textbook [1, 3, 5] or [1, 2, 4]
                return [1, 3, 5]
            elif i1 <= 2:
                # Cluster/suspended second (e.g. C-D-G): [1, 2, 5]
                return [1, 2, 5]
            else:
                return [1, 3, 5]
        else:
            # Wide spread triad (octave + 10th)
            return [1, 2, 5]

    if k == 4:
        # 4-note chord (e.g. 7th chord or octave with inner triad)
        i_top = pitches[3] - pitches[2]
        if span >= 10:
            # Octave with two inner notes
            if i_top <= 3:
                return [1, 2, 4, 5]
            else:
                return [1, 2, 3, 5]
        else:
            return [1, 2, 3, 5]

    # 5 or more notes
    fingers = [1, 2, 3, 4, 5]
    if k > 5:
        fingers.extend([5] * (k - 5))
    return fingers[:k]


def _solve_chord_fingering_lh(pitches: List[int]) -> List[int]:
    """
    Computes ergonomic, textbook fingering for Left Hand chords.
    Pitches are sorted ascending: p_0 < p_1 < ... < p_{k-1}.
    Fingers must be strictly decreasing: 5 >= f_0 > f_1 > ... > f_{k-1} >= 1.
    """
    k = len(pitches)
    if k == 0:
        return []
    if k == 1:
        p = pitches[0]
        if _is_black_key(p):
            return [2 if p > 48 else 3]
        return [5 if p < 48 else 1]

    span = pitches[-1] - pitches[0]

    if k == 2:
        p0, p1 = pitches[0], pitches[1]
        diff = p1 - p0
        if diff <= 2:
            return [3, 2] if not _is_black_key(p0) else [4, 3]
        elif diff <= 5:  # Third or fourth
            if _is_black_key(p1) and not _is_black_key(p0):
                return [4, 2]
            return [5, 3] if diff <= 4 else [5, 2]
        elif diff <= 7:  # Fifth
            return [5, 2] if diff == 6 else [5, 1]
        else:  # Sixth, octave, or wider
            return [5, 1]

    if k == 3:
        i1 = pitches[1] - pitches[0]  # lower interval
        i2 = pitches[2] - pitches[1]  # upper interval

        if span <= 12:
            # Standard close-position triads
            if i1 in (3, 4) and i2 in (3, 4):
                # Root position triad (e.g. C-E-G, 4+3 or 3+4): strictly [5, 3, 1] (NEVER 5-4-1)
                return [5, 3, 1]
            elif i1 in (3, 4) and i2 >= 5:
                # First inversion (e.g. E-G-C, 3+5): textbook [5, 3, 1]
                return [5, 3, 1]
            elif i1 >= 5 and i2 in (3, 4):
                # Second inversion (e.g. G-C-E, 5+4): textbook [5, 2, 1] (lower fourth = 5 to 2)
                return [5, 2, 1]
            elif i1 <= 2:
                return [5, 4, 1]
            else:
                return [5, 3, 1]
        else:
            return [5, 2, 1]

    if k == 4:
        i_low = pitches[1] - pitches[0]
        if span >= 10:
            if i_low <= 3:
                return [5, 4, 2, 1]
            else:
                return [5, 3, 2, 1]
        else:
            return [5, 4, 2, 1]

    fingers = [5, 4, 3, 2, 1]
    if k > 5:
        fingers.extend([1] * (k - 5))
    return fingers[:k]


def _transition_cost(
    hand: str,
    prev_pitch: int,
    prev_finger: int,
    curr_pitch: int,
    curr_finger: int,
    dt: float
) -> float:
    """
    Computes the biomechanical transition cost from (prev_pitch, prev_finger)
    to (curr_pitch, curr_finger).
    """
    cost = 0.0
    dp = curr_pitch - prev_pitch
    df = curr_finger - prev_finger

    # 1. Repeated finger on different pitch
    if curr_finger == prev_finger and dp != 0:
        cost += 12.0 / max(0.1, dt)

    # 2. Black key thumb/pinky penalty
    if curr_finger in (1, 5) and _is_black_key(curr_pitch):
        cost += 2.5

    # 3. Ergonomic direction and crossing rules
    if hand == 'right':
        if dp > 0:  # Ascending melody
            if df > 0:
                # Normal ascending order (e.g. 1 -> 2 -> 3)
                expected_dp = RESTING_SPANS_RH.get((prev_finger, curr_finger), (curr_finger - prev_finger) * 2.0)
                cost += abs(dp - expected_dp) * 0.8
            elif df < 0:
                # Crossing thumb under (e.g. 3 -> 1 or 4 -> 1)
                if curr_finger == 1 and prev_finger in (3, 4):
                    cost += 1.5 if dp <= 4 else 3.5  # standard smooth thumb-under
                else:
                    cost += 10.0 + abs(df) * 4.0  # awkward crossing
        elif dp < 0:  # Descending melody
            if df < 0:
                # Normal descending order (e.g. 5 -> 4 -> 3)
                expected_dp = RESTING_SPANS_RH.get((curr_finger, prev_finger), (prev_finger - curr_finger) * 2.0)
                cost += abs(abs(dp) - expected_dp) * 0.8
            elif df > 0:
                # Crossing finger over thumb (e.g. 1 -> 3 or 1 -> 4)
                if prev_finger == 1 and curr_finger in (3, 4):
                    cost += 1.5 if abs(dp) <= 4 else 3.5  # standard smooth crossing
                else:
                    cost += 10.0 + abs(df) * 4.0  # awkward crossing
    else:  # Left Hand (mirrors RH)
        if dp > 0:  # Ascending bass
            if df < 0:
                # Normal ascending in LH: 5 -> 4 -> 3 -> 2 -> 1
                cost += max(0.0, dp - abs(df) * 3.0) * 0.8
            elif df > 0:
                # Crossing finger over thumb in LH (e.g. 1 -> 3 or 1 -> 4)
                if prev_finger == 1 and curr_finger in (3, 4):
                    cost += 1.5 if dp <= 4 else 3.5
                else:
                    cost += 10.0 + abs(df) * 4.0
        elif dp < 0:  # Descending bass
            if df > 0:
                # Normal descending in LH: 1 -> 2 -> 3 -> 4 -> 5
                cost += max(0.0, abs(dp) - abs(df) * 3.0) * 0.8
            elif df < 0:
                # Thumb under in LH (e.g. 3 -> 1 or 4 -> 1)
                if curr_finger == 1 and prev_finger in (3, 4):
                    cost += 1.5 if abs(dp) <= 4 else 3.5
                else:
                    cost += 10.0 + abs(df) * 4.0

    return cost


def _solve_melody_viterbi(hand: str, notes: List[Dict[str, Any]]) -> None:
    """
    Uses Viterbi Dynamic Programming to assign optimal sequence of fingers
    for a single-voice melody line.
    """
    if not notes:
        return

    n_events = len(notes)
    candidate_fingers = [1, 2, 3, 4, 5]

    # Viterbi tables
    # viterbi_cost[t][f_idx] = minimal accumulated cost up to event t with finger f
    # backpointer[t][f_idx] = f_idx of previous event that achieved this minimum
    viterbi_cost = [[float('inf')] * 5 for _ in range(n_events)]
    backpointer = [[0] * 5 for _ in range(n_events)]

    # Initial step t=0
    first_p = notes[0]['midi']
    for fi, f in enumerate(candidate_fingers):
        init_cost = 0.0
        if _is_black_key(first_p) and f in (1, 5):
            init_cost += 3.0
        # Default register heuristic
        if hand == 'right':
            init_cost += abs(f - (1 if first_p < 65 else (3 if first_p < 76 else 5))) * 0.5
        else:
            init_cost += abs(f - (5 if first_p < 48 else (3 if first_p < 55 else 1))) * 0.5
        viterbi_cost[0][fi] = init_cost

    # DP forward pass
    for t in range(1, n_events):
        prev_p = notes[t - 1]['midi']
        curr_p = notes[t]['midi']
        dt = max(0.02, notes[t]['time'] - notes[t - 1]['time'])

        for curr_fi, curr_f in enumerate(candidate_fingers):
            best_prev_cost = float('inf')
            best_prev_fi = 0

            for prev_fi, prev_f in enumerate(candidate_fingers):
                t_cost = _transition_cost(hand, prev_p, prev_f, curr_p, curr_f, dt)
                total_c = viterbi_cost[t - 1][prev_fi] + t_cost
                if total_c < best_prev_cost:
                    best_prev_cost = total_c
                    best_prev_fi = prev_fi

            viterbi_cost[t][curr_fi] = best_prev_cost
            backpointer[t][curr_fi] = best_prev_fi

    # Backtracking optimal path
    best_last_fi = min(range(5), key=lambda fi: viterbi_cost[-1][fi])
    curr_fi = best_last_fi

    for t in reversed(range(n_events)):
        notes[t]['finger'] = candidate_fingers[curr_fi]
        curr_fi = backpointer[t][curr_fi]


def assign_fingerings(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assigns realistic, ergonomic piano fingerings (1 to 5) to each note based on:
    - Hand assignment ('left' vs 'right')
    - Polyphonic chord voicing (textbook triads: RH 1-3-5, LH 5-3-1, 1st/2nd inversion rules)
    - Dynamic Programming Viterbi solver for melodic voice leading & crossing.
    """
    if not notes:
        return notes

    # Sort notes chronologically, then by pitch
    sorted_notes = sorted(notes, key=lambda n: (n.get('time', 0.0), n.get('midi', 60)))

    # Cluster notes into temporal chord slices (<= 35ms onset difference)
    clusters: List[List[Dict[str, Any]]] = []
    curr_cluster: List[Dict[str, Any]] = [sorted_notes[0]]

    for n in sorted_notes[1:]:
        if abs(n.get('time', 0.0) - curr_cluster[0].get('time', 0.0)) <= 0.035:
            curr_cluster.append(n)
        else:
            clusters.append(curr_cluster)
            curr_cluster = [n]
    clusters.append(curr_cluster)

    # Separate single-note melodic sequences vs multi-note chords for each hand
    lh_melodic_run: List[Dict[str, Any]] = []
    rh_melodic_run: List[Dict[str, Any]] = []

    for cluster in clusters:
        lh_notes = sorted([n for n in cluster if n.get('hand') == 'left'], key=lambda x: x.get('midi', 0))
        rh_notes = sorted([n for n in cluster if n.get('hand') == 'right'], key=lambda x: x.get('midi', 0))

        # --- LEFT HAND ---
        if len(lh_notes) == 1:
            lh_melodic_run.append(lh_notes[0])
        elif len(lh_notes) > 1:
            # Flush previous melodic run through Viterbi
            if lh_melodic_run:
                _solve_melody_viterbi('left', lh_melodic_run)
                lh_melodic_run = []
            # Solve chord voicing
            pitches = [n['midi'] for n in lh_notes]
            fingers = _solve_chord_fingering_lh(pitches)
            for idx, n in enumerate(lh_notes):
                n['finger'] = fingers[idx]

        # --- RIGHT HAND ---
        if len(rh_notes) == 1:
            rh_melodic_run.append(rh_notes[0])
        elif len(rh_notes) > 1:
            # Flush previous melodic run through Viterbi
            if rh_melodic_run:
                _solve_melody_viterbi('right', rh_melodic_run)
                rh_melodic_run = []
            # Solve chord voicing
            pitches = [n['midi'] for n in rh_notes]
            fingers = _solve_chord_fingering_rh(pitches)
            for idx, n in enumerate(rh_notes):
                n['finger'] = fingers[idx]

    # Flush any remaining melodic runs at end of piece
    if lh_melodic_run:
        _solve_melody_viterbi('left', lh_melodic_run)
    if rh_melodic_run:
        _solve_melody_viterbi('right', rh_melodic_run)

    return notes
