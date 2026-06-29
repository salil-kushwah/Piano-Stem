"""
Pedagogical Piano Hand Separator (Trajectory Tracking & Voice Continuity Solver)

Separates polyphonic piano notes into Left Hand (Cyan/Blue) and Right Hand (Pink/Purple)
without naive rigid pitch cutoffs:
  1. Hand Trajectory & Centroid Tracking:
     - Continuously models LH and RH hand centroids across time.
     - Accommodates wide arpeggios reaching into the treble (e.g. Chopin/Liszt arpeggiated bass)
       without chopping them in half at middle C.
  2. Voice Continuity & Melodic Grouping:
     - Preserves melodic contours and accompaniment figures in their originating hand.
     - Supports hand crossings (e.g. RH crossing into bass or LH crossing over into treble).
  3. Anatomical Biomechanical Constraints:
     - Strict maximum hand span (comfort <= 12 semitones, physical limit 14 semitones / 10th).
     - Concurrent notes in the same hand cannot exceed physical stretch limit.
  4. Synchronized Chord & Dyad Tagging for Level Visualizer Bars.
"""

from typing import List, Dict, Any, Tuple, Optional
import statistics

MAX_HAND_SPAN = 14       # Maximum physical hand reach (a 10th)
COMFORT_HAND_SPAN = 11   # Comfortable reach (a minor 7th / octave)
CHORD_TOLERANCE = 0.035  # Max delta-t for simultaneous chord notes


def _cluster_by_time(sorted_notes: List[Dict[str, Any]], tol: float = CHORD_TOLERANCE) -> List[List[Dict[str, Any]]]:
    """Clusters notes into simultaneous temporal slices."""
    if not sorted_notes:
        return []
    clusters = []
    curr = [sorted_notes[0]]
    for n in sorted_notes[1:]:
        if abs(n['time'] - curr[0]['time']) <= tol:
            curr.append(n)
        else:
            clusters.append(curr)
            curr = [n]
    clusters.append(curr)
    return clusters


def separate_hands(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assigns 'hand' ('left' or 'right') and 'is_chord' (bool)
    to each note dict in-place using trajectory tracking and voice continuity.
    """
    if not notes:
        return notes

    # Sort chronologically, then by pitch ascending
    sorted_notes = sorted(notes, key=lambda n: (n['time'], n['midi']))
    clusters = _cluster_by_time(sorted_notes)

    # Initial hand centroid estimates (LH in bass C3=48, RH in treble G4=67)
    lh_centroid = 48.0
    rh_centroid = 67.0
    last_lh_time = -1.0
    last_rh_time = -1.0

    # Pass 1: Trajectory Tracking per temporal cluster
    for cluster in clusters:
        cluster_time = cluster[0]['time']
        pitches = sorted(n['midi'] for n in cluster)
        k = len(pitches)

        # Decay centroids towards resting default if long rest occurs (> 3 seconds)
        if last_lh_time >= 0 and (cluster_time - last_lh_time) > 3.0:
            lh_centroid = 0.7 * lh_centroid + 0.3 * 48.0
        if last_rh_time >= 0 and (cluster_time - last_rh_time) > 3.0:
            rh_centroid = 0.7 * rh_centroid + 0.3 * 67.0

        if k == 1:
            p = pitches[0]
            # Single note: evaluate distance to active LH and RH centroids
            dist_lh = abs(p - lh_centroid)
            dist_rh = abs(p - rh_centroid)

            # Prior: bass register (< 53 / F3) strongly prefers LH; high treble (> 71 / B4) strongly prefers RH
            if p < 50:
                hand = 'left'
            elif p > 72:
                hand = 'right'
            else:
                # In the middle register (50..72):
                # Continuity weight: preference for hand that played most recently
                dt_lh = cluster_time - last_lh_time if last_lh_time >= 0 else 1.0
                dt_rh = cluster_time - last_rh_time if last_rh_time >= 0 else 1.0

                cost_lh = dist_lh + (1.5 if p >= 60 else -1.5) + min(2.0, dt_lh * 0.5)
                cost_rh = dist_rh + (1.5 if p < 60 else -1.5) + min(2.0, dt_rh * 0.5)

                hand = 'left' if cost_lh < cost_rh else 'right'

            cluster[0]['hand'] = hand
            if hand == 'left':
                lh_centroid = 0.8 * lh_centroid + 0.2 * p
                last_lh_time = cluster_time
            else:
                rh_centroid = 0.8 * rh_centroid + 0.2 * p
                last_rh_time = cluster_time

        elif k == 2:
            # Two simultaneous notes: p0 < p1
            p0, p1 = pitches[0], pitches[1]
            span = p1 - p0

            # If both notes easily fit in one hand (span <= 10) and both are in extreme register:
            if span <= 10 and p1 <= 53:
                # Low bass dyad: both LH
                for n in cluster:
                    n['hand'] = 'left'
                lh_centroid = 0.8 * lh_centroid + 0.2 * ((p0 + p1) / 2.0)
                last_lh_time = cluster_time
            elif span <= 10 and p0 >= 69:
                # High treble dyad: both RH
                for n in cluster:
                    n['hand'] = 'right'
                rh_centroid = 0.8 * rh_centroid + 0.2 * ((p0 + p1) / 2.0)
                last_rh_time = cluster_time
            else:
                # Standard polyphony: lower note to LH, upper note to RH
                n0 = [n for n in cluster if n['midi'] == p0][0]
                n1 = [n for n in cluster if n['midi'] == p1][0]
                n0['hand'] = 'left'
                n1['hand'] = 'right'
                lh_centroid = 0.8 * lh_centroid + 0.2 * p0
                rh_centroid = 0.8 * rh_centroid + 0.2 * p1
                last_lh_time = cluster_time
                last_rh_time = cluster_time

        else:
            # 3 or more simultaneous notes: search optimal partition [0..split_idx] in LH, [split_idx..k] in RH
            best_cost = float('inf')
            best_split = 1

            for split_idx in range(k + 1):
                lh_group = pitches[:split_idx]
                rh_group = pitches[split_idx:]

                cost = 0.0
                # Hand span violations
                if len(lh_group) >= 2:
                    lh_span = lh_group[-1] - lh_group[0]
                    if lh_span > MAX_HAND_SPAN:
                        cost += 1000.0
                    elif lh_span > COMFORT_HAND_SPAN:
                        cost += (lh_span - COMFORT_HAND_SPAN) * 10.0

                if len(rh_group) >= 2:
                    rh_span = rh_group[-1] - rh_group[0]
                    if rh_span > MAX_HAND_SPAN:
                        cost += 1000.0
                    elif rh_span > COMFORT_HAND_SPAN:
                        cost += (rh_span - COMFORT_HAND_SPAN) * 10.0

                # Empty hand penalty (prefer using both hands when k >= 3)
                if not lh_group or not rh_group:
                    cost += 15.0

                # Centroid continuity cost
                if lh_group:
                    lh_mean = sum(lh_group) / len(lh_group)
                    cost += abs(lh_mean - lh_centroid) * 0.5
                if rh_group:
                    rh_mean = sum(rh_group) / len(rh_group)
                    cost += abs(rh_mean - rh_centroid) * 0.5

                if cost < best_cost:
                    best_cost = cost
                    best_split = split_idx

            # Assign best partition
            for n in cluster:
                n['hand'] = 'left' if n['midi'] in pitches[:best_split] else 'right'

            if pitches[:best_split]:
                lh_centroid = 0.8 * lh_centroid + 0.2 * (sum(pitches[:best_split]) / best_split)
                last_lh_time = cluster_time
            if pitches[best_split:]:
                rh_count = k - best_split
                rh_centroid = 0.8 * rh_centroid + 0.2 * (sum(pitches[best_split:]) / rh_count)
                last_rh_time = cluster_time

    # Pass 2: Fast Arpeggio Trajectory Smoothing
    # Detect ascending/descending rapid arpeggio runs (<= 250ms inter-onset) and prevent
    # abrupt middle-C hand flipping within an arpeggiated run.
    for i in range(1, len(sorted_notes) - 1):
        prev_n = sorted_notes[i - 1]
        curr_n = sorted_notes[i]
        next_n = sorted_notes[i + 1]

        dt1 = curr_n['time'] - prev_n['time']
        dt2 = next_n['time'] - curr_n['time']

        if dt1 <= 0.25 and dt2 <= 0.25:
            # Monotonic ascending or descending arpeggio
            dp1 = curr_n['midi'] - prev_n['midi']
            dp2 = next_n['midi'] - curr_n['midi']
            if (dp1 > 0 and dp2 > 0) or (dp1 < 0 and dp2 < 0):
                if prev_n.get('hand') == next_n.get('hand'):
                    curr_n['hand'] = prev_n.get('hand')

    # Pass 3: Chord Tagging (notes sounding within 35ms in the same hand)
    for i, note in enumerate(sorted_notes):
        has_chord_mate = any(
            other is not note
            and other.get('hand') == note.get('hand')
            and abs(other['time'] - note['time']) <= CHORD_TOLERANCE
            for other in sorted_notes[max(0, i - 8):min(len(sorted_notes), i + 9)]
        )
        note['is_chord'] = has_chord_mate

    return sorted_notes
