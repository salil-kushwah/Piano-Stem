"""
Harmonic Chord Recognition & Lead Sheet Engine
Identifies musical chord symbols (e.g. Em, G, D/F#, Cmaj7, Am7, Bbdim)
from polyphonic MIDI note clusters and rolling harmonic windows.
Features key-aware enharmonic spelling (flats vs sharps) and slash-chord bass detection.
"""
from typing import List, Dict, Any, Optional, Set, Tuple

# Diatonic & Enharmonic Pitch Spellings
SHARP_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
FLAT_NAMES  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

FLAT_KEYS = {
    'F Major', 'D Minor', 'Bb Major', 'G Minor', 'Eb Major', 'C Minor',
    'Ab Major', 'F Minor', 'Db Major', 'Bb Minor', 'Gb Major', 'Eb Minor'
}

# Standard Chord Templates: semitone intervals relative to root (0)
# Prioritized from complex extensions to basic triads
CHORD_TEMPLATES = [
    # Extended 9ths & Adds
    ("maj9",   {0, 4, 7, 11, 2}, 5),
    ("m9",     {0, 3, 7, 10, 2}, 5),
    ("9",      {0, 4, 7, 10, 2}, 5),
    ("add9",   {0, 4, 7, 2},     4),
    ("m(add9)",{0, 3, 7, 2},     4),
    # 7ths
    ("maj7",   {0, 4, 7, 11},    4),
    ("m7",     {0, 3, 7, 10},    4),
    ("7",      {0, 4, 7, 10},    4),
    ("m7b5",   {0, 3, 6, 10},    4),
    ("dim7",   {0, 3, 6, 9},     4),
    ("m(maj7)",{0, 3, 7, 11},    4),
    ("7sus4",  {0, 5, 7, 10},    4),
    # 6ths
    ("6",      {0, 4, 7, 9},     4),
    ("m6",     {0, 3, 7, 9},     4),
    # Suspended
    ("sus4",   {0, 5, 7},        3),
    ("sus2",   {0, 2, 7},        3),
    # Triads
    ("",       {0, 4, 7},        3),  # Major
    ("m",      {0, 3, 7},        3),  # Minor
    ("dim",    {0, 3, 6},        3),
    ("aug",    {0, 4, 8},        3),
    ("5",      {0, 7},           2),  # Power chord / open fifth
]


def get_pitch_names(key_name: Optional[str] = None) -> List[str]:
    """Returns flat or sharp pitch names based on key signature."""
    if key_name and key_name in FLAT_KEYS:
        return FLAT_NAMES
    return SHARP_NAMES


def identify_chord_from_pitches(
    pitches: List[int],
    key_name: Optional[str] = None,
    bass_pitch: Optional[int] = None,
    min_confidence: float = 0.50
) -> Optional[Dict[str, Any]]:
    """
    Identifies root, quality, and slash bass for a set of MIDI pitches.
    Returns:
        {
            'symbol': 'Em7/B',
            'root': 'E',
            'quality': 'm7',
            'bass': 'B',
            'is_inversion': True,
            'confidence': 0.92
        }
    """
    if not pitches or len(pitches) < 2:
        return None

    pitch_names = get_pitch_names(key_name)
    pcs = set(p % 12 for p in pitches)

    if bass_pitch is None:
        bass_pitch = min(pitches)
    bass_pc = bass_pitch % 12
    bass_name = pitch_names[bass_pc]

    best_match: Optional[Dict[str, Any]] = None
    best_score = -999.0

    # Test all 12 pitch classes as candidate roots
    for root in range(12):
        root_name = pitch_names[root]

        for quality, intervals, min_notes in CHORD_TEMPLATES:
            # Shift intervals to candidate root
            template_pcs = set((root + i) % 12 for i in intervals)

            # Check overlap
            common = pcs.intersection(template_pcs)
            missing = template_pcs - pcs
            rogue = pcs - template_pcs

            # Enforce minimum required notes for chord template
            if len(common) < min_notes:
                continue

            # Must contain the candidate root to be valid
            has_root = root in pcs
            if not has_root:
                continue

            has_third = ((root + 3) % 12 in pcs) or ((root + 4) % 12 in pcs)

            # Scoring:
            # Match count + bonus for root present + bonus for bass matching root - penalty for rogue notes
            score = len(common) * 2.5
            if has_root:
                score += 2.0
            if bass_pc == root:
                score += 2.5  # Root position bonus
            elif bass_pc in template_pcs:
                score += 1.0  # Recognized inversion bonus
            else:
                score -= 1.5  # Unrelated bass penalty

            # Penalty for missing core chord tones
            score -= len(missing) * 1.2
            # Penalty for unrelated rogue notes in cluster
            score -= len(rogue) * 1.8

            if score > best_score:
                best_score = score
                is_inversion = (bass_pc != root) and (bass_pc in template_pcs)
                symbol = f"{root_name}{quality}"
                if is_inversion:
                    symbol = f"{symbol}/{bass_name}"
                elif bass_pc != root and len(pcs) >= 3:
                    # Slash bass over foreign root
                    symbol = f"{symbol}/{bass_name}"

                confidence = max(0.2, min(1.0, (score + 5.0) / 15.0))
                best_match = {
                    'symbol': symbol,
                    'root': root_name,
                    'quality': quality,
                    'bass': bass_name,
                    'is_inversion': is_inversion,
                    'confidence': round(confidence, 2)
                }

    if best_match and best_match['confidence'] >= min_confidence:
        return best_match
    return None


def detect_chord_at_time(
    notes: List[Dict[str, Any]],
    current_time: float,
    window_back: float = 0.50,
    window_forward: float = 0.06,
    key_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Rolling window harmonic detector for real-time visualizer HUD.
    Captures both simultaneous and rolling/arpeggiated notes active at current_time.
    """
    if not notes:
        return None

    # Collect notes sounding now or struck within the rolling decay window
    active_pitches = []
    bass_candidates = []

    t_start_min = current_time - window_back
    t_end_max = current_time + window_forward

    for n in notes:
        n_start = n.get("time", 0.0)
        if n_start > t_end_max:
            break
        n_dur = n.get("duration", 0.5)
        n_end = n_start + n_dur
        midi = n.get("midi", 60)

        # Check if note is physically sounding or was struck within the rolling window
        is_sounding = (n_start <= current_time <= n_end)
        is_in_recent_window = (t_start_min <= n_start <= t_end_max)

        if is_sounding or is_in_recent_window:
            active_pitches.append(midi)
            # Bass candidates: prioritize notes currently sounding or struck recently
            bass_candidates.append((midi, n_start))

    if len(active_pitches) < 2:
        return None

    # Find the true bass note: lowest pitch among notes sounding or most recently struck in bass
    lowest_pitch = min(active_pitches)

    return identify_chord_from_pitches(
        active_pitches,
        key_name=key_name,
        bass_pitch=lowest_pitch
    )


def annotate_notes_with_chords(
    notes: List[Dict[str, Any]],
    key_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Clusters simultaneous notes (within 45ms) and tags them with detected chord symbols.
    Also tags rolling arpeggios.
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: n.get("time", 0.0))
    n_count = len(sorted_notes)

    # 1. Cluster notes struck within 45ms
    clusters: List[List[Dict[str, Any]]] = []
    cur_cluster = [sorted_notes[0]]

    for i in range(1, n_count):
        n = sorted_notes[i]
        if (n.get("time", 0.0) - cur_cluster[0].get("time", 0.0)) <= 0.045:
            cur_cluster.append(n)
        else:
            clusters.append(cur_cluster)
            cur_cluster = [n]
    clusters.append(cur_cluster)

    # 2. Identify chord for each cluster
    for cluster in clusters:
        if len(cluster) >= 2:
            pitches = [n.get("midi", 60) for n in cluster]
            chord_info = identify_chord_from_pitches(pitches, key_name=key_name)
            if chord_info:
                sym = chord_info["symbol"]
                for n in cluster:
                    n["chord_name"] = sym
                    n["is_chord"] = True
            else:
                for n in cluster:
                    n["is_chord"] = True
        else:
            cluster[0].setdefault("is_chord", False)

    return sorted_notes
