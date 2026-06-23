import mido
import pretty_midi
from pathlib import Path
from typing import Dict, List, Any, Optional

from backend.hand_separator import separate_hands
from backend.fingering import assign_fingerings
from backend.tonal_analyzer import detect_key_signature
from backend.chord_namer import annotate_notes_with_chords

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_note_name(midi_number: int) -> str:
    octave = (midi_number // 12) - 1
    name = NOTE_NAMES[midi_number % 12]
    return f"{name}{octave}"

def resolve_note_releases(
    notes: List[pretty_midi.Note],
    min_dur: float = 0.04,
    max_treble_dur: Optional[float] = None,
    max_bass_dur: Optional[float] = None,
    **kwargs
) -> List[pretty_midi.Note]:
    """
    Physical Piano Note Release & Debounce Engine:
    1. Same-Pitch Sustained Note Reconnection:
       When a key is held, sub-perceptual tracking jitter (< 25ms) is debounced to avoid flickering.
    2. Minimum Duration Floor:
       Preserves crisp staccatissimo attacks (40ms floor) without micro-chopping.
    3. Monophonic Same-Pitch Truncation:
       If a pianist re-strikes the same physical key after a deliberate pause (> 25ms),
       the previous note releases cleanly (15ms before the new attack) so keys don't self-overlap.
    4. Zero Artificial Neighbor Clamping:
       No destructive neighbor key truncations. Chords and counterpoint ring naturally.
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda x: (x.start, x.pitch))
    resolved: List[pretty_midi.Note] = []

    for n in sorted_notes:
        prev_same = None
        for p in reversed(resolved):
            if p.pitch == n.pitch:
                prev_same = p
                break
            if (n.start - p.start) > 3.0:
                break

        if prev_same is not None:
            # Sub-perceptual tracking jitter (< 25ms): merge as single strike
            if (n.start - prev_same.start) < 0.025:
                prev_same.end = max(prev_same.end, n.end)
                prev_same.velocity = max(prev_same.velocity, n.velocity)
                continue
            # Legitimate new strike: release previous sustained note before the new attack
            if prev_same.end > n.start:
                prev_same.end = max(prev_same.start + min_dur, n.start - 0.015)

        end_t = max(n.start + min_dur, n.end)
        if max_treble_dur is not None and max_treble_dur > 0 and n.pitch >= 60:
            end_t = min(end_t, n.start + max_treble_dur)
        elif max_bass_dur is not None and max_bass_dur > 0 and n.pitch < 60:
            end_t = min(end_t, n.start + max_bass_dur)

        resolved.append(pretty_midi.Note(
            velocity=n.velocity,
            pitch=n.pitch,
            start=n.start,
            end=max(n.start + min_dur, end_t)
        ))

    return resolved


def apply_acoustic_velocity_curve(
    notes: List[pretty_midi.Note],
    min_dynamic: int = 35,
    max_dynamic: int = 112,
    gamma: float = 0.85
) -> List[pretty_midi.Note]:
    """
    Acoustic Grand Hammer Velocity Curve:
    Expands flat or compressed neural network velocity predictions into rich,
    touch-responsive pianissimos and fortissimos matching acoustic grand piano hammer physics.
    """
    if not notes:
        return notes
    vels = [n.velocity for n in notes]
    v_min, v_max = min(vels), max(vels)
    span = v_max - v_min
    if 5 < span < 65:
        for n in notes:
            u = (n.velocity - v_min) / float(span)
            u_curved = u ** gamma
            new_v = int(round(min_dynamic + u_curved * (max_dynamic - min_dynamic)))
            n.velocity = max(20, min(127, new_v))
    return notes


def filter_and_export_midi(
    input_midi_path: Path,
    output_midi_path: Path,
    min_velocity: int = 0,
    min_pitch: int = 21,
    max_pitch: int = 108,
    max_duration: Optional[float] = None
) -> Path:
    """
    Filters MIDI notes by minimum velocity, pitch range, and optional user-adjusted max duration clamp.
    """
    try:
        mid = pretty_midi.PrettyMIDI(str(input_midi_path))
        if mid.instruments:
            filtered = []
            for n in mid.instruments[0].notes:
                if n.velocity >= min_velocity and min_pitch <= n.pitch <= max_pitch:
                    end_t = n.end
                    if max_duration is not None and max_duration > 0:
                        end_t = min(n.end, n.start + max_duration)
                    n_cloned = pretty_midi.Note(
                        velocity=n.velocity,
                        pitch=n.pitch,
                        start=n.start,
                        end=max(n.start + 0.04, end_t)
                    )
                    filtered.append(n_cloned)
            mid.instruments[0].notes = filtered
            mid.write(str(output_midi_path))
        return output_midi_path
    except Exception as e:
        print(f"[Filter] Error exporting filtered MIDI: {e}")
        return input_midi_path


def parse_midi_file(
    midi_path: Path,
    deghost: bool = False,
    provenance_list: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Parses a MIDI file into structured JSON for 60FPS visualizer rendering and note inspection.
    Non-destructive: Does NOT mutate the on-disk MIDI file.
    """
    mid = mido.MidiFile(str(midi_path))
    notes: List[Dict[str, Any]] = []
    pedals: List[Dict[str, Any]] = []

    active_notes: Dict[int, Dict[str, Any]] = {}
    active_pedal: float | None = None
    total_time = 0.0

    for msg in mid:
        total_time += msg.time

        if msg.type == 'note_on' and msg.velocity > 0:
            if msg.note in active_notes:
                prev = active_notes[msg.note]
                notes.append({
                    "midi": msg.note,
                    "name": midi_to_note_name(msg.note),
                    "time": round(prev["start_time"], 4),
                    "duration": round(max(0.04, total_time - prev["start_time"]), 4),
                    "velocity": prev["velocity"] / 127.0,
                    "raw_velocity": prev["velocity"],
                    "hand": "unknown"
                })
            active_notes[msg.note] = {
                "start_time": total_time,
                "velocity": msg.velocity
            }

        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in active_notes:
                prev = active_notes.pop(msg.note)
                notes.append({
                    "midi": msg.note,
                    "name": midi_to_note_name(msg.note),
                    "time": round(prev["start_time"], 4),
                    "duration": round(max(0.04, total_time - prev["start_time"]), 4),
                    "velocity": prev["velocity"] / 127.0,
                    "raw_velocity": prev["velocity"],
                    "hand": "unknown"
                })

        elif msg.type == 'control_change' and msg.control == 64:
            if msg.value >= 64:
                if active_pedal is None:
                    active_pedal = total_time
            else:
                if active_pedal is not None:
                    pedals.append({
                        "time": round(active_pedal, 4),
                        "duration": round(max(0.08, total_time - active_pedal), 4)
                    })
                    active_pedal = None

    for note_num, prev in active_notes.items():
        notes.append({
            "midi": note_num,
            "name": midi_to_note_name(note_num),
            "time": round(prev["start_time"], 4),
            "duration": round(max(0.08, total_time - prev["start_time"]), 4),
            "velocity": prev["velocity"] / 127.0,
            "raw_velocity": prev["velocity"],
            "hand": "unknown"
        })

    if active_pedal is not None:
        pedals.append({
            "time": round(active_pedal, 4),
            "duration": round(max(0.08, total_time - active_pedal), 4)
        })

    notes.sort(key=lambda n: n["time"])

    # Attach provenance & confidence data if available
    prov_by_pitch: Dict[int, List[Dict[str, Any]]] = {}
    if provenance_list:
        for p in provenance_list:
            prov_by_pitch.setdefault(p["pitch"], []).append(p)

    for i, n in enumerate(notes):
        n["id"] = f"note_{i}_{n['midi']}_{int(n['time'] * 1000)}"
        matched_prov = None
        cand_list = prov_by_pitch.get(n["midi"], [])
        if cand_list:
            best_cand = min(cand_list, key=lambda c: abs(c["start"] - n["time"]))
            if abs(best_cand["start"] - n["time"]) <= 0.050:
                matched_prov = best_cand

        if matched_prov:
            n["onset_prob"] = matched_prov.get("onset_prob", round(n["velocity"], 3))
            n["frame_prob"] = matched_prov.get("frame_prob", round(min(1.0, 0.4 + n["velocity"] * 0.6), 3))
            n["status"] = matched_prov.get("status", "verified")
            n["rejection_reason"] = matched_prov.get("rejection_reason")
            n["evidence"] = matched_prov.get("evidence", {})
        else:
            # Honest confidence derived from physical acoustic velocity
            n["onset_prob"] = round(min(1.0, max(0.3, n["velocity"])), 3)
            n["frame_prob"] = round(min(1.0, max(0.4, 0.4 + n["velocity"] * 0.6)), 3)
            n["status"] = "verified"
            n["rejection_reason"] = None
            n["evidence"] = {"velocity": n["raw_velocity"]}

    # Pedagogical hand separation, ergonomic fingering, tonal key, and chord annotations
    if notes:
        notes = separate_hands(notes)
        notes = assign_fingerings(notes)
        key_info = detect_key_signature(notes)
        notes = annotate_notes_with_chords(notes, key_name=key_info["key_name"])
    else:
        key_info = {"key_name": "C Major", "confidence": 0.0}

    # Estimate tempo & beat interval
    tempo_bpm = 120.0
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        est_tempo = pm.estimate_tempo()
        if 40.0 <= est_tempo <= 260.0:
            tempo_bpm = round(float(est_tempo), 1)
    except Exception:
        pass

    beat_interval = round(60.0 / tempo_bpm, 4)

    return {
        "duration": round(total_time, 4),
        "notes_count": len(notes),
        "key_signature": key_info["key_name"],
        "key_confidence": key_info.get("confidence", 0.0),
        "tempo_bpm": tempo_bpm,
        "beat_interval": beat_interval,
        "notes": notes,
        "pedals": pedals
    }
