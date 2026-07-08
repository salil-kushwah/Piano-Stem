"""
Benchmark Evaluator — Objective mir_eval Ground Truth Evaluation Suite

Standardized MIR Evaluation & Comprehensive False-Note Diagnostics:
  - Onset tolerance: 50ms (0.050s)
  - Pitch tolerance: 50 cents (exact semitone matching in true Hertz)
  - Dual Reporting:
      1. Unaligned (zero clock shift — honest production synchronization check)
      2. Optimal Lag Aligned (for isolating lead-in audio latency)
  - False Notes Per Minute metric
  - Error breakdowns:
      1. By pitch register: Bass (< MIDI 40), Mid (MIDI 40-70), Treble (> MIDI 70)
      2. By strike dynamic: Pianissimo (< 40), Mezzo (40-80), Forte (> 80)
      3. By harmonic relationship: Octave (+12/-12), 12th/17th (+19/+24/+28), Semitone smear, Other
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import collections

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pretty_midi
import mir_eval


def midi_to_hz(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((float(midi_note) - 69.0) / 12.0))


def load_notes_as_intervals(midi_path: Path, dedup_simultaneous: bool = True) -> Tuple[np.ndarray, np.ndarray, List[pretty_midi.Note]]:
    """
    Loads MIDI notes into (intervals, pitches_hz, raw_notes) arrays for mir_eval.
    intervals: shape (N, 2) in seconds
    pitches_hz: shape (N,) frequencies in Hz
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)

    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    if not notes:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64), []

    if dedup_simultaneous:
        dedup = []
        for n in notes:
            if any(abs(n.start - x.start) < 0.035 and n.pitch == x.pitch for x in dedup):
                continue
            dedup.append(n)
        notes = dedup

    intervals = np.array([[n.start, max(n.start + 0.04, n.end)] for n in notes], dtype=np.float64)
    pitches_hz = np.array([midi_to_hz(n.pitch) for n in notes], dtype=np.float64)
    return intervals, pitches_hz, notes


def find_optimal_time_lag(
    ref_intervals: np.ndarray,
    ref_pitches: np.ndarray,
    est_intervals: np.ndarray,
    est_pitches: np.ndarray,
    max_lag: float = 3.5,
    lag_step: float = 0.01
) -> Tuple[float, float]:
    """Finds constant time lag between reference MIDI and audio recording using mir_eval F1 search."""
    if len(ref_intervals) == 0 or len(est_intervals) == 0:
        return 0.0, 0.0

    best_lag = 0.0
    best_f1 = -1.0

    shifts = np.arange(-max_lag, max_lag + lag_step, lag_step)
    for shift in shifts:
        valid_mask = (est_intervals[:, 0] + shift) >= 0
        if not np.any(valid_mask):
            continue
        shifted_int = est_intervals[valid_mask] + shift
        shifted_pitches = est_pitches[valid_mask]

        _, _, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            shifted_int, shifted_pitches,
            onset_tolerance=0.050,
            pitch_tolerance=50.0,
            offset_ratio=None
        )
        if f1 > best_f1:
            best_f1 = f1
            best_lag = float(shift)

    return best_lag, best_f1


def diagnose_unmatched_notes(
    ref_notes: List[pretty_midi.Note],
    est_notes: List[pretty_midi.Note],
    time_lag: float,
    onset_tol: float = 0.050
) -> Dict[str, Any]:
    """
    Detailed acoustic error diagnosis:
    Categorizes false notes by pitch interval, register, and dynamic level.
    """
    unmatched_est = []
    unmatched_ref = []

    for p in est_notes:
        t_est = p.start + time_lag
        matched = any(g.pitch == p.pitch and abs(g.start - t_est) <= onset_tol for g in ref_notes)
        if not matched:
            unmatched_est.append(p)

    for g in ref_notes:
        matched = any(p.pitch == g.pitch and abs((p.start + time_lag) - g.start) <= onset_tol for p in est_notes)
        if not matched:
            unmatched_ref.append(g)

    # Classify error types
    octave_ghosts = 0
    sub_octave_ghosts = 0
    harmonic_ghosts = 0
    semitone_smears = 0
    other_errors = 0

    bass_errors = 0
    mid_errors = 0
    treble_errors = 0

    pianissimo_errors = 0
    mezzo_errors = 0
    forte_errors = 0

    for p in unmatched_est:
        t_est = p.start + time_lag
        # Register
        if p.pitch < 40:
            bass_errors += 1
        elif p.pitch <= 70:
            mid_errors += 1
        else:
            treble_errors += 1

        # Dynamic
        if p.velocity < 40:
            pianissimo_errors += 1
        elif p.velocity <= 80:
            mezzo_errors += 1
        else:
            forte_errors += 1

        # Harmonic relationship against concurrent reference notes
        concurrent_ref = [g for g in ref_notes if abs(g.start - t_est) <= 0.15]
        found_rel = False
        for g in concurrent_ref:
            diff = p.pitch - g.pitch
            if diff in (12, 24):
                octave_ghosts += 1
                found_rel = True
                break
            elif diff in (-12, -24):
                sub_octave_ghosts += 1
                found_rel = True
                break
            elif diff in (19, 28):
                harmonic_ghosts += 1
                found_rel = True
                break
            elif abs(diff) == 1:
                semitone_smears += 1
                found_rel = True
                break
        if not found_rel:
            other_errors += 1

    return {
        "false_positive_count": len(unmatched_est),
        "false_negative_count": len(unmatched_ref),
        "by_register": {
            "bass_lt_40": bass_errors,
            "mid_40_70": mid_errors,
            "treble_gt_70": treble_errors
        },
        "by_dynamic": {
            "pianissimo_lt_40": pianissimo_errors,
            "mezzo_40_80": mezzo_errors,
            "forte_gt_80": forte_errors
        },
        "by_harmonic_type": {
            "octave_ghosts": octave_ghosts,
            "sub_octave_ghosts": sub_octave_ghosts,
            "harmonic_12th_17th": harmonic_ghosts,
            "semitone_smears": semitone_smears,
            "other_errors": other_errors
        }
    }


def evaluate_transcription(
    ref_midi_path: Path,
    est_midi_path: Path,
    onset_tolerance: float = 0.050,
    audio_duration_sec: Optional[float] = None,
    auto_align_lag: bool = True
) -> Dict[str, Any]:
    """
    Evaluates estimated transcription MIDI against ground truth reference MIDI using mir_eval.
    Reports both unaligned and aligned results, false notes per minute, and error breakdown.
    """
    ref_intervals, ref_pitches, ref_notes = load_notes_as_intervals(ref_midi_path, dedup_simultaneous=True)
    est_intervals, est_pitches, est_notes = load_notes_as_intervals(est_midi_path, dedup_simultaneous=False)

    if len(ref_intervals) == 0:
        return {"error": "Reference MIDI contains no notes."}
    if len(est_intervals) == 0:
        return {
            "ref_notes": len(ref_intervals),
            "est_notes": 0,
            "unaligned_f1": 0.0,
            "aligned_f1": 0.0,
            "time_lag": 0.0,
            "false_notes_per_min": 0.0
        }

    # 1. Evaluate Unaligned (Zero Time Lag Shift)
    p_unaligned, r_unaligned, f1_unaligned, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=50.0,
        offset_ratio=None
    )

    # 2. Evaluate with Optimal Lag Search
    time_lag = 0.0
    if auto_align_lag:
        time_lag, _ = find_optimal_time_lag(ref_intervals, ref_pitches, est_intervals, est_pitches)

    valid_mask = (est_intervals[:, 0] + time_lag) >= 0
    shifted_int = est_intervals[valid_mask] + time_lag
    shifted_pitches = est_pitches[valid_mask]

    p_on, r_on, f1_on, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches,
        shifted_int, shifted_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=50.0,
        offset_ratio=None
    )

    p_off, r_off, f1_off, avg_overlap = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches,
        shifted_int, shifted_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=50.0,
        offset_ratio=0.20,
        offset_min_tolerance=0.050
    )

    # Detailed Error Diagnosis
    diag = diagnose_unmatched_notes(ref_notes, est_notes, time_lag, onset_tol=onset_tolerance)

    # False Notes Per Minute
    dur_sec = audio_duration_sec or float(ref_intervals[-1, 1]) if len(ref_intervals) > 0 else 60.0
    dur_min = max(0.1, dur_sec / 60.0)
    false_per_min = diag["false_positive_count"] / dur_min

    return {
        "ref_notes": len(ref_intervals),
        "est_notes": len(est_intervals),
        "time_lag": round(time_lag, 3),
        "unaligned": {
            "onset_precision": float(p_unaligned),
            "onset_recall": float(r_unaligned),
            "onset_f1": float(f1_unaligned),
        },
        "aligned": {
            "onset_precision": float(p_on),
            "onset_recall": float(r_on),
            "onset_f1": float(f1_on),
            "offset_precision": float(p_off),
            "offset_recall": float(r_off),
            "offset_f1": float(f1_off),
        },
        "false_notes_per_min": round(false_per_min, 1),
        "diagnosis": diag
    }


def run_ground_truth_suite(workspace_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Runs evaluation on all available ground truth pairs with comprehensive error diagnosis."""
    from backend.pipeline import ProcessingPipeline

    test_pairs = [
        ("Another Love", workspace_dir / "another love.mp3", workspace_dir / "another love midi.mid", 21.4),
        ("Wicked Game", workspace_dir / "wicked game.mp3", workspace_dir / "wicked game midi.mid", 76.4),
        ("Idea 21", workspace_dir / "idea 21.mp3", workspace_dir / "idea 21 midi.mid", 58.6)
    ]

    results = {}
    print("=" * 80)
    print("OBJECTIVE GROUND TRUTH BENCHMARK SUITE (Dual mir_eval + False Note Diagnostics)")
    print("=" * 80)

    for name, audio_p, ref_mid_p, dur_s in test_pairs:
        if not audio_p.exists() or not ref_mid_p.exists():
            print(f"[Skip] {name}: Missing file.")
            continue

        print(f"\n--- Testing: {name} ---")
        task_id = f"eval_{name.lower().replace(' ', '_')}"
        pipeline = ProcessingPipeline(task_id=task_id)

        res = pipeline.run(
            input_audio_path=audio_p,
            is_mixed_audio=False,
            precision_mode="clean",
            model_engine="studio_master"
        )

        est_mid = Path(res["midi_file"])
        scores = evaluate_transcription(ref_mid_p, est_mid, audio_duration_sec=dur_s)
        results[name] = scores

        al = scores["aligned"]
        unal = scores["unaligned"]
        diag = scores["diagnosis"]

        print(f"  Reference Notes: {scores['ref_notes']} | Estimated Notes: {scores['est_notes']} (Lead Lag: {scores['time_lag']:+.3f}s)")
        print(f"  [Aligned Onset]   Precision: {al['onset_precision']:.1%} | Recall: {al['onset_recall']:.1%} | F1: {al['onset_f1']:.1%}")
        print(f"  [Unaligned Onset] Precision: {unal['onset_precision']:.1%} | Recall: {unal['onset_recall']:.1%} | F1: {unal['onset_f1']:.1%}")
        print(f"  [False Notes/Min] {scores['false_notes_per_min']}/min (FP: {diag['false_positive_count']}, FN: {diag['false_negative_count']})")
        print(f"  [Harmonic Breakdown] Octave Ghosts: {diag['by_harmonic_type']['octave_ghosts']} | Sub-Octave: {diag['by_harmonic_type']['sub_octave_ghosts']} | Semitone Smears: {diag['by_harmonic_type']['semitone_smears']} | Other: {diag['by_harmonic_type']['other_errors']}")

    print("\n" + "=" * 80)
    return results


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run_ground_truth_suite(root)
