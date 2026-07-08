"""
Synthetic Mixed-Audio Benchmark & Stress Tester

Generates realistic mixed audio with exact ground-truth MIDI:
  1. Renders clean piano audio from reference MIDI via 17.1MB SoundFont.
  2. Blends with non-piano stems (vocals, drums, bass) at controlled Signal-to-Noise Ratios (SNR).
  3. Executes the full demixing + transcription pipeline end-to-end.
  4. Evaluates note precision, recall, false notes per minute, and pitch error breakdown
     (semitone, octave, other) using True Hz mir_eval metrics.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import soundfile as sf
import librosa
import pretty_midi
import mir_eval

from backend.synth import synthesize_midi_to_audio
from backend.benchmark_eval import load_notes_as_intervals


def create_synthetic_mix(
    piano_midi_path: Path,
    output_mix_path: Path,
    aux_stem_path: Optional[Path] = None,
    snr_db: float = 3.0,
    sr: int = 44100
) -> Tuple[Path, float]:
    """
    Renders reference piano MIDI and blends with auxiliary non-piano audio stem at specified SNR.
    Returns (output_mix_path, duration_seconds).
    """
    clean_piano_wav = output_mix_path.parent / f"{output_mix_path.stem}_clean_piano.wav"
    synthesize_midi_to_audio(piano_midi_path, clean_piano_wav, sample_rate=sr)

    piano_audio, _ = sf.read(str(clean_piano_wav))
    if piano_audio.ndim > 1:
        piano_audio = piano_audio.mean(axis=1)

    duration_s = len(piano_audio) / float(sr)

    if aux_stem_path and Path(aux_stem_path).exists():
        aux_audio, aux_sr = sf.read(str(aux_stem_path))
        if aux_audio.ndim > 1:
            aux_audio = aux_audio.mean(axis=1)
        if aux_sr != sr:
            aux_audio = librosa.resample(aux_audio, orig_sr=aux_sr, target_sr=sr)

        # Match lengths
        if len(aux_audio) < len(piano_audio):
            reps = int(np.ceil(len(piano_audio) / len(aux_audio)))
            aux_audio = np.tile(aux_audio, reps)[:len(piano_audio)]
        else:
            aux_audio = aux_audio[:len(piano_audio)]

        # Calculate target scale factor for desired SNR
        p_rms = np.sqrt(np.mean(piano_audio ** 2) + 1e-12)
        a_rms = np.sqrt(np.mean(aux_audio ** 2) + 1e-12)

        desired_aux_rms = p_rms / (10.0 ** (snr_db / 20.0))
        scale = desired_aux_rms / (a_rms + 1e-12)
        mix_audio = piano_audio + aux_audio * scale

        # Normalize to prevent digital clipping
        peak = np.max(np.abs(mix_audio))
        if peak > 0.95:
            mix_audio = (mix_audio / peak) * 0.95
    else:
        mix_audio = piano_audio

    sf.write(str(output_mix_path), mix_audio, sr, subtype="FLOAT")
    return output_mix_path, duration_s


def evaluate_mixed_pipeline(
    ref_midi_path: Path,
    est_midi_path: Path,
    duration_seconds: float,
    onset_tolerance: float = 0.050
) -> Dict[str, Any]:
    """
    Evaluates transcription accuracy on mixed audio with True Hz pitch checking,
    false notes per minute, and pitch error breakdown.
    """
    ref_intervals, ref_pitches_hz = load_notes_as_intervals(ref_midi_path, dedup_simultaneous=True)
    est_intervals, est_pitches_hz = load_notes_as_intervals(est_midi_path, dedup_simultaneous=True)

    if len(ref_intervals) == 0:
        return {"error": "Empty reference notes."}
    if len(est_intervals) == 0:
        return {
            "ref_notes": len(ref_intervals),
            "est_notes": 0,
            "onset_precision": 0.0,
            "onset_recall": 0.0,
            "onset_f1": 0.0,
            "false_notes_per_min": 0.0
        }

    # True Hz mir_eval evaluation (50 cents pitch tolerance)
    p_on, r_on, f1_on, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches_hz,
        est_intervals, est_pitches_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=50.0,
        offset_ratio=None
    )

    matched_pairs = mir_eval.transcription.match_notes(
        ref_intervals, ref_pitches_hz,
        est_intervals, est_pitches_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=50.0,
        offset_ratio=None
    )
    matched_est_indices = set(p[1] for p in matched_pairs)
    unmatched_est_indices = [i for i in range(len(est_intervals)) if i not in matched_est_indices]

    duration_min = max(0.1, duration_seconds / 60.0)
    false_notes_per_min = len(unmatched_est_indices) / duration_min

    # Classify pitch error breakdown for unmatched notes
    ref_pm = pretty_midi.PrettyMIDI(str(ref_midi_path))
    ref_notes_list = sorted([n for inst in ref_pm.instruments for n in inst.notes], key=lambda x: x.start)
    est_pm = pretty_midi.PrettyMIDI(str(est_midi_path))
    est_notes_list = sorted([n for inst in est_pm.instruments for n in inst.notes], key=lambda x: x.start)

    semitone_errors = 0
    octave_errors = 0
    other_errors = 0

    for idx in unmatched_est_indices:
        if idx >= len(est_notes_list):
            continue
        est_n = est_notes_list[idx]
        # Find closest reference note in time
        nearby_ref = [r for r in ref_notes_list if abs(r.start - est_n.start) <= 0.08]
        if nearby_ref:
            pitch_diff = min(abs(est_n.pitch - r.pitch) for r in nearby_ref)
            if pitch_diff in (1, 2):
                semitone_errors += 1
            elif pitch_diff in (12, 24):
                octave_errors += 1
            else:
                other_errors += 1
        else:
            other_errors += 1

    return {
        "ref_notes": len(ref_intervals),
        "est_notes": len(est_intervals),
        "onset_precision": round(float(p_on) * 100.0, 2),
        "onset_recall": round(float(r_on) * 100.0, 2),
        "onset_f1": round(float(f1_on) * 100.0, 2),
        "false_notes_count": len(unmatched_est_indices),
        "false_notes_per_min": round(false_notes_per_min, 1),
        "error_breakdown": {
            "semitone_errors": semitone_errors,
            "octave_errors": octave_errors,
            "other_errors": other_errors
        }
    }


def run_synthetic_benchmark_cli(workspace_dir: Path):
    from backend.pipeline import ProcessingPipeline
    print("=" * 72)
    print("SYNTHETIC MIXED-AUDIO BENCHMARK SUITE (True Hz mir_eval)")
    print("=" * 72)

    ref_midi = workspace_dir / "another love midi.mid"
    if not ref_midi.exists():
        print("Missing reference MIDI for synthetic benchmark.")
        return

    out_mix = workspace_dir / "storage" / "benchmark_synth_mix.wav"
    out_mix.parent.mkdir(parents=True, exist_ok=True)

    print("\n1. Rendering reference SoundFont piano audio...")
    mix_path, duration_s = create_synthetic_mix(
        piano_midi_path=ref_midi,
        output_mix_path=out_mix,
        aux_stem_path=None,  # Clean baseline test
        snr_db=6.0
    )
    print(f"Rendered {duration_s:.1f}s synthetic test track to {mix_path.name}.")

    print("\n2. Executing Studio Master pipeline on synthetic audio...")
    pipeline = ProcessingPipeline(task_id="synth_bench")
    res = pipeline.run(
        input_audio_path=mix_path,
        is_mixed_audio=False,
        precision_mode="clean",
        model_engine="studio_master"
    )

    print("\n3. Evaluating accuracy metrics against ground truth...")
    scores = evaluate_mixed_pipeline(ref_midi, Path(res["midi_file"]), duration_seconds=duration_s)
    print(f"  Reference Notes: {scores['ref_notes']} | Transcribed Notes: {scores['est_notes']}")
    print(f"  Onset Precision: {scores['onset_precision']}% | Recall: {scores['onset_recall']}% | F1: {scores['onset_f1']}%")
    print(f"  False Notes/Min: {scores['false_notes_per_min']} (Total False Notes: {scores['false_notes_count']})")
    print(f"  Error Breakdown: Semitone: {scores['error_breakdown']['semitone_errors']} | Octave: {scores['error_breakdown']['octave_errors']} | Other: {scores['error_breakdown']['other_errors']}")
    print("=" * 72)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run_synthetic_benchmark_cli(root)
