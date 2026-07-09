"""
Filter Ablation Study — Systematic Stage-by-Stage MIR Evaluation

Systematically evaluates each architectural pipeline stage:
  Stage 1: Raw ByteDance CRNN (pure neural baseline, zero post-processing)
  Stage 2: Raw + Calibrated Multi-Evidence Spectral Sieve
  Stage 3: Raw + Sieve + Non-Destructive Voicing Solver
  Stage 4: Raw + Sieve + Voicing Solver + Chord Display Alignment
  Stage 5: Full Production Studio Master Output

Evaluated against Official Ground Truth Reference MIDIs:
  - Another Love
  - Wicked Game
  - Idea 21
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import soundfile as sf
import librosa
import pretty_midi
from backend.config import DEVICE
from backend.studio_master import get_bytedance_model
from backend.spectral_sieve import verify_notes_with_spectral_sieve
from backend.voicing_solver import apply_voicing_solver
from backend.quantizer import align_chord_onsets
from backend.midi_parser import resolve_note_releases
from backend.benchmark_eval import evaluate_transcription


def run_filter_ablation_study(workspace_dir: Path) -> Dict[str, Dict[str, Any]]:
    test_pairs = [
        ("Another Love", workspace_dir / "another love.mp3", workspace_dir / "another love midi.mid", 21.4),
        ("Wicked Game", workspace_dir / "wicked game.mp3", workspace_dir / "wicked game midi.mid", 76.4),
        ("Idea 21", workspace_dir / "idea 21.mp3", workspace_dir / "idea 21 midi.mid", 58.6)
    ]

    out_dir = workspace_dir / "storage" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing ByteDance CRNN Transcriptor on {DEVICE}...")
    pt = get_bytedance_model()
    pt.onset_threshold = 0.45
    pt.frame_threshold = 0.18
    pt.offset_threshold = 0.28
    pt.offset_threshod = 0.28

    all_results = {}

    for name, audio_p, ref_mid_p, dur_s in test_pairs:
        if not audio_p.exists() or not ref_mid_p.exists():
            print(f"[Skip] {name}: Missing file.")
            continue

        print(f"\n{'='*88}\nSTAGE-BY-STAGE ABLATION STUDY: {name.upper()} ({dur_s}s)\n{'='*88}")
        track_results = {}

        # 1. Prepare 16kHz audio
        audio, sr = sf.read(str(audio_p))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio

        # ---------------------------------------------------------
        # Stage 1: Raw ByteDance CRNN (No filters)
        # ---------------------------------------------------------
        raw_mid = out_dir / f"{name.lower().replace(' ', '_')}_1_raw.mid"
        out_dict = pt.transcribe(audio_16k, str(raw_mid))
        reg_onsets = out_dict.get('output_dict', {}).get('reg_onset_output')
        frame_probs = out_dict.get('output_dict', {}).get('frame_output')

        scores_raw = evaluate_transcription(ref_mid_p, raw_mid, audio_duration_sec=dur_s)
        track_results["1_Raw_CRNN"] = scores_raw

        pm_raw = pretty_midi.PrettyMIDI(str(raw_mid))
        raw_notes = pm_raw.instruments[0].notes

        # ---------------------------------------------------------
        # Stage 2: Raw + Calibrated Multi-Evidence Spectral Sieve
        # ---------------------------------------------------------
        notes_clone = [pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end) for n in raw_notes]
        sieved_notes, sieve_stats = verify_notes_with_spectral_sieve(
            notes_clone,
            audio_16k=audio_16k,
            sr=16000,
            reg_onset_probs=reg_onsets,
            frame_probs=frame_probs,
            snr_threshold_db=4.2,
            mode="clean"
        )
        sieve_mid = out_dir / f"{name.lower().replace(' ', '_')}_2_sieve.mid"
        pm_sieve = pretty_midi.PrettyMIDI()
        inst_s = pretty_midi.Instrument(program=0)
        inst_s.notes = sieved_notes
        pm_sieve.instruments.append(inst_s)
        pm_sieve.write(str(sieve_mid))
        scores_sieve = evaluate_transcription(ref_mid_p, sieve_mid, audio_duration_sec=dur_s)
        track_results["2_Spectral_Sieve"] = scores_sieve

        # ---------------------------------------------------------
        # Stage 3: Sieve + Voicing Solver
        # ---------------------------------------------------------
        voiced_notes, _ = apply_voicing_solver(sieved_notes, tolerance=0.035)
        voiced_mid = out_dir / f"{name.lower().replace(' ', '_')}_3_voiced.mid"
        pm_voiced = pretty_midi.PrettyMIDI()
        inst_v = pretty_midi.Instrument(program=0)
        inst_v.notes = voiced_notes
        pm_voiced.instruments.append(inst_v)
        pm_voiced.write(str(voiced_mid))
        scores_voiced = evaluate_transcription(ref_mid_p, voiced_mid, audio_duration_sec=dur_s)
        track_results["3_Sieve_Voicing"] = scores_voiced

        # ---------------------------------------------------------
        # Stage 4: Sieve + Voicing Solver + Chord Alignment
        # ---------------------------------------------------------
        aligned_notes = align_chord_onsets([pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end) for n in voiced_notes], tolerance=0.035)
        align_mid = out_dir / f"{name.lower().replace(' ', '_')}_4_aligned.mid"
        pm_align = pretty_midi.PrettyMIDI()
        inst_a = pretty_midi.Instrument(program=0)
        inst_a.notes = aligned_notes
        pm_align.instruments.append(inst_a)
        pm_align.write(str(align_mid))
        scores_align = evaluate_transcription(ref_mid_p, align_mid, audio_duration_sec=dur_s)
        track_results["4_Display_Aligned"] = scores_align

        # ---------------------------------------------------------
        # Stage 5: Full Pipeline Export
        # ---------------------------------------------------------
        resolved_notes = resolve_note_releases(voiced_notes)
        full_mid = out_dir / f"{name.lower().replace(' ', '_')}_5_full.mid"
        pm_full = pretty_midi.PrettyMIDI()
        inst_f = pretty_midi.Instrument(program=0)
        inst_f.notes = resolved_notes
        pm_full.instruments.append(inst_f)
        pm_full.write(str(full_mid))
        scores_full = evaluate_transcription(ref_mid_p, full_mid, audio_duration_sec=dur_s)
        track_results["5_Full_Pipeline"] = scores_full

        # Print Table
        print(f"Reference Notes: {scores_raw['ref_notes']} | Time Lag: {scores_raw['time_lag']:+.3f}s")
        print(f"{'Stage / Configuration':<24} | {'Notes':<6} | {'Onset P':<8} {'Onset R':<8} {'Onset F1':<8} | {'False/Min':<10} | {'Oct Ghosts':<10}")
        print("-" * 88)
        for cfg_name, sc in track_results.items():
            al = sc["aligned"]
            diag = sc.get("diagnosis", {})
            oct_gh = diag.get("by_harmonic_type", {}).get("octave_ghosts", 0) + diag.get("by_harmonic_type", {}).get("sub_octave_ghosts", 0)
            false_pm = f"{sc.get('false_notes_per_min', 0.0):.1f}/m"
            print(f"{cfg_name:<24} | {sc['est_notes']:<6} | {al['onset_precision']:<8.1%} {al['onset_recall']:<8.1%} {al['onset_f1']:<8.1%} | {false_pm:<10} | {oct_gh:<10}")

        all_results[name] = track_results

    return all_results


if __name__ == "__main__":
    run_filter_ablation_study(ROOT)
