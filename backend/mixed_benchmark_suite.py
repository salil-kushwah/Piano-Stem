"""
Mixed Audio Evaluation Suite — Multi-Regime Benchmark for Piano AI Studio

Evaluates the two mixed-audio production regimes against ground truth:
  Regime A: Full Band Mix (Drums, Bass, Guitars, Vocals, Piano)
            Tests BS-RoFormer 6-stem isolation + Multi-Stem Bleed Guard Matrix (Vocal + Drum + Bass + Cymbal Comb Killer).
  Regime B: Piano & Vocal Duet (Sparse Acoustic Arrangement)
            Tests High-Purity 2-stem UVR InstVoc (Zero Piano Cannibalization) vs 6-stem baseline.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import soundfile as sf
import librosa
import pretty_midi

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline import ProcessingPipeline
from backend.benchmark_eval import evaluate_transcription
from backend.synth import synthesize_midi_to_audio


def run_mixed_audio_benchmarks(workspace_dir: Path):
    print("=" * 88)
    print("PIANO AI STUDIO — SOTA MIXED AUDIO BENCHMARK SUITE")
    print("=" * 88)

    stems_dir = workspace_dir / "storage" / "stems"
    out_dir = workspace_dir / "storage" / "mixed_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Regime B Benchmark: Piano + Vocal Duet (Another Love)
    # -------------------------------------------------------------
    al_mp3 = workspace_dir / "another love.mp3"
    al_midi = workspace_dir / "another love midi.mid"

    if al_mp3.exists() and al_midi.exists():
        print("\n" + "#" * 80)
        print("TEST 1: REGIME B — ACOUSTIC PIANO & VOCAL DUET (Another Love)")
        print("Evaluating High-Purity 2-Stem InstVoc (Zero Piano Cannibalization) vs Raw Mix")
        print("#" * 80)

        # Run with Auto-Detect (which selects mdx_inst for duets)
        t_start = time.time()
        pipeline_duet = ProcessingPipeline(task_id="bench_al_duet")
        res_duet = pipeline_duet.run(
            input_audio_path=al_mp3,
            is_mixed_audio=True,
            precision_mode="clean",
            model_engine="studio_master",
            stem_mode="auto"
        )
        elapsed_duet = round(time.time() - t_start, 2)

        dur_duet = sf.info(str(al_mp3)).duration
        scores_duet = evaluate_transcription(al_midi, Path(res_duet["midi_file"]), audio_duration_sec=dur_duet)
        al_aligned = scores_duet["aligned"]
        print(f"\n[Result: Regime B Duet] Elapsed: {elapsed_duet}s (Duration: {dur_duet:.1f}s)")
        print(f"  Reference Notes: {scores_duet['ref_notes']} | Predictions: {scores_duet['est_notes']}")
        print(f"  Onset Precision: {al_aligned['onset_precision']:.1%} | Onset Recall: {al_aligned['onset_recall']:.1%}")
        print(f"  Onset F1 Score:  {al_aligned['onset_f1']:.1%}")
        lbl = res_duet.get('mix_regime', {}).get('label', 'Duet') if res_duet.get('mix_regime') else 'Duet'
        print(f"  Active Profile: {str(lbl).encode('ascii', errors='replace').decode('ascii')}")

    # -------------------------------------------------------------
    # 2. Regime A Benchmark: Full Band Mix with Drums & Bass
    # -------------------------------------------------------------
    p_path = stems_dir / "Lucifer__piano.wav"
    d_path = stems_dir / "Lucifer__drums.wav"
    b_path = stems_dir / "Lucifer__bass.wav"
    v_path = stems_dir / "Lucifer__vocals.wav"
    ref_midi = workspace_dir / "another love midi.mid"
    clean_piano_wav = out_dir / "ref_piano_soundfont.wav"

    if d_path.exists() and b_path.exists() and ref_midi.exists():
        print("\n" + "#" * 80)
        print("TEST 2: REGIME A — FULL BAND MIX (Piano + Real Drums + Real Bass + Real Vocals)")
        print("Evaluating BS-RoFormer 6S Isolation + Transkun SOTA Transcription")
        print("#" * 80)

        # Synthesize ground truth piano audio via SoundFont
        synthesize_midi_to_audio(ref_midi, clean_piano_wav, sample_rate=44100)
        p, sr = sf.read(str(clean_piano_wav))
        if p.ndim > 1: p = p.mean(axis=1)

        v, _ = sf.read(str(v_path))
        d, _ = sf.read(str(d_path))
        b, _ = sf.read(str(b_path))

        if v.ndim > 1: v = v.mean(axis=1)
        if d.ndim > 1: d = d.mean(axis=1)
        if b.ndim > 1: b = b.mean(axis=1)

        dur_samples = min(len(p), sr * 30)
        p = p[:dur_samples]
        v = (np.tile(v, int(np.ceil(dur_samples / len(v)))) if len(v) < dur_samples else v)[:dur_samples]
        d = (np.tile(d, int(np.ceil(dur_samples / len(d)))) if len(d) < dur_samples else d)[:dur_samples]
        b = (np.tile(b, int(np.ceil(dur_samples / len(b)))) if len(b) < dur_samples else b)[:dur_samples]

        # Blend into high-energy Full Band Mix (0 dB SNR: equal piano vs band energy)
        p_rms = np.sqrt(np.mean(p ** 2) + 1e-12)
        band = 0.5 * v + 0.4 * d + 0.3 * b
        band_rms = np.sqrt(np.mean(band ** 2) + 1e-12)
        full_mix_audio = p + band * (p_rms / (band_rms + 1e-12))

        peak = np.max(np.abs(full_mix_audio))
        if peak > 0.95:
            full_mix_audio = (full_mix_audio / peak) * 0.95

        mix_wav = out_dir / "synthetic_full_band_mix.wav"
        sf.write(str(mix_wav), full_mix_audio, sr, subtype="FLOAT")

        t_start_a = time.time()
        pipeline_full = ProcessingPipeline(task_id="bench_full_mix")
        res_full = pipeline_full.run(
            input_audio_path=mix_wav,
            is_mixed_audio=True,
            precision_mode="clean",
            model_engine="studio_master",
            stem_mode="bs_roformer"
        )
        elapsed_full = round(time.time() - t_start_a, 2)

        dur_mix = len(full_mix_audio) / sr
        scores_full = evaluate_transcription(ref_midi, Path(res_full["midi_file"]), audio_duration_sec=dur_mix)
        wg_aligned = scores_full["aligned"]
        print(f"\n[Result: Regime A Full Mix] Elapsed: {elapsed_full}s (Duration: {dur_mix:.1f}s)")
        print(f"  Reference Notes: {scores_full['ref_notes']} | Predictions: {scores_full['est_notes']}")
        print(f"  Onset Precision: {wg_aligned['onset_precision']:.1%} | Onset Recall: {wg_aligned['onset_recall']:.1%}")
        print(f"  Onset F1 Score:  {wg_aligned['onset_f1']:.1%}")
        print(f"  False Notes / Min: {scores_full.get('false_notes_per_min', 0.0):.1f}/m")

    print("\n" + "=" * 88)
    print("MIXED AUDIO BENCHMARK SUITE COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    run_mixed_audio_benchmarks(ROOT)
