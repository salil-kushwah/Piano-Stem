"""
Audio Mix Regime Classifier — Intelligent Mix Analysis for Piano AI Studio

Automatically detects audio composition to determine optimal demixing route:
  1. Regime A: Full Band Mix (Drums, Bass, Guitars, Vocals, Piano) -> BS-RoFormer 6S
  2. Regime B: Piano + Vocal Duet (Sparse Acoustic Arrangement)   -> UVR InstVoc HQ (Zero Piano Cannibalization)
  3. Regime C: Solo Piano Recording (No voices or instruments)     -> Direct Acoustic Processing
"""

from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import soundfile as sf
import librosa


def detect_audio_regime(
    audio_path: Union[str, Path],
    max_duration_sec: float = 45.0
) -> Dict[str, Any]:
    """
    Analyzes an audio recording and classifies it into one of three production regimes:
      - 'full_mix': Heavy accompaniment with drums/bass/guitars. Needs 6-stem BS-RoFormer.
      - 'vocal_piano_duet': Acoustic voice + piano only. Needs 2-stem InstVoc to prevent cannibalization.
      - 'solo_piano': Clean acoustic piano alone. Zero demixing.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Load up to max_duration_sec at 22.05 kHz for fast, accurate MIR analysis
    y, sr = librosa.load(str(path), sr=22050, mono=True, duration=max_duration_sec)
    total_len = len(y)
    if total_len < sr * 2:  # Less than 2 seconds
        return {
            "regime": "solo_piano",
            "label": "🎹 Solo Piano Recording",
            "recommended_stem_mode": "none",
            "confidence": 0.50,
            "metrics": {}
        }

    # 1. Harmonic-Percussive Source Separation (HPSS) Energy Ratio
    y_harm, y_perc = librosa.effects.hpss(y, margin=(1.0, 3.0))
    e_total = float(np.sum(y ** 2) + 1e-9)
    e_perc = float(np.sum(y_perc ** 2))
    e_harm = float(np.sum(y_harm ** 2))
    perc_ratio = e_perc / e_total

    # 2. Rhythmic Beat Strength & Periodic Drum Pulse Detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo_val = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)

    # Autocorrelation of onset envelope measures regular drum groove/tempo periodicity
    ac = librosa.autocorrelate(onset_env, max_size=int(sr / 512 * 2))  # 2-second max lag
    if len(ac) > 1 and ac[0] > 0:
        norm_ac = ac / ac[0]
        peak_periodicity = float(np.max(norm_ac[10:])) if len(norm_ac) > 10 else 0.0
    else:
        peak_periodicity = 0.0

    # 3. Sub-Bass Energy Ratio (Kick Drum & Bass Guitar: 30 - 120 Hz)
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    sub_bass_bins = np.where((freqs >= 30) & (freqs <= 120))[0]
    mid_treble_bins = np.where((freqs > 120) & (freqs <= 6000))[0]

    sub_bass_energy = float(np.sum(stft[sub_bass_bins, :] ** 2))
    total_band_energy = float(np.sum(stft[mid_treble_bins, :] ** 2) + 1e-9)
    bass_ratio = sub_bass_energy / total_band_energy

    # 4. Vocal Consonant Sibilance & Fricative Detection (Zero-Crossing Rate Spikes)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_spikes = int(np.sum(zcr > 0.12))

    metrics = {
        "percussive_ratio": round(perc_ratio, 4),
        "rhythmic_periodicity": round(peak_periodicity, 4),
        "bass_ratio": round(bass_ratio, 4),
        "vocal_zcr_spikes": zcr_spikes,
        "detected_tempo": round(tempo_val, 1)
    }

    # Classification Decision Matrix:
    has_full_drums = (perc_ratio >= 0.08 or (perc_ratio >= 0.055 and peak_periodicity >= 0.42))
    has_heavy_bass = (bass_ratio >= 0.10)
    has_vocals = (zcr_spikes >= 45 or perc_ratio >= 0.032)

    if has_full_drums and (has_heavy_bass or peak_periodicity >= 0.45):
        regime = "full_mix"
        label = "Full Band Mix (Drums, Bass, Guitars, Vocals, Piano)"
        rec_mode = "bs_roformer"
        confidence = min(0.95, 0.70 + perc_ratio + (peak_periodicity * 0.2))
    elif has_full_drums:
        regime = "full_mix"
        label = "Full Band Mix (Percussion & Accompaniment Detected)"
        rec_mode = "bs_roformer"
        confidence = 0.85
    elif has_vocals:
        # Sparse acoustic mix: voice present, but no drum kit/bass guitar -> Acoustic Duet
        regime = "vocal_piano_duet"
        label = "Piano & Vocal Duet (Sparse Acoustic Arrangement)"
        rec_mode = "mdx_inst"
        confidence = 0.90
    else:
        # Extremely clean harmonic spectrum, minimal vocal sibilance, minimal percussive energy
        regime = "solo_piano"
        label = "Solo Piano Recording"
        rec_mode = "none"
        confidence = 0.92

    return {
        "regime": regime,
        "label": label,
        "recommended_stem_mode": rec_mode,
        "confidence": round(confidence, 2),
        "metrics": metrics
    }
