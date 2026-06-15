"""
Spectral Sieve — Calibrated Multi-Evidence Note Verification Engine

Verifies candidate transcribed MIDI notes against true audio spectral acoustics:
  1. Calibrated STFT in dBFS (full scale sinusoid = 0.0 dBFS, noise floor ~ -70 to -90 dBFS).
  2. Chord-Aware Local Spectral SNR: Masks adjacent chord notes from the local noise floor
     to ensure real chord tones are never falsely rejected due to elevated local energy.
  3. Harmonic & Sub-Harmonic Multi-Evidence Verification:
     - Upper octaves (+12, +19, +24, +28) are tested for independent odd harmonic excitation (3 * f₀)
       and neural onset probability. Without independent physical excitation, candidate overtones
       are identified as sympathetic string resonance.
     - Sub-octaves (-12, -24) are tested against true fundamental energy to reject combination-tone phantoms.
  4. Cross-Stem Demixer Leakage Rejection: If vocal or drum stems are provided, compares spectral energy
     at f₀ between piano stem and auxiliary stem to reject non-piano bleed.
  5. Multi-Mode Operation:
     - "clean": Conservative precision mode (optimal for commercial mixes and benchmark precision).
     - "balanced": Default mode balancing high recall and high precision.
     - "sensitive": Minimal filtering, preserving faint pianissimo touches.
     - "diagnostic": Does NOT delete any candidate notes; flags suspicious notes with full provenance.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pretty_midi


STFT_N_FFT = 2048          # ~7.8 Hz resolution at 16kHz sample rate (~128ms window)
STFT_HOP_LENGTH = 256      # 16ms hop resolution at 16kHz
FREQ_TOLERANCE_CENTS = 65  # ±65 cents tolerance around expected fundamental
MIN_SNR_DB = 3.2           # Peak at f₀ must stand at least 3.2 dB above local frequency floor
MIN_ABS_ENERGY_DBFS = -72.0 # Absolute floor below which a note is inaudible residual
CROSS_STEM_MARGIN_DB = 6.5 # Reject if auxiliary stem is > 6.5 dB louder at f₀ than piano stem


def midi_to_freq(midi_note: int) -> float:
    """Converts MIDI note number to fundamental frequency in Hz."""
    return 440.0 * (2.0 ** ((float(midi_note) - 69.0) / 12.0))


def freq_to_bin(freq: float, sr: int, n_fft: int = STFT_N_FFT) -> int:
    """Converts frequency in Hz to FFT frequency bin index."""
    return int(round(freq * n_fft / float(sr)))


def time_to_frame(time_sec: float, sr: int, hop_length: int = STFT_HOP_LENGTH) -> int:
    """Converts time in seconds to STFT frame index."""
    return int(round(time_sec * float(sr) / float(hop_length)))


def compute_calibrated_stft_db(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = STFT_N_FFT,
    hop_length: int = STFT_HOP_LENGTH
) -> np.ndarray:
    """
    Computes calibrated magnitude STFT in dBFS.
    A full-scale sinusoid (amplitude 1.0) produces 0.0 dBFS.
    Shape: (n_bins, n_frames)
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)), mode='constant')

    window = np.hanning(n_fft).astype(np.float32)
    win_scale = float(np.sum(window) / 2.0)

    pad_length = n_fft // 2
    audio_padded = np.pad(audio, (pad_length, pad_length), mode='reflect')
    n_frames = 1 + (len(audio_padded) - n_fft) // hop_length

    stft_mag = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        segment = audio_padded[start:start + n_fft] * window
        spectrum = np.abs(np.fft.rfft(segment))
        stft_mag[:, i] = spectrum / win_scale

    stft_db = 20.0 * np.log10(np.maximum(stft_mag, 1e-6))
    return stft_db


def verify_notes_with_spectral_sieve(
    notes: List[pretty_midi.Note],
    audio_16k: np.ndarray,
    sr: int = 16000,
    vocal_audio_16k: Optional[np.ndarray] = None,
    drums_audio_16k: Optional[np.ndarray] = None,
    bass_audio_16k: Optional[np.ndarray] = None,
    reg_onset_probs: Optional[np.ndarray] = None,
    frame_probs: Optional[np.ndarray] = None,
    snr_threshold_db: float = MIN_SNR_DB,
    mode: str = "clean"
) -> Tuple[List[pretty_midi.Note], Dict[str, Any]]:
    """
    Evaluates candidate notes against audio acoustics and auxiliary stems.
    
    Args:
        notes: List of pretty_midi.Note candidates from transcription.
        audio_16k: Monophonic audio resampled to 16kHz.
        sr: Sample rate (default 16000).
        vocal_audio_16k: Optional isolated vocal stem for cross-stem bleed check.
        drums_audio_16k: Optional isolated drum stem for percussive bleed check.
        bass_audio_16k: Optional isolated bass stem for low-end bass guitar/kick bleed check.
        reg_onset_probs: Optional ByteDance neural onset probability tensor.
        frame_probs: Optional ByteDance neural frame probability tensor.
        snr_threshold_db: Minimum peak SNR in dB above local noise floor.
        mode: "clean", "balanced", "sensitive", or "diagnostic".
    
    Returns:
        (retained_notes, stats)
    """
    if not notes or len(audio_16k) == 0:
        return notes, {
            "total_candidates": len(notes),
            "retained": len(notes),
            "rejected_snr": 0,
            "rejected_harmonics": 0,
            "rejected_vocal_bleed": 0,
            "rejected_drum_bleed": 0,
            "rejected_bass_bleed": 0,
            "rejected_percussion_comb": 0,
            "rejected_confidence": 0,
            "total_rejected": 0,
            "mode": mode,
            "provenance": []
        }

    # Compute calibrated piano spectrogram
    stft_piano = compute_calibrated_stft_db(audio_16k, sr=sr)
    n_bins, n_frames = stft_piano.shape

    # Optionally compute auxiliary stem spectrograms if provided
    stft_vocal = None
    if vocal_audio_16k is not None and len(vocal_audio_16k) > 0:
        stft_vocal = compute_calibrated_stft_db(vocal_audio_16k, sr=sr)

    stft_drums = None
    if drums_audio_16k is not None and len(drums_audio_16k) > 0:
        stft_drums = compute_calibrated_stft_db(drums_audio_16k, sr=sr)

    stft_bass = None
    if bass_audio_16k is not None and len(bass_audio_16k) > 0:
        stft_bass = compute_calibrated_stft_db(bass_audio_16k, sr=sr)

    sorted_notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    retained: List[pretty_midi.Note] = []
    provenance_list: List[Dict[str, Any]] = []

    rejected_snr = 0
    rejected_harmonics = 0
    rejected_vocal_bleed = 0
    rejected_drum_bleed = 0
    rejected_bass_bleed = 0
    rejected_percussion_comb = 0
    rejected_confidence = 0

    is_diagnostic = (mode == "diagnostic")

    for idx, note in enumerate(sorted_notes):
        pitch = note.pitch
        f0 = midi_to_freq(pitch)
        bin_f0 = freq_to_bin(f0, sr, STFT_N_FFT)
        onset_fr = time_to_frame(note.start, sr, STFT_HOP_LENGTH)
        onset_fr = max(0, min(n_frames - 1, onset_fr))

        # -------------------------------------------------------------
        # 1. Neural Model Confidence Extraction
        # -------------------------------------------------------------
        on_prob = 0.5
        fr_prob = 0.5
        if reg_onset_probs is not None:
            p_idx = pitch - 21
            if 0 <= p_idx < 88:
                f_idx = int(round(note.start * 100))
                fr_min = max(0, f_idx - 2)
                fr_max = min(len(reg_onset_probs), f_idx + 3)
                if fr_max > fr_min:
                    on_prob = float(np.max(reg_onset_probs[fr_min:fr_max, p_idx]))
                if frame_probs is not None and len(frame_probs) > fr_min:
                    fr_prob = float(np.max(frame_probs[fr_min:min(len(frame_probs), f_idx + 3), p_idx]))

        note_provenance = {
            "pitch": pitch,
            "start": round(note.start, 4),
            "end": round(note.end, 4),
            "velocity": note.velocity,
            "onset_prob": round(on_prob, 4),
            "frame_prob": round(fr_prob, 4),
            "status": "accepted",
            "rejection_reason": None,
            "evidence": {}
        }

        # Safe guard: high confidence neural hits with moderate velocity are rarely false
        has_high_confidence = (on_prob >= 0.55)

        # -------------------------------------------------------------
        # 2. Harmonic & Sub-Harmonic Explained-Away Filter
        # -------------------------------------------------------------
        is_harmonic_ghost = False
        harmonic_reason = None

        # Look in neighboring temporal window (both concurrent strikes and ringing held notes)
        local_context = sorted_notes[max(0, idx - 25):min(len(sorted_notes), idx + 26)]
        for prev in local_context:
            if prev is note:
                continue

            # Upper octave check (+12, +19, +24, +28 semitones above a lower active fundamental)
            if prev.pitch < note.pitch:
                is_concurrent = abs(prev.start - note.start) <= 0.050
                is_sounding = (prev.start <= note.start <= (prev.end + 0.15))
                if is_concurrent or is_sounding:
                    interval = note.pitch - prev.pitch
                    if interval in (12, 19, 24, 28):
                        # Extract lower fundamental's onset confidence
                        prev_on = 0.5
                        if reg_onset_probs is not None:
                            p_low_idx = prev.pitch - 21
                            if 0 <= p_low_idx < 88:
                                prev_f_idx = int(round(prev.start * 100))
                                prev_on = float(np.max(reg_onset_probs[max(0, prev_f_idx - 2):min(len(reg_onset_probs), prev_f_idx + 3), p_low_idx]))

                        # Check independent 3rd harmonic of the higher candidate note (3 * f0)
                        b_h3 = freq_to_bin(f0 * 3, sr, STFT_N_FFT)
                        h3_energy = float(stft_piano[min(n_bins - 1, b_h3), onset_fr]) if b_h3 < n_bins else -80.0
                        peak_energy = float(stft_piano[min(n_bins - 1, bin_f0), onset_fr])
                        diff_h3 = peak_energy - h3_energy

                        note_provenance["evidence"]["harmonic_check"] = {
                            "lower_fundamental_pitch": prev.pitch,
                            "interval": interval,
                            "lower_onset_prob": round(prev_on, 3),
                            "h3_energy_dbfs": round(h3_energy, 1),
                            "peak_energy_dbfs": round(peak_energy, 1),
                            "diff_h3_db": round(diff_h3, 1)
                        }

                        # If lower fundamental has superior confidence, and higher note lacks independent 3rd harmonic
                        if (prev_on > 0.70 and on_prob < 0.52) and (h3_energy < -74.0 and diff_h3 > 28.0):
                            is_harmonic_ghost = True
                            harmonic_reason = f"Upper harmonic ({interval} semitones) of {prev.pitch} without independent 3rd harmonic"
                            break

            # Sub-octave check (candidate note is 12 or 24 semitones BELOW a loud fundamental)
            if prev.pitch > note.pitch:
                is_concurrent = abs(prev.start - note.start) <= 0.050
                if is_concurrent:
                    interval_down = prev.pitch - note.pitch
                    if interval_down in (12, 24):
                        prev_on = 0.5
                        if reg_onset_probs is not None:
                            p_hi_idx = prev.pitch - 21
                            if 0 <= p_hi_idx < 88:
                                prev_f_idx = int(round(prev.start * 100))
                                prev_on = float(np.max(reg_onset_probs[max(0, prev_f_idx - 2):min(len(reg_onset_probs), prev_f_idx + 3), p_hi_idx]))

                        peak_energy = float(stft_piano[min(n_bins - 1, bin_f0), onset_fr])
                        prev_bin = freq_to_bin(midi_to_freq(prev.pitch), sr, STFT_N_FFT)
                        prev_peak = float(stft_piano[min(n_bins - 1, prev_bin), onset_fr])

                        if prev_on > 0.70 and on_prob < 0.50 and peak_energy < (prev_peak - 18.0):
                            is_harmonic_ghost = True
                            harmonic_reason = f"Sub-octave ghost (-{interval_down} semitones) below fundamental {prev.pitch}"
                            break

        if is_harmonic_ghost:
            rejected_harmonics += 1
            note_provenance["status"] = "flagged" if is_diagnostic else "rejected"
            note_provenance["rejection_reason"] = harmonic_reason
            provenance_list.append(note_provenance)
            if not is_diagnostic:
                continue

        # -------------------------------------------------------------
        # 3. Chord-Aware Local Spectral Peak SNR & Attack Delta Verification
        # -------------------------------------------------------------
        # Mask out bins belonging to candidate notes in the same simultaneous cluster
        cluster = [other for other in local_context if abs(other.start - note.start) <= 0.045]
        cluster_bins = set()
        for c in cluster:
            cb = freq_to_bin(midi_to_freq(c.pitch), sr, STFT_N_FFT)
            for b in range(cb - 2, cb + 3):
                cluster_bins.add(b)

        ratio = 2.0 ** (FREQ_TOLERANCE_CENTS / 1200.0)
        bin_lo = max(0, freq_to_bin(f0 / ratio, sr, STFT_N_FFT))
        bin_hi = min(n_bins - 1, freq_to_bin(f0 * ratio, sr, STFT_N_FFT))

        fr_start = max(0, onset_fr - 1)
        fr_end = min(n_frames, onset_fr + 3)
        onset_window = stft_piano[bin_lo:bin_hi + 1, fr_start:fr_end]
        if onset_window.size == 0:
            retained.append(note)
            provenance_list.append(note_provenance)
            continue

        peak_db = float(np.max(onset_window))

        # Reverb-Resistant Local Noise Floor: compare contemporaneous floor with pre-onset floor
        # to ensure lingering reverberation wash from earlier notes doesn't falsely elevate the floor
        search_bins = [b for b in range(max(0, bin_lo - 12), min(n_bins, bin_hi + 13)) if b not in cluster_bins]
        contemp_floor = float(np.median(stft_piano[search_bins, onset_fr])) if search_bins else -75.0
        pre_fr = max(0, onset_fr - 2)
        pre_floor = float(np.median(stft_piano[search_bins, pre_fr])) if search_bins else -75.0
        local_floor_db = min(contemp_floor, pre_floor)
        snr = peak_db - local_floor_db

        # Temporal attack flux: energy jump at f0 relative to pre-onset frame
        pre_energy = float(np.max(stft_piano[bin_lo:bin_hi + 1, max(0, onset_fr - 2):max(1, onset_fr)])) if onset_fr >= 1 else -80.0
        temporal_onset_delta = peak_db - pre_energy

        note_provenance["evidence"]["spectral_snr"] = {
            "peak_dbfs": round(peak_db, 1),
            "local_floor_dbfs": round(local_floor_db, 1),
            "snr_db": round(snr, 1),
            "temporal_onset_delta_db": round(temporal_onset_delta, 1)
        }

        # Notes must have reasonable SNR and absolute energy, unless neural onset confidence is high
        # or there is a confirmed physical hammer attack transient (>= 3.5 dB jump)
        has_attack_transient = (temporal_onset_delta >= 3.5)
        has_high_confidence = (on_prob >= 0.45 or (on_prob >= 0.35 and note.velocity >= 40))
        snr_floor = snr_threshold_db if mode == "clean" else (snr_threshold_db - 0.8)
        is_sub_audible = (peak_db < MIN_ABS_ENERGY_DBFS)
        is_low_snr = (snr < snr_floor and on_prob < 0.45 and note.velocity < 35 and not has_attack_transient)

        if not has_high_confidence and (is_sub_audible or is_low_snr):
            rejected_snr += 1
            reason = "Sub-audible energy floor" if is_sub_audible else f"Low spectral SNR ({snr:.1f} dB < {snr_floor:.1f} dB)"
            note_provenance["status"] = "flagged" if is_diagnostic else "rejected"
            note_provenance["rejection_reason"] = reason
            provenance_list.append(note_provenance)
            if not is_diagnostic:
                continue

        # -------------------------------------------------------------
        # 4. Cross-Stem Vocal Leakage Check
        # -------------------------------------------------------------
        if stft_vocal is not None and onset_fr < stft_vocal.shape[1]:
            v_fr_start = max(0, onset_fr - 1)
            v_fr_end = min(stft_vocal.shape[1], onset_fr + 3)
            v_window = stft_vocal[bin_lo:bin_hi + 1, v_fr_start:v_fr_end]
            if v_window.size > 0:
                vocal_peak_db = float(np.max(v_window))
                note_provenance["evidence"]["vocal_bleed"] = {
                    "vocal_peak_dbfs": round(vocal_peak_db, 1),
                    "piano_peak_dbfs": round(peak_db, 1)
                }
                if vocal_peak_db > (peak_db + CROSS_STEM_MARGIN_DB) and note.velocity < 50 and on_prob < 0.52:
                    rejected_vocal_bleed += 1
                    note_provenance["status"] = "flagged" if is_diagnostic else "rejected"
                    note_provenance["rejection_reason"] = f"Vocal stem leakage (+{vocal_peak_db - peak_db:.1f} dB above piano)"
                    provenance_list.append(note_provenance)
                    if not is_diagnostic:
                        continue

        # -------------------------------------------------------------
        # 5. Cross-Stem Drum Leakage Check
        # -------------------------------------------------------------
        if stft_drums is not None and onset_fr < stft_drums.shape[1]:
            d_fr_start = max(0, onset_fr - 1)
            d_fr_end = min(stft_drums.shape[1], onset_fr + 3)
            d_window = stft_drums[bin_lo:bin_hi + 1, d_fr_start:d_fr_end]
            if d_window.size > 0:
                drum_peak_db = float(np.max(d_window))
                note_dur = note.end - note.start
                note_provenance["evidence"]["drum_bleed"] = {
                    "drum_peak_dbfs": round(drum_peak_db, 1),
                    "piano_peak_dbfs": round(peak_db, 1),
                    "duration": round(note_dur, 3)
                }
                if drum_peak_db > (peak_db + CROSS_STEM_MARGIN_DB) and (note.velocity < 40 or note_dur < 0.10) and on_prob < 0.50:
                    rejected_drum_bleed += 1
                    note_provenance["status"] = "flagged" if is_diagnostic else "rejected"
                    note_provenance["rejection_reason"] = f"Drum transient leakage (+{drum_peak_db - peak_db:.1f} dB)"
                    provenance_list.append(note_provenance)
                    if not is_diagnostic:
                        continue

        # -------------------------------------------------------------
        # 6. Cross-Stem Bass Guitar & Kick Bleed Check (Pitch < 45 / < 110 Hz)
        # -------------------------------------------------------------
        if stft_bass is not None and pitch < 45 and onset_fr < stft_bass.shape[1]:
            b_fr_start = max(0, onset_fr - 1)
            b_fr_end = min(stft_bass.shape[1], onset_fr + 3)
            b_window = stft_bass[bin_lo:bin_hi + 1, b_fr_start:b_fr_end]
            if b_window.size > 0:
                bass_peak_db = float(np.max(b_window))
                note_provenance["evidence"]["bass_bleed"] = {
                    "bass_peak_dbfs": round(bass_peak_db, 1),
                    "piano_peak_dbfs": round(peak_db, 1)
                }
                # If bass stem energy at f0 exceeds piano energy and neural onset prob is not high:
                if bass_peak_db > (peak_db + 1.2) and on_prob < 0.70 and note.velocity < 60:
                    rejected_bass_bleed += 1
                    note_provenance["status"] = "flagged" if is_diagnostic else "rejected"
                    note_provenance["rejection_reason"] = f"Bass stem leakage (+{bass_peak_db - peak_db:.1f} dB above piano)"
                    provenance_list.append(note_provenance)
                    if not is_diagnostic:
                        continue

        # -------------------------------------------------------------
        # 7. Treble Percussion Disambiguation (Cymbal / Hi-Hat Killer)
        # -------------------------------------------------------------
        # In treble registers (MIDI >= 72, f0 >= 523 Hz), cymbals produce broadband diffuse noise.
        # Genuine piano strings have discrete harmonics (2*f0 and 3*f0) with clear inter-harmonic dips.
        if pitch >= 72 and onset_fr < n_frames:
            f0_h2 = f0 * 2.0
            if f0_h2 < (sr / 2.0 - 200):
                b_h2 = freq_to_bin(f0_h2, sr, STFT_N_FFT)
                h2_energy = float(stft_piano[b_h2, onset_fr])

                # Inter-harmonic midpoint between f0 and 2*f0 (1.5 * f0)
                b_mid = freq_to_bin(f0 * 1.5, sr, STFT_N_FFT)
                mid_noise = float(stft_piano[b_mid, onset_fr])

                # Harmonic Comb Contrast: energy contrast between harmonic peak and inter-harmonic trough
                comb_contrast = h2_energy - mid_noise
                note_provenance["evidence"]["treble_comb_contrast"] = round(comb_contrast, 1)

                # If drum energy is present and there is low harmonic contrast (diffuse noise splash)
                has_drum_presence = (stft_drums is not None and float(np.max(stft_drums[bin_lo:bin_hi + 1, fr_start:fr_end])) > -60.0)
                if has_drum_presence and comb_contrast < 1.0 and on_prob < 0.68 and note.velocity < 50:
                    rejected_percussion_comb += 1
                    note_provenance["status"] = "flagged" if is_diagnostic else "rejected"
                    note_provenance["rejection_reason"] = f"Cymbal / hi-hat transient splash (flat spectrum, contrast {comb_contrast:.1f} dB)"
                    provenance_list.append(note_provenance)
                    if not is_diagnostic:
                        continue

        # Note successfully validated
        provenance_list.append(note_provenance)
        retained.append(note)

    stats = {
        "total_candidates": len(notes),
        "retained": len(retained) if not is_diagnostic else len(notes),
        "rejected_snr": rejected_snr,
        "rejected_harmonics": rejected_harmonics,
        "rejected_vocal_bleed": rejected_vocal_bleed,
        "rejected_drum_bleed": rejected_drum_bleed,
        "rejected_bass_bleed": rejected_bass_bleed,
        "rejected_percussion_comb": rejected_percussion_comb,
        "rejected_confidence": rejected_confidence,
        "total_rejected": len(notes) - len(retained),
        "mode": mode,
        "provenance": provenance_list
    }

    output_notes = notes if is_diagnostic else retained
    return output_notes, stats


def spectral_sieve_filter(
    midi_path: Path,
    audio: np.ndarray,
    sr: int = 16000,
    strict_mode: bool = False
) -> Path:
    """Backward-compatible file-based wrapper for spectral sieve."""
    try:
        mid = pretty_midi.PrettyMIDI(str(midi_path))
        if not mid.instruments:
            return midi_path

        notes = mid.instruments[0].notes
        snr_th = MIN_SNR_DB + (1.0 if strict_mode else 0.0)
        retained, stats = verify_notes_with_spectral_sieve(
            notes,
            audio,
            sr=sr,
            snr_threshold_db=snr_th,
            mode="clean" if strict_mode else "balanced"
        )
        mid.instruments[0].notes = retained
        mid.write(str(midi_path))
        print(f"[Spectral Sieve] Retained {len(retained)}/{len(notes)} notes. Stats: {stats}")
        return midi_path
    except Exception as e:
        print(f"[Spectral Sieve] Filter error: {e}")
        return midi_path
