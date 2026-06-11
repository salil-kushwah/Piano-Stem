"""
backend/adaptive_decoder.py
===========================
Adaptive Dynamic Programming Decoder & Multi-Model Consensus Engine.

Replaces the legacy, brittle RegressionPostProcessor (is_monotonic_neighbour)
from ByteDance with:
1. Adaptive Prominence Peak Picking (scipy.signal.find_peaks) with sub-frame parabolic interpolation.
2. Frame-Activation Backtracking: Recovers soft inner-chord notes and blunted attacks.
3. Hysteresis Dropout-Proof Note Interval Tracker: Prevents 1-frame demixing notches from truncating notes.
4. Physical Transient Flux Disambiguation: Eliminates sympathetic string resonance overtones (2f0, 3f0).
5. Multi-Model Consensus Fusion: Optionally fuses Spotify Basic Pitch Harmonic CQT deep salience.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import scipy.signal as signal
import pretty_midi
import librosa


class AdaptiveNoteDecoder:
    """
    High-Performance Adaptive Note Decoder for 88-Key Acoustic Piano Transcription.
    """

    def __init__(
        self,
        onset_threshold: float = 0.34,
        frame_threshold: float = 0.15,
        offset_threshold: float = 0.20,
        prominence: float = 0.08,
        min_peak_distance_frames: int = 3,
        max_dropout_frames: int = 3,
        min_duration_sec: float = 0.045,
        max_duration_sec: float = 8.0,
        fps: float = 100.0,
        begin_note: int = 21,
        classes_num: int = 88,
    ):
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.offset_threshold = offset_threshold
        self.prominence = prominence
        self.min_peak_distance_frames = min_peak_distance_frames
        self.max_dropout_frames = max_dropout_frames
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.fps = fps
        self.begin_note = begin_note
        self.classes_num = classes_num

    def decode(
        self,
        output_dict: Dict[str, np.ndarray],
        audio: Optional[np.ndarray] = None,
        sr: int = 16000,
        bp_output: Optional[Dict[str, np.ndarray]] = None,
        enable_overtone_culling: bool = True,
        enable_frame_backtracking: bool = True,
    ) -> List[pretty_midi.Note]:
        """
        Executes end-to-end adaptive decoding from raw model prediction matrices.
        """
        reg_onset = output_dict.get("reg_onset_output")
        frame_out = output_dict.get("frame_output")
        vel_out = output_dict.get("velocity_output")
        reg_offset = output_dict.get("reg_offset_output")

        if reg_onset is None or frame_out is None:
            return []

        frames_num, classes_num = reg_onset.shape
        classes_num = min(classes_num, self.classes_num)

        # -------------------------------------------------------------
        # Step 1: Adaptive Prominence Peak Detection (Sub-Frame Accurate)
        # -------------------------------------------------------------
        onsets_per_key: Dict[int, List[Tuple[float, float, float]]] = {}
        # key: pitch -> list of (onset_sec, peak_prob, frame_idx)

        for k in range(classes_num):
            pitch = k + self.begin_note
            prob_curve = reg_onset[:, k]

            # Scipy prominence peak finder
            peaks, props = signal.find_peaks(
                prob_curve,
                height=self.onset_threshold,
                prominence=self.prominence,
                distance=self.min_peak_distance_frames,
            )

            key_onsets = []
            for p in peaks:
                # Sub-frame parabolic interpolation for microsecond timing
                if 0 < p < frames_num - 1:
                    alpha = prob_curve[p - 1]
                    beta = prob_curve[p]
                    gamma = prob_curve[p + 1]
                    denom = alpha - 2.0 * beta + gamma
                    if abs(denom) > 1e-6:
                        p_sub = 0.5 * (alpha - gamma) / denom
                        p_sub = np.clip(p_sub, -0.5, 0.5)
                    else:
                        p_sub = 0.0
                else:
                    p_sub = 0.0

                onset_sec = (p + p_sub) / self.fps
                peak_prob = float(prob_curve[p])
                key_onsets.append((onset_sec, peak_prob, float(p)))

            onsets_per_key[pitch] = key_onsets

        # -------------------------------------------------------------
        # Step 2: Frame-Activation Backtracking (Recovering Dropped Notes)
        # -------------------------------------------------------------
        if enable_frame_backtracking:
            self._backtrack_missing_onsets(
                frame_out=frame_out,
                onsets_per_key=onsets_per_key,
                classes_num=classes_num,
                frames_num=frames_num,
            )

        # -------------------------------------------------------------
        # Step 3: Optional Basic Pitch Consensus Fusion
        # -------------------------------------------------------------
        if bp_output is not None:
            self._fuse_basic_pitch_consensus(
                bp_output=bp_output,
                reg_onset=reg_onset,
                onsets_per_key=onsets_per_key,
                classes_num=classes_num,
                frames_num=frames_num,
            )

        # -------------------------------------------------------------
        # Step 4: Hysteresis Note Interval Tracking & Duration Solver
        # -------------------------------------------------------------
        raw_notes = self._track_note_intervals(
            onsets_per_key=onsets_per_key,
            frame_out=frame_out,
            vel_out=vel_out,
            reg_offset=reg_offset,
            classes_num=classes_num,
            frames_num=frames_num,
        )

        # -------------------------------------------------------------
        # Step 5: Physical Transient Flux Disambiguation (Overtone Culling)
        # -------------------------------------------------------------
        if enable_overtone_culling and audio is not None and len(raw_notes) > 0:
            verified_notes = self._cull_sympathetic_overtones(
                notes=raw_notes,
                audio=audio,
                sr=sr,
            )
        else:
            verified_notes = raw_notes

        return verified_notes

    def _backtrack_missing_onsets(
        self,
        frame_out: np.ndarray,
        onsets_per_key: Dict[int, List[Tuple[float, float, float]]],
        classes_num: int,
        frames_num: int,
        sustained_frame_thresh: float = 0.42,
        min_run_length: int = 6,
    ) -> None:
        """
        Scans frame probabilities for sustained active regions where the onset head
        completely missed the note (e.g. blunted attack transients in demixed audio).
        """
        for k in range(classes_num):
            pitch = k + self.begin_note
            f_curve = frame_out[:, k]
            existing_frames = {int(round(item[2])) for item in onsets_per_key.get(pitch, [])}

            # Smooth over 1-2 frame demixing notches using median filter
            is_active = f_curve >= sustained_frame_thresh
            is_active_smooth = signal.medfilt(is_active.astype(int), kernel_size=3).astype(bool)

            # Step through runs
            in_run = False
            run_start = 0
            last_run_end = -100

            for i in range(frames_num):
                if is_active_smooth[i] and not in_run:
                    in_run = True
                    run_start = i
                elif not is_active_smooth[i] and in_run:
                    in_run = False
                    run_len = i - run_start
                    gap_from_last = run_start - last_run_end
                    last_run_end = i

                    if run_len >= min_run_length and gap_from_last >= 4:
                        # Check if any existing onset is within this run (-4 to +6 frames)
                        has_onset = any(
                            (run_start - 4) <= ex <= (run_start + 6)
                            for ex in existing_frames
                        )
                        if not has_onset:
                            # Evaluate positive onset slope in the early portion of the run
                            window_end = min(frames_num, run_start + 4)
                            diff = np.diff(f_curve[max(0, run_start - 2):window_end])
                            max_grad = np.max(diff) if len(diff) > 0 else 0.0

                            if max_grad >= 0.06:
                                # Synthesize recovered onset at maximum slope
                                best_offset = int(np.argmax(diff)) if len(diff) > 0 else 0
                                synth_frame = max(0, run_start - 2 + best_offset)
                                synth_sec = synth_frame / self.fps
                                peak_prob = float(f_curve[synth_frame])

                                onsets_per_key[pitch].append((synth_sec, peak_prob, float(synth_frame)))
                                existing_frames.add(synth_frame)

            # Sort onsets chronologically after backtracking
            onsets_per_key[pitch].sort(key=lambda x: x[0])

    def _fuse_basic_pitch_consensus(
        self,
        bp_output: Dict[str, np.ndarray],
        reg_onset: np.ndarray,
        onsets_per_key: Dict[int, List[Tuple[float, float, float]]],
        classes_num: int,
        frames_num: int,
    ) -> None:
        """
        Fuses Spotify Basic Pitch Harmonic CQT activation to cross-validate marginal onsets.
        """
        bp_note = bp_output.get("note")    # (T_bp, 88)
        bp_onset = bp_output.get("onset")  # (T_bp, 88)
        if bp_note is None or bp_onset is None:
            return

        fps_bp = 86.0  # Basic Pitch frame rate
        t_bp = np.arange(len(bp_onset)) / fps_bp
        t_bd = np.arange(frames_num) / self.fps

        # Resample / interpolate Basic Pitch onset to ByteDance timeline
        for k in range(min(classes_num, bp_onset.shape[1])):
            pitch = k + self.begin_note
            bp_on_interp = np.interp(t_bd, t_bp, bp_onset[:, k])
            bp_note_interp = np.interp(t_bd, t_bp, bp_note[:, k])

            existing_frames = {int(round(item[2])) for item in onsets_per_key.get(pitch, [])}

            # Check for marginal ByteDance onsets (0.24 <= p < onset_threshold)
            # that have strong Basic Pitch validation
            peaks, _ = signal.find_peaks(reg_onset[:, k], height=0.24, distance=self.min_peak_distance_frames)
            for p in peaks:
                if reg_onset[p, k] < self.onset_threshold:
                    # Marginal! Check Basic Pitch
                    if bp_on_interp[p] >= 0.35 or bp_note_interp[p] >= 0.45:
                        has_nearby = any(abs(ex - p) <= 3 for ex in existing_frames)
                        if not has_nearby:
                            onset_sec = p / self.fps
                            onsets_per_key[pitch].append((onset_sec, float(reg_onset[p, k]), float(p)))
                            existing_frames.add(p)

            onsets_per_key[pitch].sort(key=lambda x: x[0])

    def _track_note_intervals(
        self,
        onsets_per_key: Dict[int, List[Tuple[float, float, float]]],
        frame_out: np.ndarray,
        vel_out: Optional[np.ndarray],
        reg_offset: Optional[np.ndarray],
        classes_num: int,
        frames_num: int,
    ) -> List[pretty_midi.Note]:
        """
        Traces note durations with hysteresis dropout protection, resolving staccato
        truncation and runaway pedal tails.
        """
        notes: List[pretty_midi.Note] = []

        for k in range(classes_num):
            pitch = k + self.begin_note
            key_onsets = onsets_per_key.get(pitch, [])
            if not key_onsets:
                continue

            f_curve = frame_out[:, k]
            v_curve = vel_out[:, k] if vel_out is not None else None
            off_curve = reg_offset[:, k] if reg_offset is not None else None

            num_onsets = len(key_onsets)

            for idx, (t_start, peak_prob, frame_idx_f) in enumerate(key_onsets):
                f_start = int(round(frame_idx_f))
                next_onset_frame = int(round(key_onsets[idx + 1][2])) if idx + 1 < num_onsets else frames_num

                # Trace forward from f_start to find offset
                f_cur = f_start + 1
                dropout_count = 0
                f_end = None

                while f_cur < next_onset_frame and f_cur < frames_num:
                    prob = f_curve[f_cur]

                    # Check offset head if available
                    if off_curve is not None and off_curve[f_cur] >= self.offset_threshold:
                        f_end = f_cur
                        break

                    # Hysteresis frame check
                    if prob <= self.frame_threshold:
                        dropout_count += 1
                        if dropout_count >= self.max_dropout_frames:
                            # Reached genuine end of sounding note
                            f_end = max(f_start + 2, f_cur - self.max_dropout_frames + 1)
                            break
                    else:
                        dropout_count = 0

                    # Cap maximum duration
                    if (f_cur - f_start) / self.fps >= self.max_duration_sec:
                        f_end = f_cur
                        break

                    f_cur += 1

                if f_end is None:
                    # Truncated by next onset or end of audio
                    f_end = min(next_onset_frame, f_cur)

                t_end = f_end / self.fps

                # Clamp to physical acoustic limits (50ms minimum)
                if (t_end - t_start) < self.min_duration_sec:
                    t_end = t_start + self.min_duration_sec + 0.005

                # Velocity calculation
                if v_curve is not None and 0 <= f_start < frames_num:
                    norm_vel = float(v_curve[f_start])
                else:
                    norm_vel = peak_prob

                # Dynamic velocity scaling
                scaled_vel = int(np.clip(30 + 82 * (norm_vel ** 0.85), 20, 127))

                notes.append(
                    pretty_midi.Note(
                        velocity=scaled_vel,
                        pitch=pitch,
                        start=float(t_start),
                        end=float(t_end),
                    )
                )

        notes.sort(key=lambda n: (n.start, n.pitch))
        return notes

    def _cull_sympathetic_overtones(
        self,
        notes: List[pretty_midi.Note],
        audio: np.ndarray,
        sr: int = 16000,
    ) -> List[pretty_midi.Note]:
        """
        Disambiguates genuine physical finger strikes from sympathetic string resonance
        overtones (2f0, 3f0, 4f0) using attack transient flux analysis (Delta E).
        """
        if not notes:
            return notes

        n_fft = 2048
        hop = 320
        stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop))
        stft_db = librosa.amplitude_to_db(stft, ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        num_frames = stft_db.shape[1]

        verified: List[pretty_midi.Note] = []

        for cand in notes:
            p = cand.pitch
            t_s = cand.start

            is_overtone = False

            # Check if an active lower fundamental exists at octave (12), octave+fifth (19), or double octave (24)
            for other in notes:
                if other is cand:
                    continue

                diff = p - other.pitch
                if diff in (12, 19, 24):
                    # Co-occurring strike within 35ms
                    if abs(t_s - other.start) <= 0.035:
                        f0 = 440.0 * 2.0 ** ((p - 69) / 12.0)
                        bin_idx = int(np.argmin(np.abs(freqs - f0)))
                        frame_idx = int(round(t_s * sr / hop))
                        frame_idx = min(num_frames - 1, max(0, frame_idx))

                        pre_idx = max(0, frame_idx - 2)
                        post_idx = min(num_frames - 1, frame_idx + 1)

                        delta_e = float(stft_db[bin_idx, post_idx] - stft_db[bin_idx, pre_idx])

                        # Genuine physical hammer strikes create a sharp transient jump Delta E >= 3.0 dB
                        # Sympathetic resonance is smooth and exhibits Delta E < 3.0 dB
                        if delta_e < 3.0 and cand.velocity <= 68:
                            is_overtone = True
                            # Merge duration into the parent lower note
                            other.end = max(other.end, cand.end)
                            break

            if not is_overtone:
                verified.append(cand)

        return verified
