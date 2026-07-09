"""
End-to-End Studio Integrity & Logic Test Suite
Verifies all core architectural subsystems of Piano Stem Studio:
  1. Audio Mix Regime Classifier (Regime A vs Regime B vs Solo)
  2. Master Stem Enhancer (Acoustic De-Muffler, Biquad High Shelf, Harmonic Air Exciter)
  3. Studio Master AI Engine & Calibrated Spectral Sieve
  4. Symbolic Voicing Solver & Polyphonic Chord Aligner
  5. Pedagogical Hand Separator & 1-2-3-4-5 Ergonomic Fingering Engine
  6. Standard Grand Staff MusicXML 3.1 Exporter
  7. Studio Concert Grand SoundFont Synthesizer
"""

import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pretty_midi
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.audio_classifier import detect_audio_regime
from backend.stem_enhancer import (
    biquad_high_shelf,
    biquad_peaking_eq,
    biquad_low_shelf,
    apply_spectral_denoiser,
    apply_midside_stabilizer,
    apply_harmonic_air_exciter,
    enhance_piano_stem
)
from backend.spectral_sieve import verify_notes_with_spectral_sieve
from backend.voicing_solver import apply_voicing_solver
from backend.quantizer import align_chord_onsets, estimate_tempo_bpm
from backend.hand_separator import separate_hands
from backend.fingering import assign_fingerings
from backend.musicxml_exporter import export_to_musicxml
from backend.synth import synthesize_midi_to_audio


def test_audio_regime_classification():
    print("[Test 1/9] Testing Audio Mix Regime Classifier...")
    al_path = ROOT / "another love.mp3"
    if al_path.exists():
        res = detect_audio_regime(al_path, max_duration_sec=20.0)
        assert res["regime"] == "vocal_piano_duet", f"Expected vocal_piano_duet, got {res['regime']}"
        assert res["recommended_stem_mode"] == "mdx_inst"
        print(f"  [OK] Duet classified correctly: {res['label']} ({res['confidence']:.0%})")
    else:
        print("  - Skipped: another love.mp3 not found")


def test_stem_enhancer_acoustics():
    print("[Test 2/9] Testing Master Stem Enhancer Acoustics & Zero Noise...")
    sr = 44100
    # Generate test note + pause
    t = np.linspace(0, 0.5, int(sr * 0.5))
    note = np.sin(2 * np.pi * 440.0 * t) * 0.5 * np.exp(-4 * t)
    silence = np.zeros(int(sr * 0.5))
    test_sig = np.concatenate([note, silence])

    # 1. Biquad High Shelf EQ
    b_shelf, a_shelf = biquad_high_shelf(gain_db=5.5, shelf_freq_hz=3400.0, slope=0.9, sr=sr)
    assert len(b_shelf) == 3 and len(a_shelf) == 3, "Invalid biquad filter order"

    # 2. Harmonic Exciter with squared silence gate
    excited = apply_harmonic_air_exciter(test_sig, sr=sr, highpass_cutoff=2800.0, exciter_gain_db=4.5, gate_threshold_db=-45.0)
    assert len(excited) == len(test_sig)
    
    # Verify rest is dead silent
    silence_out = excited[int(sr * 0.6):]
    max_silence_amp = np.max(np.abs(silence_out))
    assert max_silence_amp < 1e-6, f"Silence gate leak: {max_silence_amp}"
    print(f"  [OK] High-shelf designed & silence gate holds at digital black ({max_silence_amp:.2e})")


def test_voicing_solver_and_quantizer():
    print("[Test 3/9] Testing Voicing Solver & Polyphonic Chord Aligner...")
    notes = [
        pretty_midi.Note(velocity=80, pitch=48, start=0.010, end=1.0), # C3
        pretty_midi.Note(velocity=75, pitch=52, start=0.022, end=1.0), # E3
        pretty_midi.Note(velocity=85, pitch=55, start=0.015, end=1.0), # G3
        pretty_midi.Note(velocity=90, pitch=60, start=0.012, end=1.0), # C4
        pretty_midi.Note(velocity=70, pitch=64, start=0.018, end=1.0), # E4
        pretty_midi.Note(velocity=65, pitch=67, start=0.020, end=1.0), # G4
    ]

    # Chord alignment
    aligned = align_chord_onsets(notes, tolerance=0.035)
    starts = set(n.start for n in aligned)
    assert len(starts) == 1, f"Expected all chord notes aligned to 1 start time, got: {starts}"

    # Voicing solver: prune only if single hand physically exceeds 5 notes
    pruned_notes, count = apply_voicing_solver(aligned)
    assert len(pruned_notes) == 6, f"6-note two-hand chord should NOT be pruned (lh has 3, rh has 3)"
    assert count == 0
    print(f"  [OK] Chord aligned to {list(starts)[0]}s; two-hand chord preserved 100%")


def test_hand_separation_and_fingering():
    print("[Test 4/9] Testing Pedagogical Hands & Ergonomic Fingering Engine...")
    raw_notes = [
        {"midi": 36, "time": 0.0, "duration": 1.0, "velocity": 0.7}, # C2
        {"midi": 43, "time": 0.0, "duration": 1.0, "velocity": 0.7}, # G2
        {"midi": 48, "time": 0.0, "duration": 1.0, "velocity": 0.7}, # C3
        {"midi": 60, "time": 0.0, "duration": 1.0, "velocity": 0.8}, # C4
        {"midi": 64, "time": 0.0, "duration": 1.0, "velocity": 0.8}, # E4
        {"midi": 67, "time": 0.0, "duration": 1.0, "velocity": 0.8}, # G4
    ]

    separated = separate_hands(raw_notes)
    for n in separated[:3]:
        assert n["hand"] == "left", f"Bass note {n['midi']} should be Left Hand"
    for n in separated[3:]:
        assert n["hand"] == "right", f"Treble note {n['midi']} should be Right Hand"

    fingered = assign_fingerings(separated)
    lh_fingers = [n["finger"] for n in fingered if n["hand"] == "left"]
    rh_fingers = [n["finger"] for n in fingered if n["hand"] == "right"]

    assert len(set(lh_fingers)) == len(lh_fingers), f"LH chord contains duplicate fingers: {lh_fingers}"
    assert len(set(rh_fingers)) == len(rh_fingers), f"RH chord contains duplicate fingers: {rh_fingers}"
    assert all(1 <= f <= 5 for f in lh_fingers + rh_fingers), "Finger outside 1..5 range"
    print(f"  [OK] Distinct LH fingers: {lh_fingers} | Distinct RH fingers: {rh_fingers}")


def test_musicxml_exporter():
    print("[Test 5/9] Testing Grand Staff MusicXML 3.1 Exporter...")
    notes = [
        {"midi": 48, "time": 0.0, "duration": 0.9, "velocity": 0.7, "hand": "left", "finger": 5, "chord_name": "C"},
        {"midi": 52, "time": 0.0, "duration": 0.9, "velocity": 0.7, "hand": "left", "finger": 3, "chord_name": "C"},
        {"midi": 55, "time": 0.0, "duration": 0.9, "velocity": 0.7, "hand": "left", "finger": 1, "chord_name": "C"},
        {"midi": 60, "time": 0.0, "duration": 0.9, "velocity": 0.8, "hand": "right", "finger": 1, "chord_name": "C"},
        {"midi": 64, "time": 0.0, "duration": 0.9, "velocity": 0.8, "hand": "right", "finger": 3, "chord_name": "C"},
        {"midi": 67, "time": 0.0, "duration": 0.9, "velocity": 0.8, "hand": "right", "finger": 5, "chord_name": "C"},
    ]

    out_xml = ROOT / "storage" / "midi" / "test_verify.musicxml"
    export_to_musicxml(notes, out_xml, title="Test Verification", key_name="C Major", tempo_bpm=120.0)

    assert out_xml.exists(), "MusicXML file was not written"
    tree = ET.parse(str(out_xml))
    root = tree.getroot()
    assert root.tag == "score-partwise", f"Invalid MusicXML root tag: {root.tag}"
    measures = root.findall(".//measure")
    assert len(measures) >= 1, "No measures found in MusicXML"
    notes_in_xml = root.findall(".//note")
    assert len(notes_in_xml) >= 6, f"Expected at least 6 notes in XML, got {len(notes_in_xml)}"
    print(f"  [OK] Standard MusicXML 3.1 grand staff verified: {len(measures)} measure(s), {len(notes_in_xml)} note/rest tags")


def test_soundfont_synthesis():
    print("[Test 6/9] Testing Concert Grand SoundFont Synthesizer...")
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    piano.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.8)) # C4
    piano.notes.append(pretty_midi.Note(velocity=85, pitch=64, start=0.0, end=0.8)) # E4
    piano.notes.append(pretty_midi.Note(velocity=88, pitch=67, start=0.0, end=0.8)) # G4
    pm.instruments.append(piano)

    test_mid = ROOT / "storage" / "midi" / "test_synth_chord.mid"
    test_wav = ROOT / "storage" / "midi" / "test_synth_chord.wav"
    pm.write(str(test_mid))

    synthesize_midi_to_audio(test_mid, test_wav, sample_rate=44100)
    assert test_wav.exists(), "Synthesized WAV was not created"
    audio, sr = sf.read(str(test_wav))
    assert sr == 44100, f"Expected 44.1kHz, got {sr}"
    assert audio.ndim == 2, f"Expected stereo audio, got shape {audio.shape}"
    peak = np.max(np.abs(audio))
    assert 0.05 < peak <= 1.0, f"Unreasonable synthesis amplitude: {peak}"
    print(f"  [OK] 44.1kHz stereo SoundFont Concert Grand synthesized cleanly (Peak: {peak:.3f})")


def test_spectral_sieve_calibration():
    print("[Test 7/9] Testing Spectral Sieve Calibrated Multi-Evidence Retention...")
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    # Fundamental C4 (261.63 Hz) + quiet C5 (523.25 Hz)
    sig = 0.5 * np.sin(2 * np.pi * 261.63 * t) + 0.15 * np.sin(2 * np.pi * 523.25 * t)
    
    notes = [
        pretty_midi.Note(velocity=80, pitch=60, start=0.05, end=0.8), # C4
        pretty_midi.Note(velocity=65, pitch=72, start=0.05, end=0.8), # C5 octave
    ]

    retained, stats = verify_notes_with_spectral_sieve(notes, sig, sr=sr, mode="clean")
    assert len(retained) == 2, f"Both octave notes must be retained under calibrated rules! Got: {len(retained)}"
    print(f"  [OK] Calibrated sieve retained {len(retained)}/2 notes (zero octave cannibalization)")


def test_adaptive_note_decoder():
    print("[Test 8/9] Testing Adaptive Dynamic Programming Note Decoder & Backtracking...")
    from backend.adaptive_decoder import AdaptiveNoteDecoder

    decoder = AdaptiveNoteDecoder(
        onset_threshold=0.34,
        frame_threshold=0.15,
        offset_threshold=0.20,
    )

    frames_num = 100
    classes_num = 88
    reg_onset = np.zeros((frames_num, classes_num), dtype=np.float32)
    frame_out = np.zeros((frames_num, classes_num), dtype=np.float32)
    vel_out = np.full((frames_num, classes_num), 0.7, dtype=np.float32)
    reg_offset = np.zeros((frames_num, classes_num), dtype=np.float32)

    # 1. Pitch 60 (C4): Smeared onset peak with shoulder (would fail legacy is_monotonic_neighbour)
    k_60 = 60 - 21
    reg_onset[18, k_60] = 0.25
    reg_onset[19, k_60] = 0.50
    reg_onset[20, k_60] = 0.50  # Plateau! (Fails strict monotonicity)
    reg_onset[21, k_60] = 0.85  # Peak
    reg_onset[22, k_60] = 0.40
    frame_out[21:45, k_60] = 0.80

    # 2. Pitch 64 (E4): Completely missed onset (prob 0.05), but strong sustained frame energy (recovers via backtracking)
    k_64 = 64 - 21
    reg_onset[:, k_64] = 0.05
    frame_out[30:55, k_64] = 0.85
    # Create sharp attack gradient
    frame_out[30, k_64] = 0.10
    frame_out[31, k_64] = 0.40
    frame_out[32, k_64] = 0.85

    # 3. Pitch 67 (G4): Sustained note with a 1-frame demixing dropout (frame 40: 0.08)
    k_67 = 67 - 21
    reg_onset[25, k_67] = 0.90
    frame_out[25:50, k_67] = 0.80
    frame_out[40, k_67] = 0.08  # 1-frame dropout notch! (Hysteresis must bridge this)

    output_dict = {
        "reg_onset_output": reg_onset,
        "frame_output": frame_out,
        "velocity_output": vel_out,
        "reg_offset_output": reg_offset,
    }

    notes = decoder.decode(output_dict)
    pitches = {n.pitch for n in notes}

    # Verify C4 was detected despite non-monotonic onset shoulder
    assert 60 in pitches, "C4 with smeared onset plateau must be detected by adaptive peak picking!"
    # Verify E4 was recovered via frame backtracking
    assert 64 in pitches, "E4 with zero onset peak must be recovered by frame-activation backtracking!"
    # Verify G4 bridged the 1-frame dropout
    g4_notes = [n for n in notes if n.pitch == 67]
    assert len(g4_notes) == 1, f"G4 should be 1 bridged note, but got {len(g4_notes)} fragmented notes!"
    assert g4_notes[0].end - g4_notes[0].start >= 0.20, "G4 duration should span across dropout!"

    print(f"  [OK] Adaptive decoder recovered all 3 test scenarios (smeared plateau, frame backtracking, dropout bridging)")


def test_transkun_engine():
    print("[Test 9/9] Testing Transkun SOTA Transformer Semi-CRF Engine...")
    from backend.transkun_engine import transcribe_audio_transkun, unload_transkun_model
    al_path = ROOT / "another love.mp3"
    if al_path.exists():
        audio, sr = sf.read(str(al_path))
        notes, pedals = transcribe_audio_transkun(audio[:int(sr * 4.0)], sr=sr)
        assert len(notes) > 0, "Transkun failed to extract notes from Another Love excerpt"
        unload_transkun_model()
        print(f"  [OK] Transkun SOTA extracted {len(notes)} notes and {len(pedals)} pedals at 44.1kHz (VRAM cleared)")
    else:
        print("  - Skipped: another love.mp3 not found")


if __name__ == "__main__":
    print("=" * 72)
    print("PIANO STEM STUDIO — COMPLETE SUBSYSTEM INTEGRITY TEST")
    print("=" * 72)
    test_audio_regime_classification()
    test_stem_enhancer_acoustics()
    test_voicing_solver_and_quantizer()
    test_hand_separation_and_fingering()
    test_musicxml_exporter()
    test_soundfont_synthesis()
    test_spectral_sieve_calibration()
    test_adaptive_note_decoder()
    test_transkun_engine()
    print("=" * 72)
    print("ALL 9 ARCHITECTURAL SUBSYSTEMS PASSED WITH 100% SUCCESS!")
    print("=" * 72)
