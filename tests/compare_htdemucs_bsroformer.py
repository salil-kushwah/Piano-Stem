import sys
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
import soundfile as sf
import librosa
import pretty_midi
from piano_transcription_inference import PianoTranscription

out_dir = Path("storage/test_lucifer")
ht_piano_p = out_dir / "htdemucs_6s_piano.wav"
bs_piano_p = out_dir / "bs_roformer_piano.wav"
clip_p = out_dir / "clip_45s.wav"

def to_mono_16k(audio, sr):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio

def run_comparison():
    print("=" * 75)
    print("COMPARISON: HTDEMUCS 6S vs BS-ROFORMER-SW (45s Lucifer Wicked Game Clip)")
    print("=" * 75)

    # 1. Audio Stem Properties
    ht_audio, sr_ht = sf.read(str(ht_piano_p))
    bs_audio, sr_bs = sf.read(str(bs_piano_p))
    orig_audio, sr_orig = sf.read(str(clip_p))

    ht_rms = float((ht_audio ** 2).mean() ** 0.5)
    bs_rms = float((bs_audio ** 2).mean() ** 0.5)
    orig_rms = float((orig_audio ** 2).mean() ** 0.5)

    print(f"\n1. STEM AUDIO ENERGY (RMS):")
    print(f"   Original Mix:     {orig_rms:.4f}")
    print(f"   HTDemucs 6S:      {ht_rms:.4f} ({ht_rms / orig_rms:.1%} of mix energy)")
    print(f"   BS-RoFormer-SW:   {bs_rms:.4f} ({bs_rms / orig_rms:.1%} of mix energy)")

    # 2. Resample to 16kHz for ByteDance CRNN
    ht_16k = to_mono_16k(ht_audio, sr_ht)
    bs_16k = to_mono_16k(bs_audio, sr_bs)
    orig_16k = to_mono_16k(orig_audio, sr_orig)

    # 3. Transcription with ByteDance CRNN
    print("\n2. TRANSCRIBING SEPARATED PIANO STEMS WITH BYTEDANCE CRNN (16kHz)...")
    pt = PianoTranscription(device="cuda")
    pt.onset_threshold = 0.45
    pt.frame_threshold = 0.18
    pt.offset_threshod = 0.28

    ht_mid_p = out_dir / "compare_htdemucs_crnn.mid"
    bs_mid_p = out_dir / "compare_bsroformer_crnn.mid"
    raw_mid_p = out_dir / "compare_rawmix_crnn.mid"

    pt.transcribe(ht_16k, str(ht_mid_p))
    pt.transcribe(bs_16k, str(bs_mid_p))
    pt.transcribe(orig_16k, str(raw_mid_p))

    ht_pm = pretty_midi.PrettyMIDI(str(ht_mid_p))
    bs_pm = pretty_midi.PrettyMIDI(str(bs_mid_p))
    raw_pm = pretty_midi.PrettyMIDI(str(raw_mid_p))

    ht_notes = ht_pm.instruments[0].notes if ht_pm.instruments else []
    bs_notes = bs_pm.instruments[0].notes if bs_pm.instruments else []
    raw_notes = raw_pm.instruments[0].notes if raw_pm.instruments else []

    print(f"\n3. TRANSCRIPTION NOTE COUNTS (45s Clip):")
    print(f"   Raw Mixed Audio (No demixing): {len(raw_notes)} notes")
    print(f"   HTDemucs 6S Piano Stem:        {len(ht_notes)} notes")
    print(f"   BS-RoFormer-SW Piano Stem:     {len(bs_notes)} notes")

    # Analyze notes during vocal entry (t = 16s to 28s) where singer holds prominent notes
    ht_vocal_section = [n for n in ht_notes if 16.0 <= n.start <= 28.0]
    bs_vocal_section = [n for n in bs_notes if 16.0 <= n.start <= 28.0]
    raw_vocal_section = [n for n in raw_notes if 16.0 <= n.start <= 28.0]

    # Analyze treble notes >= 80 (common hallucination band from cymbals / sibilance)
    ht_treble_bleed = [n for n in ht_notes if n.pitch >= 80]
    bs_treble_bleed = [n for n in bs_notes if n.pitch >= 80]
    raw_treble_bleed = [n for n in raw_notes if n.pitch >= 80]

    # Analyze sub-bass notes < 30 (common hallucination band from kick / 808s)
    ht_sub_bass = [n for n in ht_notes if n.pitch < 30]
    bs_sub_bass = [n for n in bs_notes if n.pitch < 30]
    raw_sub_bass = [n for n in raw_notes if n.pitch < 30]

    # Average note velocities
    ht_avg_vel = np.mean([n.velocity for n in ht_notes]) if ht_notes else 0
    bs_avg_vel = np.mean([n.velocity for n in bs_notes]) if bs_notes else 0
    raw_avg_vel = np.mean([n.velocity for n in raw_notes]) if raw_notes else 0

    print(f"\n4. BLEED & ARTIFACT METRICS:")
    print(f"   Notes in vocal solo section (16-28s): Raw={len(raw_vocal_section)} | HTDemucs={len(ht_vocal_section)} | BS-RoFormer={len(bs_vocal_section)}")
    print(f"   High treble notes (>= MIDI 80):       Raw={len(raw_treble_bleed)} | HTDemucs={len(ht_treble_bleed)} | BS-RoFormer={len(bs_treble_bleed)}")
    print(f"   Sub-bass notes (< MIDI 30):           Raw={len(raw_sub_bass)} | HTDemucs={len(ht_sub_bass)} | BS-RoFormer={len(bs_sub_bass)}")
    print(f"   Average note velocity:                Raw={raw_avg_vel:.1f} | HTDemucs={ht_avg_vel:.1f} | BS-RoFormer={bs_avg_vel:.1f}")
    print("=" * 75)

if __name__ == "__main__":
    run_comparison()
