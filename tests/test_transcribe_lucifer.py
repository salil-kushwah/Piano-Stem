import pretty_midi
import soundfile as sf
import librosa
from pathlib import Path
from piano_transcription_inference import PianoTranscription

transcriptor = PianoTranscription(device="cuda")
transcriptor.onset_threshold = 0.60
transcriptor.frame_threshold = 0.25
transcriptor.offset_threshod = 0.35

out_dir = Path("storage/test_lucifer")

for name, wav_file in [
    ("1. Direct (With Vocals)", out_dir / "clip_45s.wav"),
    ("2. Demucs 4S ('other' stem)", out_dir / "htdemucs_other.wav"),
    ("3. Demucs 6S ('piano' stem)", out_dir / "htdemucs_6s_piano.wav")
]:
    audio, sr = sf.read(str(wav_file))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    mid_path = out_dir / f"{wav_file.stem}.mid"
    transcriptor.transcribe(audio, str(mid_path))
    
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    notes = pm.instruments[0].notes
    print(f"=== {name} ===")
    print(f"  Notes detected: {len(notes)}")
    pitches = [n.pitch for n in notes]
    print(f"  Pitch range: {min(pitches)} ({pretty_midi.note_number_to_name(min(pitches))}) to {max(pitches)} ({pretty_midi.note_number_to_name(max(pitches))})")
    
    # Check notes occurring during vocal section (from ~12s onwards)
    vocal_sec_notes = [n for n in notes if n.start >= 12.0]
    print(f"  Notes in vocal section (>=12s): {len(vocal_sec_notes)}")
