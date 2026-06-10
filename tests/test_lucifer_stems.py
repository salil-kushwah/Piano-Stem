import time
import torch
import soundfile as sf
import librosa
from pathlib import Path
from demucs.api import Separator

audio_path = r"C:\Users\salil\Downloads\Lucifer_ Bloody Celestial Karaoke Jam Soundtrack _ Wicked Game  - Tom Ellis _ WaterTower.mp3"
out_dir = Path("storage/test_lucifer")
out_dir.mkdir(parents=True, exist_ok=True)

# Test first 45 seconds (contains solo piano intro + vocal entry + piano accompaniment)
print("Loading 45s clip of Lucifer Wicked Game...")
audio, sr = librosa.load(audio_path, sr=44100, mono=False, duration=45.0)
clip_path = out_dir / "clip_45s.wav"
sf.write(str(clip_path), audio.T, sr)

print("1. Running Demucs 4-Stem (htdemucs)...")
t0 = time.time()
sep4 = Separator(model="htdemucs", device="cuda" if torch.cuda.is_available() else "cpu", segment=7, shifts=1)
origin4, separated4 = sep4.separate_audio_file(str(clip_path))

import torchaudio
torchaudio.save(str(out_dir / "htdemucs_other.wav"), separated4["other"].cpu(), sep4.samplerate)
torchaudio.save(str(out_dir / "htdemucs_vocals.wav"), separated4["vocals"].cpu(), sep4.samplerate)
print(f"htdemucs finished in {time.time()-t0:.2f}s")

print("2. Running Demucs 6-Stem (htdemucs_6s)...")
t0 = time.time()
sep6 = Separator(model="htdemucs_6s", device="cuda" if torch.cuda.is_available() else "cpu", segment=7, shifts=1)
origin6, separated6 = sep6.separate_audio_file(str(clip_path))
torchaudio.save(str(out_dir / "htdemucs_6s_piano.wav"), separated6["piano"].cpu(), sep6.samplerate)
torchaudio.save(str(out_dir / "htdemucs_6s_vocals.wav"), separated6["vocals"].cpu(), sep6.samplerate)
print(f"htdemucs_6s finished in {time.time()-t0:.2f}s")

print("Stem separation complete. Files saved in storage/test_lucifer/")
