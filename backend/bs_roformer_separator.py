"""
BS-RoFormer Multi-Stem Source Demixer (6-Source: bass, drums, other, vocals, guitar, piano)

Uses the Band-Split RoFormer (BS-RoFormer) architecture with Band-Split Rotary Position
Embedding Attention across time and frequency. Outperforms HTDemucs on acoustic piano and
guitar isolation by avoiding time-domain comb-filtering and phase-cancellation smearing.
"""

import time
import yaml
import inspect
from pathlib import Path
from typing import Dict, Optional, Callable, Any
import numpy as np
import soundfile as sf
import torch
import librosa

from backend.config import DEVICE, BASE_DIR

_bs_roformer_model_instance = None
CONFIG_PATH = BASE_DIR / "storage" / "models" / "BS-Rofo-SW-Fixed.yaml"
CHECKPOINT_PATH = BASE_DIR / "storage" / "models" / "BS-Rofo-SW-Fixed.ckpt"


def get_bs_roformer_model():
    """Lazy loader for BS-RoFormer 6-Source model on GPU."""
    global _bs_roformer_model_instance
    if _bs_roformer_model_instance is None:
        from bs_roformer import BSRoformer

        if not CONFIG_PATH.exists() or not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"BS-RoFormer model files missing in {BASE_DIR / 'storage' / 'models'}")

        print(f"[BS-RoFormer] Loading configuration from {CONFIG_PATH.name}...")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        m_cfg = cfg["model"]
        sig = inspect.signature(BSRoformer.__init__)
        valid_keys = set(sig.parameters.keys())
        kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in m_cfg.items() if k in valid_keys}
        kwargs["flash_attn"] = False

        print(f"[BS-RoFormer] Initializing neural model on {DEVICE}...")
        model = BSRoformer(**kwargs).to(DEVICE)

        print(f"[BS-RoFormer] Loading checkpoint weights from {CHECKPOINT_PATH.name}...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt, strict=True)
        model.eval()

        _bs_roformer_model_instance = model
        print("[BS-RoFormer] Model loaded successfully with 100% strict weight matching.")

    return _bs_roformer_model_instance


def unload_bs_roformer_model():
    """Unloads BS-RoFormer from GPU VRAM to ensure headroom for transcriber."""
    global _bs_roformer_model_instance
    if _bs_roformer_model_instance is not None:
        del _bs_roformer_model_instance
        _bs_roformer_model_instance = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[BS-RoFormer] Model unloaded from GPU VRAM.")


def separate_bs_roformer_audio(
    input_audio_path: Path,
    output_piano_path: Path,
    output_vocal_path: Optional[Path] = None,
    output_drums_path: Optional[Path] = None,
    output_bass_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    """
    Separates mixed audio into 6 stems using BS-RoFormer:
      ['bass', 'drums', 'other', 'vocals', 'guitar', 'piano']
    Applies overlap-add chunked inference with Hann window crossfading.
    """
    t0 = time.time()

    def notify(pct: int, msg: str):
        print(f"[BS-RoFormer] ({pct}%) {msg}", flush=True)
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

    notify(10, "Loading audio for BS-RoFormer separation (44.1kHz)...")
    audio, sr = sf.read(str(input_audio_path))
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    if sr != 44100:
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=44100).T

    total_samples = len(audio)
    duration_s = total_samples / 44100.0
    notify(20, f"Loaded {duration_s:.1f}s of audio. Initializing model on {DEVICE}...")

    model = get_bs_roformer_model()

    # Overlap-add parameters
    # Chunk size: 281600 samples (~6.38s), step: 140800 samples (50% overlap)
    # Optimized for 4GB VRAM GPUs (RTX 3050 Laptop) to prevent CUDA OOM
    chunk_size = 281600
    step = chunk_size // 2
    window = np.hanning(chunk_size)

    # Pad audio to cleanly cover all samples with step-sized overlap
    pad_left = step
    needed_len = total_samples + pad_left
    if needed_len < chunk_size:
        pad_right = chunk_size - needed_len
    else:
        rem = (needed_len - chunk_size) % step
        pad_right = (step - rem) % step
    audio_padded = np.pad(audio, ((pad_left, pad_right), (0, 0)), mode="constant")
    padded_len = len(audio_padded)

    out_stems = np.zeros((6, 2, padded_len), dtype=np.float32)
    weight_sum = np.zeros(padded_len, dtype=np.float32)

    n_chunks = (padded_len - chunk_size) // step + 1
    notify(25, f"Separating 6 audio stems across {n_chunks} chunks on GPU...")

    with torch.no_grad():
        for i in range(n_chunks):
            start = i * step
            end = start + chunk_size
            chunk = audio_padded[start:end].T  # (2, chunk_size)
            tensor = torch.tensor(chunk, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            with torch.amp.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu"):
                # Output: (1, 6, 2, chunk_size)
                recon = model(tensor)[0].cpu().numpy()

            for s in range(6):
                out_stems[s, :, start:end] += recon[s] * window[np.newaxis, :]
            weight_sum[start:end] += window

            pct = 25 + int(45 * (i + 1) / n_chunks)
            notify(pct, f"Separating audio... chunk {i + 1}/{n_chunks} complete.")

    # Normalize by accumulated window weights
    valid_w = weight_sum > 1e-4
    for s in range(6):
        out_stems[s, 0, valid_w] /= weight_sum[valid_w]
        out_stems[s, 1, valid_w] /= weight_sum[valid_w]

    notify(72, "Extracting and exporting isolated audio stems...")
    output_piano_path.parent.mkdir(parents=True, exist_ok=True)

    # Instruments order: ['bass', 'drums', 'other', 'vocals', 'guitar', 'piano']
    piano_audio = out_stems[5, :, pad_left:pad_left + total_samples].T
    drums_audio = out_stems[1, :, pad_left:pad_left + total_samples].T
    bass_audio  = out_stems[0, :, pad_left:pad_left + total_samples].T
    vocal_audio = out_stems[3, :, pad_left:pad_left + total_samples].T

    piano_rms = float((piano_audio ** 2).mean() ** 0.5)
    drums_rms = float((drums_audio ** 2).mean() ** 0.5)
    bass_rms  = float((bass_audio ** 2).mean() ** 0.5)
    vocal_rms = float((vocal_audio ** 2).mean() ** 0.5)

    # Save outputs as 32-bit float to prevent PCM_16 clipping harmonics
    sf.write(str(output_piano_path), piano_audio, 44100, subtype="FLOAT")
    if output_vocal_path:
        output_vocal_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_vocal_path), vocal_audio, 44100, subtype="FLOAT")
    if output_drums_path:
        output_drums_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_drums_path), drums_audio, 44100, subtype="FLOAT")
    if output_bass_path:
        output_bass_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_bass_path), bass_audio, 44100, subtype="FLOAT")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = round(time.time() - t0, 2)
    notify(75, f"BS-RoFormer complete in {elapsed}s. Piano RMS: {piano_rms:.4f}")

    return {
        "piano_path": str(output_piano_path),
        "piano_rms": piano_rms,
        "drums_rms": drums_rms,
        "bass_rms": bass_rms,
        "vocals_rms": vocal_rms,
        "elapsed_seconds": elapsed
    }
