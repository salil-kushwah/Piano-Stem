"""
SOTA Mixed Audio Separator — UVR MDX-Net Engine (Inst_HQ_3)

Replaces legacy Demucs with the industry-standard UVR MDX-Net architecture:
  - Operates directly in the complex STFT domain (no time-domain comb filtering).
  - Isolates clean instrumentals (piano, keys, acoustic instruments) with zero phase gargling.
  - Generates pristine isolated vocal stems for cross-stem bleed rejection.
"""

from pathlib import Path
from typing import Tuple, Optional, Callable, Any
import numpy as np
import soundfile as sf
import torch
import librosa
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parent.parent / "storage" / "models" / "UVR-MDX-NET-Inst_HQ_3.onnx"
MODEL_URL = "https://huggingface.co/seanghay/uvr_models/resolve/main/UVR-MDX-NET-Inst_HQ_3.onnx"


class ConvTDFNet:
    """Frequency-domain STFT/iSTFT processor for MDX-Net architecture."""
    def __init__(self, target_name: str = "vocals", L: int = 11, dim_f: int = 3072, dim_t: int = 8, n_fft: int = 6144, hop: int = 1024):
        self.dim_c = 4
        self.dim_f = dim_f
        self.dim_t = 2 ** dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.target_name = target_name
        out_c = self.dim_c * 4 if target_name == "*" else self.dim_c
        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t])
        self.n = L // 2

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, self.dim_c, self.n_bins, self.dim_t])
        return x[:, :, : self.dim_f]

    def istft(self, x: torch.Tensor, freq_pad: Optional[torch.Tensor] = None) -> torch.Tensor:
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        c = 4 * 2 if self.target_name == "*" else 2
        x = x.reshape([-1, c, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1]).contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1, c, self.chunk_size])


_session_instance = None


def get_mdx_session() -> ort.InferenceSession:
    global _session_instance
    if _session_instance is None:
        if not MODEL_PATH.exists():
            import urllib.request
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            print(f"[MDX-Net] Downloading SOTA separation model from {MODEL_URL}...")
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))

        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")
        print(f"[MDX-Net] Initializing ONNX session with providers {providers}...")
        _session_instance = ort.InferenceSession(str(MODEL_PATH), providers=providers)
    return _session_instance


def unload_mdx_session():
    """Releases the ONNX Runtime session and frees GPU VRAM."""
    global _session_instance
    if _session_instance is not None:
        del _session_instance
        _session_instance = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[MDX-Net] Session unloaded from GPU VRAM.")



def separate_mixed_audio_mdx(
    input_audio_path: Path,
    output_instrumental_path: Path,
    output_vocal_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[Path, Optional[Path]]:
    """
    Separates a mixed audio track into clean instrumentals (piano/keys) and vocals.
    Returns (instrumental_path, vocal_path).
    """
    def notify(pct: int, msg: str):
        print(f"[MDX Separator] ({pct}%) {msg}")
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

    notify(10, "Loading audio for SOTA MDX-Net separation...")
    audio, sr = librosa.load(str(input_audio_path), sr=44100, mono=False)
    if audio.ndim == 1:
        audio = np.asfortranarray([audio, audio])

    notify(20, "Initializing UVR MDX-Net Inst_HQ_3 Engine...")
    session = get_mdx_session()
    model = ConvTDFNet(target_name="vocals", dim_f=3072, dim_t=8, n_fft=6144, hop=1024)

    n_sample = audio.shape[1]
    trim = model.n_fft // 2
    gen_size = model.chunk_size - 2 * trim
    pad = gen_size - (n_sample % gen_size)
    mix_p = np.concatenate((np.zeros((2, trim)), audio, np.zeros((2, pad)), np.zeros((2, trim))), axis=1)

    mix_waves = []
    i = 0
    while i < n_sample + pad:
        waves = np.array(mix_p[:, i : i + model.chunk_size])
        mix_waves.append(waves)
        i += gen_size

    total_chunks = len(mix_waves)
    notify(25, f"Running neural source demixing on {total_chunks} audio segments...")

    # Process chunks in mini-batches to optimize throughput
    batch_size = 4
    tar_signals = []

    for b_start in range(0, total_chunks, batch_size):
        b_end = min(b_start + batch_size, total_chunks)
        batch = torch.tensor(np.array(mix_waves[b_start:b_end]), dtype=torch.float32)

        with torch.no_grad():
            spek = model.stft(batch)
            out_spec = session.run(None, {"input": spek.numpy()})[0]
            tar_waves = model.istft(torch.tensor(out_spec))
            tar_sig = tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy()
            tar_signals.append(tar_sig)

        pct = int(25 + (b_end / total_chunks) * 45)
        notify(pct, f"Demixing stems: {b_end}/{total_chunks} chunks completed...")

    # UVR-MDX-NET Inst_HQ directly outputs the isolated instrumental (piano/accompaniment)
    inst_signal = np.concatenate(tar_signals, axis=-1)[:, :n_sample]
    # Isolated vocal stem is obtained via phase-linear spectral residual
    vocal_signal = audio - inst_signal

    notify(72, "Saving clean instrumental and vocal stems (44.1kHz)...")
    output_instrumental_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_instrumental_path), inst_signal.T, 44100, subtype="FLOAT")

    if output_vocal_path:
        output_vocal_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_vocal_path), vocal_signal.T, 44100, subtype="FLOAT")

    notify(75, "SOTA MDX-Net separation completed successfully.")
    return output_instrumental_path, output_vocal_path
