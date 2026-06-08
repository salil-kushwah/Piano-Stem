import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
STEMS_DIR = STORAGE_DIR / "stems"
MIDI_DIR = STORAGE_DIR / "midi"

MODELS_DIR = STORAGE_DIR / "models"
for folder in [UPLOADS_DIR, STEMS_DIR, MIDI_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Hardware / GPU Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SOTA Stem Separation Configuration (UVR MDX-Net Inst_HQ_3)
# Replaces legacy Demucs with frequency-domain ConvTDFNet
MDX_MODEL_PATH = MODELS_DIR / "UVR-MDX-NET-Inst_HQ_3.onnx"
MDX_BATCH_SIZE = 4


# Piano Transcription Configuration
TRANSCRIPTION_SAMPLE_RATE = 16000
