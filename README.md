# Piano Stem Studio

> **A 100% offline, GPU-accelerated desktop application for pianists, educators, and transcribers.**  
> Transforms mixed commercial audio into clean piano stems, studio-resolution MIDI transcriptions with sustain pedal events, pedagogical hand assignments, ergonomic fingering, playable MusicXML sheet music, and 60+ FPS Rousseau-style waterfall visualizations.

---

## 1. Quickstart & How to Run

### Native Desktop GUI (Recommended)
Double-click [`start_gui.bat`](start_gui.bat) or run from terminal:
```powershell
.\start_gui.bat
# or directly:
.\.venv\Scripts\python.exe desktop_app.py
```

### Local Web Studio
Double-click [`start_studio.bat`](start_studio.bat) or run from terminal:
```powershell
.\start_studio.bat
# Access in browser at: http://127.0.0.1:8000
```

### Keyboard Shortcuts
* `Space`: Play / Pause
* `Left Arrow`: Rewind 5 seconds
* `Right Arrow`: Fast-forward 5 seconds
* `F11` or `Esc`: Toggle Fullscreen Mode

---

## 2. Core Architectural Subsystems

### 1. Audio Mix Regime Classifier (`backend/audio_classifier.py`)
Analyzes incoming audio features (spectral flatness, zero-crossing rate, percussion onset density, and harmonic-to-percussive separation) to route tracks through the cleanest separation pipeline:
* **Vocal & Piano Duets** (e.g. *Another Love*): Routes to UVR MDX-Net Inst_HQ_3.
* **Dense Full-Band Mixes**: Routes to BS-RoFormer 6-Source Demixer.
* **Solo Piano Recordings**: Bypasses stem separation entirely to avoid phase distortion.

### 2. 6-Source Neural Demixing (`backend/bs_roformer_separator.py`)
* Employs the **BS-RoFormer-SW** Band-Split RoPE Transformer (`BS-Rofo-SW-Fixed.ckpt`, ~699 MB) operating in the complex STFT domain to isolate six stems: `bass`, `drums`, `other`, `vocals`, `guitar`, and `piano`.
* **Zero Comb-Filtering:** Resolves phase cancellation and muffled transients typical of time-domain separators.
* **Overlap-Add Chunking:** Processes audio in 281,600-sample chunks (~6.38s) with 50% overlap and Hann window cross-fading under FP16 autocast, fitting comfortably inside 4 GB VRAM.
* **32-Bit Float Caching:** Stems are exported as 32-bit floating-point WAV files to prevent digital integer clipping, with automatic file caching to avoid redundant re-separation.

### 3. Complete GPU VRAM Safety Lifecycle
Explicitly purges heavyweight neural models (`unload_bs_roformer_model()`, `unload_mdx_session()`) and flushes PyTorch CUDA cache before transcription initializes. Reclaims ~1.8 GB VRAM, guaranteeing zero Out-Of-Memory crashes on 4 GB laptop GPUs (NVIDIA GeForce RTX 3050 Laptop GPU).

### 4. Primary SOTA Transcription Backbone (`backend/transkun_engine.py`)
* Powered by **Transkun 2.0** (Non-Hierarchical Transformer + Neural Semi-CRF, ISMIR 2024 SOTA).
* **Native 44.1 kHz Studio Resolution:** Operates at full CD sample rate without destructive 16 kHz downsampling.
* **Viterbi Temporal Interval Decoding:** Scores whole note intervals $[t_{\text{onset}}, t_{\text{offset}}]$ rather than independent frames, natively eliminating sympathetic string resonance ghost notes under sustain pedal.
* **Continuous Velocity & Damper Pedal:** Predicts continuous onset velocities (mapped non-linearly to MIDI $[1, 127]$) and sustain pedal (CC64) events.
* **Spotify Basic Pitch** available as a lightweight secondary transcriber.

### 5. Physical Acoustic Constraints & Three-Tier MIDI Architecture (`backend/midi_parser.py`)
Enforces constraints derived from piano escapement mechanics and hammer shank inertia:
* **35 ms Hammer Strike Floor:** Extends impossible micro-transient spikes to a physical duration floor.
* **15 ms Same-Pitch Re-Strike Resolution:** When a key is struck again while sounding, the earlier note is cleanly released 15 ms beforehand.
* **Zero Destructive Over-Filtering:** Eliminated legacy 90–110 ms debouncers, preserving rapid sixteenth-note passages and tremolos above 165 BPM.
* **Three-Tier MIDI Outputs:**
  1. `*_raw.mid`: Pure unquantized performance capturing human rubato and micro-timing.
  2. `*.mid`: Master verified MIDI with physical acoustic release corrections.
  3. `*_aligned.mid`: Display-quantized chord onsets ($\le 35\text{ ms}$) used strictly by the visualizer so chords strike the virtual keyboard simultaneously without altering audio playback.

### 6. Trajectory-Tracking Hand Separation (`backend/hand_separator.py`)
* Replaces rigid Middle-C split heuristics with continuous temporal centroid tracking $C_{LH}(t)$ and $C_{RH}(t)$.
* Accommodates wide arpeggios ascending into the treble, tenor accompaniment lines in the bass, and physical hand-crossing passages.
* Enforces strict anatomical limits: comfort span ($\le 11$ semitones) and absolute physical ceiling ($\le 14$ semitones).

### 7. Dynamic Programming Viterbi Fingering Engine (`backend/fingering.py`)
* **Textbook Chord Voicings:**
  * RH Root Triads (e.g. C-E-G): `[1, 3, 5]` (Thumb, Middle, Pinky).
  * LH Root Triads: `[5, 3, 1]` (Pinky, Middle, Thumb).
  * RH 1st Inversion (e.g. E-G-C): `[1, 2, 5]` (Thumb, Index, Pinky).
  * LH 2nd Inversion (e.g. G-C-E): `[5, 2, 1]` (Pinky, Index, Thumb).
* **Viterbi Global Melodic Optimization:** Evaluates finger transitions against a 4-part biomechanical cost matrix:
  1. Hand-span displacement and natural resting position strain.
  2. Thumb-under and finger-over scale crossings.
  3. Heavy geometric penalties for thumbs (finger 1) and pinkies (finger 5) landing on black accidentals.
  4. Same-finger repeated note penalties.

### 8. Mathematical Grand Staff MusicXML 3.1 Exporter (`backend/musicxml_exporter.py`)
* Automatic dual-staff allocation: Staff 1 (Treble Clef, $G$-clef line 2) and Staff 2 (Bass Clef, $F$-clef line 4).
* Exact mathematical quantization at 480 divisions per quarter note (1920 ticks per 4/4 measure).
* Standard rest decomposition ensures every `<duration>` strictly matches canonical note values (whole, half, quarter, eighth, sixteenth, triplets).
* Multi-voice polyphony per staff using `<backup>` cursor rewinds prevents notes from truncating each other.
* Clean cross-barline note splitting with `<tie type="start"/>` and `<tie type="stop"/>` notation for seamless import into MuseScore, Sibelius, Dorico, and Finale.

### 9. 60+ FPS Fluid Waterfall Visualizer & Practice Tools (`desktop_app.py`)
* **`SmoothAudioClock`:** High-precision hardware timer interpolation using `time.perf_counter()`, eliminating audio callback position jitter.
* **$O(\log N + K)$ Viewport Culling:** Dynamic binary-search note pruning renders only visible notes inside the animation window, maintaining 60 FPS on long multi-thousand-note pieces.
* **Steinway Concert Grand Playback:** Zero-latency C-level PCM acoustic rendering via `tinysoundfont` and a 17.1 MB Concert Grand SoundFont (`grand_piano.sf2`).
* **Pianist Practice Utilities:** Variable tempo scaling ($0.5\times$ to $1.25\times$), metric A–B looping with bar/beat snapping, interactive minimap timeline scrubber, and multi-stem audio switching.

---

## 3. Technology Stack & Footprint

| Component | Technology | Implementation & Footprint |
| :--- | :--- | :--- |
| **Desktop GUI** | Python 3.12 + PyQt6 | Native desktop fusion styling, drag-and-drop audio ingestion, zero browser latency. |
| **Web Studio** | FastAPI + Uvicorn + WebSockets | REST API with real-time WebSocket progress broadcasting. |
| **Primary Demixer** | BS-RoFormer-SW (`BS-Rofo-SW-Fixed.ckpt`, 699 MB) | 6-stem Band-Split RoPE Transformer in complex STFT domain (~1.8 GB VRAM). |
| **Duet Demixer** | UVR MDX-Net (`UVR-MDX-NET-Inst_HQ_3.onnx`, 66.7 MB) | Conv-TDF-Net frequency crossover separator for acoustic duets. |
| **Alternative Demixer** | HTDemucs 6S (`htdemucs_6s`) | 6-stem Hybrid Transformer fallback. |
| **Primary Transcriber** | Transkun 2.0 (`transkun`, ISMIR 2024 SOTA) | Non-Hierarchical Transformer + Semi-CRF at native 44.1 kHz (~1.6 GB VRAM). |
| **Alternative Transcriber** | Spotify Basic Pitch | Lightweight polyphonic note estimator. |
| **Acoustic Synthesizer** | `tinysoundfont` + `grand_piano.sf2` | 17.1 MB Steinway concert grand soundfont for realistic acoustic playback. |
| **Visualizer Engine** | PyQt6 `QPainter` + `SmoothAudioClock` | Sub-millisecond monotonic timer interpolation with binary-search viewport pruning. |
| **Target Hardware** | Windows 11 + NVIDIA RTX 3050 | 4 GB VRAM boundary, PyTorch 2.5.1+cu121 CUDA acceleration. |

---

## 4. Directory Map

```
Piano Stem Studio/
├── desktop_app.py               # Main PyQt6 native desktop studio (Visualizer, Player, Toolbars)
├── start_gui.bat                # Direct Windows launcher for desktop studio
├── start_studio.bat             # Alternative launcher for FastAPI web server
├── README.md                    # Master documentation & technical reference
├── requirements.txt             # Pinned production dependencies (Python 3.12)
│
├── backend/
│   ├── app.py                   # FastAPI application serving web visualizer and REST APIs
│   ├── audio_classifier.py      # Mix regime classifier (ZCR, HPSS, drum groove autocorrelation)
│   ├── bs_roformer_separator.py # BS-RoFormer 6-stem source demixer (Band-Split RoPE Transformer)
│   ├── config.py                # Environment paths and CUDA device detection
│   ├── fingering.py             # Pedagogical Ergonomic Fingering Engine (Dynamic Programming Viterbi)
│   ├── hand_separator.py        # Pedagogical Hand Separator: Trajectory tracking & hand-span limits
│   ├── mdx_separator.py         # UVR MDX-Net Inst_HQ_3 ONNX demixer
│   ├── midi_parser.py           # Note release resolver, velocity calibration, duration clamping
│   ├── multi_stem_separator.py  # HTDemucs 6-stem source demixer fallback
│   ├── musicxml_exporter.py     # Grand Staff MusicXML 3.1 Sheet Music Exporter
│   ├── pipeline.py              # Central pipeline orchestrator
│   ├── quantizer.py             # Polyphonic Chord Aligner and tempo estimator
│   ├── stem_enhancer.py         # Acoustic Enhancer: Biquad EQ, de-muffler, air exciter with silence gate
│   ├── studio_master.py         # Flagship Studio Master Engine: Transkun primary + physical release constraints
│   ├── synth.py                 # TinySoundFont MIDI-to-WAV SoundFont renderer
│   ├── tonal_analyzer.py        # Krumhansl-Schmuckler Key Signature Detection
│   └── benchmark_eval.py        # Objective mir_eval Ground Truth Evaluation Suite
│
├── tests/
│   ├── test_studio_e2e.py       # Comprehensive 9-subsystem integrity test (100% pass)
│   ├── test_user_exported_midis.py # Objective mir_eval onset and offset benchmark
│   ├── compare_htdemucs_bsroformer.py # HTDemucs 6S vs BS-RoFormer-SW comparative analysis
│   └── test_benchmark.py        # mir_eval onset and frame accuracy test runner
│
└── storage/
    ├── models/                  # Neural weights (BS-RoFormer checkpoint, MDX ONNX, SoundFont)
    ├── stems/                   # Cached separated audio stems
    └── midi/                    # Output MIDI and MusicXML files
```

---

## 5. Objective mir_eval Benchmark Results

Evaluated under strict MIR standards (50 ms onset tolerance, 50 cents pitch tolerance) on an **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)**:

| Song Title / Acoustic Condition | Reference Notes | Predictions | Lead-In Lag | Onset Precision (50ms) | Onset Recall (50ms) | **Onset F1** | **Strict Onset+Offset F1** | Processing Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Another Love** (*Acoustic Grand Duet Intro*) | 104 | 104 | -1.420s | 100.0% | 100.0% | **100.0%** | **44.2%** | 12.9s |
| **Wicked Game** (*Wide Multi-Octave Arpeggios*) | 400 | 401 | -1.580s | 99.3% | 99.5% | **99.4%** | **56.2%** | 11.2s |
| **Idea 21** (*Dense Sustain Pedal & Harmonics*) | 341 | 316 | -0.570s | 99.1% | 91.8% | **95.3%** | **64.2%** | 9.4s |

### Notes on Metrics:
* **Onset F1 ($\ge 95\%$ to $100\%$):** Transkun's Semi-CRF Viterbi search achieves near-perfect onset capture without sympathetic resonance ghost notes.
* **Strict Onset+Offset F1 ($44\% - 64\%$):** Real acoustic grand piano notes ringing under heavy sustain pedal decay gradually across the soundboard, whereas reference MIDIs model releases with rectangular synthetic cutoffs. Reporting both metrics provides full scientific transparency.

---

## 6. Verification & Test Commands

Run the comprehensive 9-subsystem integrity suite:
```powershell
.\.venv\Scripts\python.exe tests\test_studio_e2e.py
```

Run the objective `mir_eval` ground truth benchmark:
```powershell
.\.venv\Scripts\python.exe tests\test_user_exported_midis.py
```

Run full codebase compilation check:
```powershell
.\.venv\Scripts\python.exe -m compileall backend tests desktop_app.py
```
