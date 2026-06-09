import os
import sys
import time
import shutil
from pathlib import Path
from typing import Callable, Optional, Dict, Any

from backend.config import DEVICE, STEMS_DIR, MIDI_DIR
from backend.midi_parser import parse_midi_file
from backend.mdx_separator import separate_mixed_audio_mdx, unload_mdx_session
from backend.multi_stem_separator import separate_multistem_audio, unload_htdemucs_model
from backend.bs_roformer_separator import separate_bs_roformer_audio, unload_bs_roformer_model
from backend.studio_master import run_studio_master_engine


class ProcessingPipeline:
    def __init__(self, task_id: str, progress_callback: Optional[Callable[[int, str], None]] = None):
        self.task_id = task_id
        self.progress_callback = progress_callback

    def notify(self, progress: int, stage_message: str):
        try:
            print(f"[{self.task_id}] ({progress}%) {stage_message}")
        except UnicodeEncodeError:
            print(f"[{self.task_id}] ({progress}%) {stage_message.encode('ascii', errors='replace').decode('ascii')}")
        if self.progress_callback:
            try:
                self.progress_callback(progress, stage_message)
            except Exception as e:
                print(f"Error in progress callback: {e}")

    def run(
        self,
        input_audio_path: Path,
        is_mixed_audio: bool = False,
        precision_mode: str = "clean",
        model_engine: str = "studio_master",
        stem_mode: str = "bs_roformer",
        enable_enhancer: bool = False,
        force_separation: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the modernized SOTA transcription pipeline:
        1. Source Separation (BS-RoFormer 6-Source or MDX-Net Inst_HQ_3)
           - Includes intelligent stem caching to avoid redundant re-demixing.
           - Fails loudly with descriptive error if stem isolation encounters an issue.
        2. High-Resolution Transcription (Transkun 2.0 Transformer Semi-CRF at 44.1kHz)
        3. Concert Grand SoundFont Synthesis
        4. Trajectory-tracked hand separation and Dynamic Programming Viterbi fingering
        5. Mathematically exact MusicXML 3.1 grand staff sheet music export
        """
        start_time = time.time()
        self.notify(5, "Analyzing audio file...")

        stem_audio_path = STEMS_DIR / f"{self.task_id}_piano.wav"
        enhanced_stem_path = STEMS_DIR / f"{self.task_id}_piano_enhanced.wav"
        vocal_stem_path = STEMS_DIR / f"{self.task_id}_vocals.wav"
        drums_stem_path = STEMS_DIR / f"{self.task_id}_drums.wav"
        bass_stem_path = STEMS_DIR / f"{self.task_id}_bass.wav"
        pure_piano_path = STEMS_DIR / f"{self.task_id}_pure_piano.wav"
        midi_path = MIDI_DIR / f"{self.task_id}.mid"
        musicxml_path = MIDI_DIR / f"{self.task_id}.musicxml"

        # Auto-detect mix regime if requested
        detected_regime = None
        if stem_mode in ("auto", "auto_detect", None) and is_mixed_audio:
            from backend.audio_classifier import detect_audio_regime
            self.notify(5, "Analyzing audio mix profile (detecting instruments, vocals, acoustics)...")
            try:
                profile = detect_audio_regime(input_audio_path)
                stem_mode = profile.get("recommended_stem_mode", "bs_roformer")
                detected_regime = profile
                self.notify(8, f"Detected: {profile.get('label', 'Mixed Audio')}")
            except Exception as e_cls:
                print(f"[Pipeline] Mix classifier notice: {e_cls}")
                stem_mode = "bs_roformer"

        if not is_mixed_audio:
            stem_mode = "none"

        # ---------------------------------------------------------
        # Step 1: Stem Isolation (with Caching & Loud Failure)
        # ---------------------------------------------------------
        # Check if stem is already cached
        stem_cached = (
            not force_separation
            and stem_audio_path.exists()
            and stem_audio_path.stat().st_size > 1000
        )

        if stem_cached:
            self.notify(20, f"Using cached separated piano stem: {stem_audio_path.name}")
        elif stem_mode in ("bs_roformer", "roformer_6s", "sw_roformer") or (is_mixed_audio and stem_mode not in ("demucs_6s", "mdx_inst", "none")):
            self.notify(10, "Demixing mixed track with BS-RoFormer 6-Source Demixer (Zero Comb-Filtering)...")
            try:
                separate_bs_roformer_audio(
                    input_audio_path=input_audio_path,
                    output_piano_path=stem_audio_path,
                    output_vocal_path=vocal_stem_path,
                    output_drums_path=drums_stem_path,
                    output_bass_path=bass_stem_path,
                    progress_callback=self.notify
                )
            except Exception as e:
                error_msg = f"BS-RoFormer separation failed: {e}. Model weights or execution error."
                print(f"[Pipeline] ERROR: {error_msg}")
                raise RuntimeError(error_msg)
        elif stem_mode in ("demucs_6s", "mixed", "demucs_4s"):
            self.notify(10, "Demixing mixed track with HTDemucs 6S Piano Isolation Engine...")
            try:
                separate_multistem_audio(
                    input_audio_path=input_audio_path,
                    output_piano_path=stem_audio_path,
                    output_vocal_path=vocal_stem_path,
                    output_drums_path=drums_stem_path,
                    output_bass_path=bass_stem_path,
                    progress_callback=self.notify
                )
            except Exception as e:
                error_msg = f"HTDemucs stem demixing failed: {e}."
                print(f"[Pipeline] ERROR: {error_msg}")
                raise RuntimeError(error_msg)
        elif stem_mode == "mdx_inst":
            self.notify(10, "Demixing mixed track with UVR MDX-Net Inst_HQ_3 Engine...")
            try:
                separate_mixed_audio_mdx(
                    input_audio_path=input_audio_path,
                    output_instrumental_path=stem_audio_path,
                    output_vocal_path=vocal_stem_path,
                    progress_callback=self.notify
                )
            except Exception as e:
                error_msg = f"MDX-Net stem demixing failed: {e}."
                print(f"[Pipeline] ERROR: {error_msg}")
                raise RuntimeError(error_msg)
        else:
            self.notify(20, "Solo piano recording selected. Skipping stem separation...")
            shutil.copy(input_audio_path, stem_audio_path)

        # Reclaim GPU VRAM from separation models before transcription
        unload_bs_roformer_model()
        unload_htdemucs_model()
        unload_mdx_session()

        # Step 1.5: Optional Acoustic Stem Enhancement (kept behind flag, default off)
        transcribe_audio_input = stem_audio_path
        if enable_enhancer and (is_mixed_audio or stem_mode != "none"):
            self.notify(35, "Enhancing isolated piano stem (dereverberation, de-hissing, presence shelf)...")
            try:
                from backend.stem_enhancer import enhance_piano_stem
                enhance_piano_stem(
                    stem_path=stem_audio_path,
                    output_path=enhanced_stem_path,
                    dereverb_strength=0.35,
                    high_shelf_gain_db=2.0
                )
                if enhanced_stem_path.exists():
                    transcribe_audio_input = enhanced_stem_path
            except Exception as e:
                print(f"[Pipeline] Stem enhancement notice: {e}. Using raw stem.")

        self.notify(40, "Piano track ready. Commencing high-accuracy transcription...")

        # ---------------------------------------------------------
        # Step 2: High-Resolution Piano Transcription (Transkun SOTA)
        # ---------------------------------------------------------
        try:
            from backend.synth import synthesize_midi_to_audio

            raw_midi_path = MIDI_DIR / f"{self.task_id}_raw.mid"
            display_midi_path = MIDI_DIR / f"{self.task_id}_aligned.mid"
            raw_piano_path = STEMS_DIR / f"{self.task_id}_raw_piano.wav"

            self.notify(45, "Running SOTA Transformer Semi-CRF Studio Master Engine (Transkun 2.0 @ 44.1kHz)...")
            master_res = run_studio_master_engine(
                audio_path=transcribe_audio_input,
                output_midi_path=midi_path,
                vocal_stem_path=vocal_stem_path if vocal_stem_path.exists() else None,
                drums_stem_path=drums_stem_path if drums_stem_path.exists() else None,
                bass_stem_path=bass_stem_path if bass_stem_path.exists() else None,
                precision_mode=precision_mode,
                is_mixed_audio=is_mixed_audio,
                engine="transkun",
                progress_callback=self.notify
            )
            master_stats = master_res.get("stats", {})
            provenance_list = master_stats.get("provenance", [])
            if master_res.get("raw_midi_file"):
                raw_midi_path = Path(master_res["raw_midi_file"])
            if master_res.get("display_midi_file"):
                display_midi_path = Path(master_res["display_midi_file"])

            self.notify(88, "Assigning pedagogical hands, fingering, and chord annotations...")
            visualizer_data = parse_midi_file(midi_path, provenance_list=provenance_list)

            # Parse raw MIDI for side-by-side comparison in UI
            raw_visualizer_data = None
            if raw_midi_path.exists():
                try:
                    raw_visualizer_data = parse_midi_file(raw_midi_path)
                except Exception:
                    pass

            self.notify(92, "Synthesizing authentic Concert Grand Piano (17MB SoundFont)...")
            synthesize_midi_to_audio(midi_path, pure_piano_path)

            if raw_midi_path.exists():
                try:
                    synthesize_midi_to_audio(raw_midi_path, raw_piano_path)
                except Exception as e_raw_synth:
                    print(f"[Pipeline] Raw synth notice: {e_raw_synth}")

            # Export standard MusicXML 3.1 grand staff sheet music
            try:
                from backend.musicxml_exporter import export_to_musicxml
                key_name = visualizer_data.get("key_signature", "C Major")
                tempo_bpm = visualizer_data.get("tempo_bpm", 120.0)
                export_to_musicxml(
                    notes=visualizer_data.get("notes", []),
                    output_xml_path=musicxml_path,
                    title=Path(input_audio_path).stem,
                    key_name=key_name,
                    tempo_bpm=tempo_bpm
                )
            except Exception as e_xml:
                print(f"[Pipeline] Warning: MusicXML export failed: {e_xml}")

        except Exception as e:
            print(f"Piano transcription failed: {e}")
            raise RuntimeError(f"Piano transcription failed: {e}")

        elapsed = round(time.time() - start_time, 2)
        self.notify(100, f"Processing complete in {elapsed}s!")

        return {
            "task_id": self.task_id,
            "elapsed_seconds": elapsed,
            "original_audio_file": str(input_audio_path),
            "stem_audio_file": str(stem_audio_path),
            "enhanced_stem_file": str(enhanced_stem_path) if enhanced_stem_path.exists() else None,
            "pure_piano_file": str(pure_piano_path),
            "raw_piano_file": str(raw_piano_path) if raw_piano_path.exists() else None,
            "raw_midi_file": str(raw_midi_path) if raw_midi_path.exists() else None,
            "midi_file": str(midi_path),
            "display_midi_file": str(display_midi_path) if display_midi_path.exists() else None,
            "musicxml_file": str(musicxml_path) if musicxml_path.exists() else None,
            "vocal_stem_file": str(vocal_stem_path) if vocal_stem_path.exists() else None,
            "drums_stem_file": str(drums_stem_path) if drums_stem_path.exists() else None,
            "bass_stem_file": str(bass_stem_path) if bass_stem_path.exists() else None,
            "mix_regime": detected_regime,
            "visualizer_data": visualizer_data,
            "raw_visualizer_data": raw_visualizer_data,
            "stats": master_stats
        }
