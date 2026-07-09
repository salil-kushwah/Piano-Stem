import time
from pathlib import Path
from backend.pipeline import ProcessingPipeline

full_audio_path = Path(r"C:\Users\salil\Downloads\Lucifer_ Bloody Celestial Karaoke Jam Soundtrack _ Wicked Game  - Tom Ellis _ WaterTower.mp3")

print("Running full end-to-end pipeline on Lucifer Wicked Game (Full 2.36 min)...")
t0 = time.time()
pipeline = ProcessingPipeline(task_id="lucifer_wg")
result = pipeline.run(
    input_audio_path=full_audio_path,
    is_mixed_audio=True,
    precision_mode="clean",
    model_engine="studio_master",
    stem_mode="mdx_inst"
)
elapsed = time.time() - t0
print(f"Full pipeline completed in {elapsed:.2f}s!")
print(f"Isolated Piano Audio: {result['stem_audio_file']}")
print(f"Synthesized Grand:    {result['pure_piano_file']}")
print(f"Clean MIDI File:      {result['midi_file']}")
print(f"Visualizer notes:     {result['visualizer_data']['notes_count']}")
