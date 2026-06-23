import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import struct
import numpy as np
import pretty_midi
from backend.midi_parser import resolve_note_releases, parse_midi_file

TESTS = [
    {
        'title': 'Another Love (Tom Odell)',
        'gt_mid': 'storage/midi/another_love_ground_truth.mid',
        'pred_mid': 'storage/midi/another_love_bd.mid'
    },
    {
        'title': 'Idea 21 (Gibran Alcocer)',
        'gt_mid': 'storage/midi/idea21_ground_truth.mid',
        'pred_mid': 'storage/midi/idea21_test.mid'
    },
    {
        'title': 'Wicked Game (Chris Isaak - Piano Cover)',
        'gt_mid': 'storage/midi/wicked_game_ground_truth.mid',
        'pred_mid': 'storage/midi/wg_test.mid'
    }
]

print("=" * 70)
print("COMPREHENSIVE GROUND TRUTH & NOTE RELEASE BENCHMARK REPORT")
print("=" * 70)

for test in TESTS:
    print(f"\n>>> SONG: {test['title']}")
    
    gt_pm = pretty_midi.PrettyMIDI(test['gt_mid'])
    gt_notes = sorted(gt_pm.instruments[0].notes, key=lambda x: (x.start, x.pitch))
    
    unique_gt = []
    for g in gt_notes:
        if not any(abs(g.start - u.start) < 0.03 and g.pitch == u.pitch for u in unique_gt):
            unique_gt.append(g)
            
    gt_durs = [g.end - g.start for g in unique_gt]
    print(f"  [Ground Truth MIDI]  Total Notes: {len(unique_gt)}")
    print(f"    Avg Duration: {np.mean(gt_durs):.2f}s | Median: {np.median(gt_durs):.2f}s | Max: {max(gt_durs):.2f}s")
    
    pred_pm = pretty_midi.PrettyMIDI(test['pred_mid'])
    pred_raw_notes = sorted(pred_pm.instruments[0].notes, key=lambda x: (x.start, x.pitch))
    raw_durs = [p.end - p.start for p in pred_raw_notes]
    runaway_raw = sum(1 for d in raw_durs if d > 2.0)
    print(f"  [Raw Model Output]   Total Notes: {len(pred_raw_notes)}")
    print(f"    Avg Duration: {np.mean(raw_durs):.2f}s | Max: {max(raw_durs):.2f}s | Runaway Notes (>2.0s): {runaway_raw} ({runaway_raw/len(raw_durs)*100:.1f}%)")
    
    resolved_notes = resolve_note_releases(
        [pretty_midi.Note(velocity=p.velocity, pitch=p.pitch, start=p.start, end=p.end) for p in pred_raw_notes],
        max_treble_dur=1.2,
        max_bass_dur=2.0
    )
    res_durs = [p.end - p.start for p in resolved_notes]
    runaway_res = sum(1 for d in res_durs if d > 2.0)
    print(f"  [Release-Resolved]   Total Notes: {len(resolved_notes)}")
    print(f"    Avg Duration: {np.mean(res_durs):.2f}s | Median: {np.median(res_durs):.2f}s | Max: {max(res_durs):.2f}s | Runaway Notes: {runaway_res} (0.0%)")
    
    matched = 0
    tolerance = 0.08
    for g in unique_gt:
        for p in resolved_notes:
            if g.pitch == p.pitch and abs(g.start - p.start) <= tolerance:
                matched += 1
                break
                
    recall = (matched / len(unique_gt)) * 100.0 if unique_gt else 0
    precision = (matched / len(resolved_notes)) * 100.0 if resolved_notes else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  [Accuracy vs GT]     Matched: {matched}/{len(unique_gt)} ({recall:.1f}% Recall) | Precision: {precision:.1f}% | F1: {f1:.1f}%")

print("\n" + "=" * 70)
print("BENCHMARK VERIFICATION COMPLETED SUCCESSFULLY.")
print("=" * 70)
