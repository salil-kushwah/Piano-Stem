import struct
import pretty_midi
from backend.midi_parser import midi_to_note_name

with open('another love.cwp', 'rb') as f:
    data = f.read()

records = []
for i in range(len(data) - 17):
    if data[i+6] == 0x90 and data[i+4:i+6] == b'\xff\xff':
        tick = struct.unpack('<I', data[i:i+4])[0]
        note = data[i+7]
        vel = data[i+8]
        dur = struct.unpack('<I', data[i+9:i+13])[0]
        if 21 <= note <= 108 and 1 <= vel <= 127 and 10 <= dur <= 50000:
            records.append((tick, note, vel, dur))

gt_notes = []
for r in sorted(records, key=lambda x: (x[0], x[1])):
    if not any(abs(r[0] - prev['tick']) < 50 and r[1] == prev['pitch'] for prev in gt_notes):
        t_start = r[0] * (60.0 / (120.0 * 960)) + 1.4225
        t_end = (r[0] + r[3]) * (60.0 / (120.0 * 960)) + 1.4225
        gt_notes.append({'tick': r[0], 'start': t_start, 'end': t_end, 'pitch': r[1], 'vel': r[2]})

bd = pretty_midi.PrettyMIDI('storage/midi/another_love_bd.mid')
bd_notes = sorted(bd.instruments[0].notes, key=lambda x: (x.start, x.pitch))

bp = pretty_midi.PrettyMIDI('storage/midi/another_love_bp.mid')
bp_notes = sorted(bp.instruments[0].notes, key=lambda x: (x.start, x.pitch))

# Studio Master verified notes
sm_path = 'storage/midi/e2e_test.mid'
sm = pretty_midi.PrettyMIDI(sm_path)
sm_notes = sorted(sm.instruments[0].notes, key=lambda x: (x.start, x.pitch))

print('=== EVALUATION AGAINST GROUND TRUTH (104 Notes) ===')
for model_name, preds in [('ByteDance CRNN', bd_notes), ('Spotify Basic Pitch', bp_notes), ('Studio Master (Dual-AI Consensus)', sm_notes)]:
    matched = 0
    missed_notes = []
    for g in gt_notes:
        found = False
        for p in preds:
            if g['pitch'] == p.pitch and abs(g['start'] - p.start) <= 0.08:
                matched += 1
                found = True
                break
        if not found:
            missed_notes.append(g)
    
    extra = 0
    for p in preds:
        found = False
        for g in gt_notes:
            if g['pitch'] == p.pitch and abs(g['start'] - p.start) <= 0.08:
                found = True
                break
        if not found:
            extra += 1

    precision = matched / len(preds) if preds else 0
    recall = matched / len(gt_notes) if gt_notes else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f'Model: {model_name}')
    print(f'  Predictions: {len(preds)}')
    print(f'  Matched:     {matched} / {len(gt_notes)}')
    print(f'  Recall:      {recall*100:.1f}%')
    print(f'  Precision:   {precision*100:.1f}%')
    print(f'  F1 Score:    {f1*100:.1f}%')
    print(f'  Extra Notes: {extra}')
    print(f'  Missed:      {len(missed_notes)}')
    if missed_notes:
        print('  First 5 missed notes:')
        for m in missed_notes[:5]:
            pname = midi_to_note_name(m['pitch'])
            t = m['start']
            v = m['vel']
            print(f'    Time {t:5.2f}s | note={pname} ({m["pitch"]}) | vel={v}')
    print()
