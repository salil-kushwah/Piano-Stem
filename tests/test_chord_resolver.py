import pretty_midi
import numpy as np

def resolve_chords_and_releases(notes, max_treble_dur=1.2, max_bass_dur=2.0, min_dur=0.18):
    """
    Intelligent Polyphonic Chord & Release Engine:
    1. Anti-Click Gate: Lengthens micro-chopped chord notes (< 150ms) to min_dur (180ms)
    2. Chord Cohesion: Notes struck within 45ms belonging to the same chord harmonize their releases.
       Eliminates fractured chords where one note dies in 10ms and another hangs for 6s.
    3. Monophonic Same-Pitch Truncation: Prevents same key from overlapping.
    4. Musical Duration Caps: Caps runaway reverb tails while preserving chord harmonies.
    """
    sorted_notes = sorted(notes, key=lambda x: (x.start, x.pitch))
    
    # Step 1: Detect Chords (clusters of notes starting within 45ms)
    chords = []
    if sorted_notes:
        cur_chord = [sorted_notes[0]]
        for n in sorted_notes[1:]:
            if (n.start - cur_chord[0].start) <= 0.045:
                cur_chord.append(n)
            else:
                chords.append(cur_chord)
                cur_chord = [n]
        chords.append(cur_chord)
    
    # Step 2: Harmonize chord note durations
    harmonized = []
    for c in chords:
        if len(c) > 1:
            # Multi-note chord!
            # Find the dominant musical duration among notes that didn't glitch (durs > 0.15s and < 4.0s)
            valid_durs = [n.end - n.start for n in c if 0.15 <= (n.end - n.start) <= 3.5]
            if valid_durs:
                target_dur = float(np.median(valid_durs))
            else:
                target_dur = 0.8  # Default solid chord length
            
            # Apply target duration to chord notes with small register variance (bass notes can ring slightly longer)
            for n in c:
                orig_dur = n.end - n.start
                if orig_dur < 0.15 or orig_dur > 3.5:
                    # Fix glitched notes (either 10ms click or 6s runaway)
                    if n.pitch < 60:
                        n.end = n.start + min(target_dur * 1.25, max_bass_dur)
                    else:
                        n.end = n.start + min(target_dur, max_treble_dur)
                else:
                    # Keep natural touch but cap to register maximum
                    cap = max_treble_dur if n.pitch >= 60 else max_bass_dur
                    n.end = min(n.end, n.start + cap)
                
                n.end = max(n.start + min_dur, n.end)
                harmonized.append(n)
        else:
            # Single melody / bass note
            n = c[0]
            dur = n.end - n.start
            cap = max_treble_dur if n.pitch >= 60 else max_bass_dur
            if dur < min_dur:
                n.end = n.start + min_dur
            else:
                n.end = min(n.end, n.start + cap)
            harmonized.append(n)
            
    # Step 3: Monophonic Same-Pitch Truncation
    harmonized.sort(key=lambda x: (x.start, x.pitch))
    for i, n in enumerate(harmonized):
        next_same = next((nxt for nxt in harmonized[i+1:] if nxt.pitch == n.pitch), None)
        if next_same and next_same.start < n.end:
            n.end = max(n.start + min_dur, next_same.start - 0.03)

    return harmonized

pm = pretty_midi.PrettyMIDI('storage/test_lucifer/htdemucs_other.mid')
orig_notes = [pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end) for n in pm.instruments[0].notes]
resolved = resolve_chords_and_releases(orig_notes)

durs = [n.end - n.start for n in resolved]
print("=== AFTER INTELLIGENT CHORD COHESION & RELEASE ENGINE ===")
print(f"Total Notes: {len(resolved)}")
print(f"Durations: min={min(durs):.3f}s, avg={np.mean(durs):.3f}s, median={np.median(durs):.3f}s, max={max(durs):.3f}s")
print(f"Notes < 0.15s: {sum(1 for d in durs if d < 0.15)}")
print(f"Notes > 2.0s:  {sum(1 for d in durs if d > 2.0)}")

# Check chord sample at 17.88s (which previously had a 0.01s broken note)
chord_17 = [n for n in resolved if abs(n.start - 17.88) < 0.05]
print("\nChord at 17.88s:")
for n in chord_17:
    print(f"  {pretty_midi.note_number_to_name(n.pitch)}: start={n.start:.2f}s, dur={n.end-n.start:.3f}s, vel={n.velocity}")

# Check chord sample at 44.00s (which previously had two 6.0s runaway notes)
chord_44 = [n for n in resolved if abs(n.start - 44.00) < 0.05]
print("\nChord at 44.00s:")
for n in chord_44:
    print(f"  {pretty_midi.note_number_to_name(n.pitch)}: start={n.start:.2f}s, dur={n.end-n.start:.3f}s, vel={n.velocity}")
