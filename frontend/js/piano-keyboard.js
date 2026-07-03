/**
 * PianoKeyboard: Computes accurate geometry and renders an authentic 88-key piano.
 * MIDI range: 21 (A0) to 108 (C8)
 * 52 White keys, 36 Black keys.
 */

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const IS_BLACK_KEY = [false, true, false, true, false, false, true, false, true, false, true, false];

export class PianoKeyboard {
    constructor() {
        this.startMidi = 21;  // A0
        this.endMidi = 108;   // C8
        this.totalWhiteKeys = 52;
        this.keys = [];
        this.keyMap = new Map(); // midi -> key object
        this.activeKeys = new Map(); // midi -> { velocity, color }
        this.initKeyMetadata();
    }

    initKeyMetadata() {
        let whiteIndex = 0;
        for (let midi = this.startMidi; midi <= this.endMidi; midi++) {
            const noteInOctave = midi % 12;
            const isBlack = IS_BLACK_KEY[noteInOctave];
            const octave = Math.floor(midi / 12) - 1;
            const name = `${NOTE_NAMES[noteInOctave]}${octave}`;

            const keyInfo = {
                midi,
                name,
                isBlack,
                whiteIndex: isBlack ? -1 : whiteIndex,
                x: 0,
                width: 0,
                height: 0
            };

            this.keys.push(keyInfo);
            this.keyMap.set(midi, keyInfo);

            if (!isBlack) {
                whiteIndex++;
            }
        }
    }

    updateDimensions(canvasWidth, keyboardHeight) {
        this.width = canvasWidth;
        this.height = keyboardHeight;
        this.whiteKeyWidth = canvasWidth / this.totalWhiteKeys;
        this.blackKeyWidth = this.whiteKeyWidth * 0.62;
        this.blackKeyHeight = keyboardHeight * 0.64;

        // Position white keys
        for (const key of this.keys) {
            if (!key.isBlack) {
                key.x = key.whiteIndex * this.whiteKeyWidth;
                key.width = this.whiteKeyWidth;
                key.height = keyboardHeight;
            }
        }

        // Position black keys between corresponding white keys
        for (const key of this.keys) {
            if (key.isBlack) {
                const noteInOctave = key.midi % 12;
                // Find preceding white key
                const prevWhite = this.keyMap.get(key.midi - 1);
                if (prevWhite) {
                    key.x = prevWhite.x + this.whiteKeyWidth - (this.blackKeyWidth / 2);
                    // Fine-tune acoustic piano black key offsets
                    if (noteInOctave === 1) key.x -= this.whiteKeyWidth * 0.08; // C#
                    if (noteInOctave === 3) key.x += this.whiteKeyWidth * 0.08; // D#
                    if (noteInOctave === 6) key.x -= this.whiteKeyWidth * 0.10; // F#
                    if (noteInOctave === 10) key.x += this.whiteKeyWidth * 0.10; // A#
                    key.width = this.blackKeyWidth;
                    key.height = this.blackKeyHeight;
                }
            }
        }
    }

    pressKey(midi, velocity = 0.8, color = null) {
        this.activeKeys.set(midi, { velocity, color });
    }

    releaseKey(midi) {
        this.activeKeys.delete(midi);
    }

    releaseAll() {
        this.activeKeys.clear();
    }

    getKey(midi) {
        return this.keyMap.get(midi);
    }

    render(ctx, yOffset, theme) {
        ctx.save();
        ctx.translate(0, yOffset);

        // 1. Draw keyboard base shadow / backing
        ctx.fillStyle = '#111215';
        ctx.fillRect(0, 0, this.width, this.height);

        // 2. Render White Keys
        for (const key of this.keys) {
            if (key.isBlack) continue;
            const active = this.activeKeys.get(key.midi);

            ctx.save();
            if (active) {
                // Key pressed glow & color
                const activeColor = active.color || theme.whiteActive;
                const grad = ctx.createLinearGradient(key.x, 0, key.x, key.height);
                grad.addColorStop(0, '#ffffff');
                grad.addColorStop(0.5, activeColor);
                grad.addColorStop(1, theme.accent);
                ctx.fillStyle = grad;
                ctx.shadowColor = theme.glow;
                ctx.shadowBlur = 18;
            } else {
                // Realistic ivory gradient
                const grad = ctx.createLinearGradient(key.x, 0, key.x, key.height);
                grad.addColorStop(0, '#f2f3f5');
                grad.addColorStop(0.85, '#ffffff');
                grad.addColorStop(1, '#d8dbe0');
                ctx.fillStyle = grad;
            }

            // Draw rounded white key
            const r = 3;
            ctx.beginPath();
            ctx.roundRect(key.x + 0.5, 0, key.width - 1, key.height, [0, 0, r, r]);
            ctx.fill();

            // Key divider line
            ctx.strokeStyle = '#22252a';
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.restore();
        }

        // 3. Render Black Keys
        for (const key of this.keys) {
            if (!key.isBlack) continue;
            const active = this.activeKeys.get(key.midi);

            ctx.save();
            if (active) {
                const activeColor = active.color || theme.blackActive;
                const grad = ctx.createLinearGradient(key.x, 0, key.x, key.height);
                grad.addColorStop(0, '#444');
                grad.addColorStop(0.6, activeColor);
                grad.addColorStop(1, theme.accent);
                ctx.fillStyle = grad;
                ctx.shadowColor = theme.glow;
                ctx.shadowBlur = 22;
            } else {
                // Matte ebony gradient with edge bevel
                const grad = ctx.createLinearGradient(key.x, 0, key.x, key.height);
                grad.addColorStop(0, '#1c1d21');
                grad.addColorStop(0.8, '#0b0c0e');
                grad.addColorStop(1, '#1e2024');
                ctx.fillStyle = grad;
                ctx.shadowColor = 'rgba(0, 0, 0, 0.6)';
                ctx.shadowBlur = 6;
                ctx.shadowOffsetY = 4;
            }

            const r = 2;
            ctx.beginPath();
            ctx.roundRect(key.x, 0, key.width, key.height, [0, 0, r, r]);
            ctx.fill();

            // Gloss highlight on black key
            if (!active) {
                ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
                ctx.fillRect(key.x + 2, 2, key.width - 4, key.height * 0.7);
            }

            ctx.restore();
        }

        // 4. Red felt trim bar along the top of keys
        ctx.fillStyle = '#b71c1c';
        ctx.fillRect(0, 0, this.width, 3);

        ctx.restore();
    }
}
