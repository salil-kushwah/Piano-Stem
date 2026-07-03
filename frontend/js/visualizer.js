/**
 * RousseauVisualizer: 60FPS Falling-Notes Piano Visualizer Engine
 * Features:
 * - High-DPI Canvas Rendering
 * - Rousseau-Style Rounded Pill Falling Notes with Neon Gradients
 * - Impact Particle Bursts & Key Strike Flash
 * - Authentic 88-Key Keyboard with Real-time Pressed Keys
 * - Left/Right Hand Color Themes
 * - Sustain Pedal Status Visualizer
 */

import { PianoKeyboard } from './piano-keyboard.js';

export const THEMES = {
    rousseau_cyan: {
        id: 'rousseau_cyan',
        name: 'Rousseau Classic (Electric Blue / Cyan)',
        leftHand: { top: '#0072ff', bottom: '#00c6ff', glow: '#00c6ff' },
        rightHand: { top: '#00c6ff', bottom: '#00f2fe', glow: '#00f2fe' },
        glow: 'rgba(0, 198, 255, 0.65)',
        accent: '#00c6ff',
        whiteActive: '#66e2ff',
        blackActive: '#0090cc'
    },
    magenta_neon: {
        id: 'magenta_neon',
        name: 'Cyber Magenta / Purple',
        leftHand: { top: '#7928ca', bottom: '#b800ff', glow: '#b800ff' },
        rightHand: { top: '#ff0080', bottom: '#ff007f', glow: '#ff007f' },
        glow: 'rgba(235, 0, 141, 0.65)',
        accent: '#ff007f',
        whiteActive: '#ff77bb',
        blackActive: '#cc0066'
    },
    amber_gold: {
        id: 'amber_gold',
        name: 'Imperial Gold / Amber',
        leftHand: { top: '#b8860b', bottom: '#f39c12', glow: '#f39c12' },
        rightHand: { top: '#f1c40f', bottom: '#f39c12', glow: '#f1c40f' },
        glow: 'rgba(243, 156, 18, 0.65)',
        accent: '#f39c12',
        whiteActive: '#fce5cd',
        blackActive: '#d68910'
    },
    emerald_aura: {
        id: 'emerald_aura',
        name: 'Emerald Matrix',
        leftHand: { top: '#007965', bottom: '#00af91', glow: '#00af91' },
        rightHand: { top: '#00e676', bottom: '#69f0ae', glow: '#69f0ae' },
        glow: 'rgba(0, 230, 118, 0.65)',
        accent: '#00e676',
        whiteActive: '#a7f3d0',
        blackActive: '#059669'
    }
};

class Particle {
    constructor(x, y, color) {
        this.x = x;
        this.y = y;
        this.vx = (Math.random() - 0.5) * 4;
        this.vy = -Math.random() * 5 - 2;
        this.size = Math.random() * 3.5 + 1.5;
        this.life = 1.0;
        this.decay = Math.random() * 0.04 + 0.02;
        this.color = color;
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.vy += 0.15; // gravity
        this.life -= this.decay;
    }

    draw(ctx) {
        if (this.life <= 0) return;
        ctx.save();
        ctx.globalAlpha = Math.max(0, this.life);
        ctx.fillStyle = this.color;
        ctx.shadowColor = this.color;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }
}

export class RousseauVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.keyboard = new PianoKeyboard();

        this.notes = [];
        this.pedals = [];
        this.duration = 0;
        this.currentTime = 0;
        this.isPlaying = false;

        // Visual parameters
        this.noteFallSpeed = 400; // pixels per second falling down
        this.keyboardHeight = 140; // bottom keyboard height
        this.theme = THEMES.rousseau_cyan;
        this.particles = [];

        // Sustain pedal state
        this.isPedalActive = false;

        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    setTheme(themeKey) {
        if (THEMES[themeKey]) {
            this.theme = THEMES[themeKey];
        }
    }

    loadMidiData(parsedData) {
        this.notes = parsedData.notes || [];
        this.pedals = parsedData.pedals || [];
        this.duration = parsedData.duration || 0;
        this.currentTime = 0;
        this.particles = [];
        this.keyboard.releaseAll();
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;

        this.canvas.width = Math.floor(rect.width * dpr);
        this.canvas.height = Math.floor(rect.height * dpr);

        this.ctx.resetTransform();
        this.ctx.scale(dpr, dpr);

        this.fallAreaHeight = this.height - this.keyboardHeight;
        this.keyboard.updateDimensions(this.width, this.keyboardHeight);
    }

    seek(timeSeconds) {
        this.currentTime = Math.max(0, Math.min(timeSeconds, this.duration));
        this.particles = [];
        this.keyboard.releaseAll();
    }

    update(currentTime) {
        this.currentTime = currentTime;
        this.keyboard.releaseAll();

        // Check active notes hitting keyboard right now
        for (const note of this.notes) {
            if (this.currentTime >= note.time && this.currentTime <= note.time + note.duration) {
                const colors = note.hand === 'left' ? this.theme.leftHand : this.theme.rightHand;
                this.keyboard.pressKey(note.midi, note.velocity, colors.glow);

                // Spawn spark particles on impact
                if (Math.abs(this.currentTime - note.time) < 0.05 && Math.random() < 0.4) {
                    const key = this.keyboard.getKey(note.midi);
                    if (key) {
                        const sparkX = key.x + (key.width / 2);
                        const sparkY = this.fallAreaHeight;
                        this.particles.push(new Particle(sparkX, sparkY, colors.glow));
                    }
                }
            }
        }

        // Check sustain pedal
        this.isPedalActive = false;
        for (const pedal of this.pedals) {
            if (this.currentTime >= pedal.time && this.currentTime <= pedal.time + pedal.duration) {
                this.isPedalActive = true;
                break;
            }
        }

        // Update particle physics
        for (let i = this.particles.length - 1; i >= 0; i--) {
            this.particles[i].update();
            if (this.particles[i].life <= 0) {
                this.particles.splice(i, 1);
            }
        }
    }

    render() {
        const ctx = this.ctx;
        const w = this.width;
        const h = this.height;

        // 1. Dark space background with subtle vertical vignette
        const bgGrad = ctx.createLinearGradient(0, 0, 0, this.fallAreaHeight);
        bgGrad.addColorStop(0, '#07080a');
        bgGrad.addColorStop(0.7, '#0c0e12');
        bgGrad.addColorStop(1, '#0e1117');
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, w, h);

        // 2. Subtle guideline lanes behind falling notes
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
        ctx.lineWidth = 1;
        for (const key of this.keyboard.keys) {
            if (!key.isBlack) {
                ctx.beginPath();
                ctx.moveTo(key.x, 0);
                ctx.lineTo(key.x, this.fallAreaHeight);
                ctx.stroke();
            }
        }

        // 3. Falling Notes
        // A note hits the keyboard (y = fallAreaHeight) exactly when currentTime == note.time
        // Note Y coordinate: fallAreaHeight - ((note.time - currentTime) * noteFallSpeed)
        const visibleWindowSec = (this.fallAreaHeight / this.noteFallSpeed) + 1;
        const minTime = this.currentTime - 0.5;
        const maxTime = this.currentTime + visibleWindowSec;

        for (const note of this.notes) {
            const noteEnd = note.time + note.duration;
            if (noteEnd < minTime || note.time > maxTime) continue;

            const key = this.keyboard.getKey(note.midi);
            if (!key) continue;

            const noteYBottom = this.fallAreaHeight - ((note.time - this.currentTime) * this.noteFallSpeed);
            const noteHeight = Math.max(12, note.duration * this.noteFallSpeed);
            const noteYTop = noteYBottom - noteHeight;

            // Clip notes so they do not protrude below the keyboard strike line
            const drawYTop = Math.max(0, noteYTop);
            const drawYBottom = Math.min(this.fallAreaHeight, noteYBottom);
            const drawHeight = drawYBottom - drawYTop;

            if (drawHeight <= 0) continue;

            const colors = note.hand === 'left' ? this.theme.leftHand : this.theme.rightHand;
            const x = key.x + 1.5;
            const noteWidth = key.width - 3;

            ctx.save();
            // Glow effect for struck notes
            const isNoteStriking = (this.currentTime >= note.time && this.currentTime <= noteEnd);
            if (isNoteStriking) {
                ctx.shadowColor = colors.glow;
                ctx.shadowBlur = 18;
            } else {
                ctx.shadowColor = 'rgba(0, 0, 0, 0.4)';
                ctx.shadowBlur = 6;
            }

            // Note pill gradient
            const noteGrad = ctx.createLinearGradient(x, drawYTop, x, drawYBottom);
            noteGrad.addColorStop(0, colors.top);
            noteGrad.addColorStop(1, colors.bottom);
            ctx.fillStyle = noteGrad;

            // Rounded pill shape
            const radius = Math.min(5, noteWidth / 2, drawHeight / 2);
            ctx.beginPath();
            ctx.roundRect(x, drawYTop, noteWidth, drawHeight, [radius, radius, radius, radius]);
            ctx.fill();

            // Specular highlight line along center
            ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
            ctx.fillRect(x + 2, drawYTop + 2, Math.max(1, noteWidth * 0.25), Math.max(2, drawHeight - 4));

            ctx.restore();
        }

        // 4. Keyboard Strike Line Reflection & Impact Glow
        ctx.save();
        const glowGrad = ctx.createLinearGradient(0, this.fallAreaHeight - 25, 0, this.fallAreaHeight);
        glowGrad.addColorStop(0, 'rgba(0, 198, 255, 0)');
        glowGrad.addColorStop(1, this.theme.glow);
        ctx.fillStyle = glowGrad;
        ctx.fillRect(0, this.fallAreaHeight - 25, w, 25);
        ctx.restore();

        // 5. Draw Impact Particles
        for (const p of this.particles) {
            p.draw(ctx);
        }

        // 6. Render 88-Key Keyboard at Bottom
        this.keyboard.render(ctx, this.fallAreaHeight, this.theme);

        // 7. Sustain Pedal Indicator Badge
        if (this.isPedalActive) {
            ctx.save();
            ctx.fillStyle = 'rgba(0, 198, 255, 0.85)';
            ctx.shadowColor = '#00c6ff';
            ctx.shadowBlur = 10;
            ctx.font = '600 11px Outfit, Inter, sans-serif';
            ctx.fillText('SUSTAIN PEDAL', 24, this.fallAreaHeight - 14);
            ctx.restore();
        }
    }
}
