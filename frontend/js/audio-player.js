/**
 * AudioPlayer: Synchronized audio and Web Audio API piano synthesizer.
 * Supports:
 * - Isolated piano WAV playback
 * - Built-in polyphonic acoustic synthesis
 * - Variable playback speed (0.5x to 1.5x)
 * - A-B Loop repetition for practicing
 */

export class AudioPlayer {
    constructor() {
        this.audioElement = new Audio();
        this.audioElement.preload = 'auto';

        this.audioContext = null;
        this.synthEnabled = false;

        this.playbackRate = 1.0;
        this.isPlaying = false;
        this.duration = 0;
        this.currentTime = 0;

        // A-B Looping
        this.loopA = null;
        this.loopB = null;

        // Callbacks
        this.onTimeUpdate = null;
        this.onStateChange = null;

        this.initAudioListeners();
    }

    initAudioListeners() {
        this.audioElement.addEventListener('timeupdate', () => {
            this.currentTime = this.audioElement.currentTime;
            this.checkLoop();
            if (this.onTimeUpdate) this.onTimeUpdate(this.currentTime);
        });

        this.audioElement.addEventListener('ended', () => {
            if (this.loopA !== null && this.loopB !== null) {
                this.seek(this.loopA);
                this.play();
            } else {
                this.isPlaying = false;
                if (this.onStateChange) this.onStateChange(false);
            }
        });

        this.audioElement.addEventListener('loadedmetadata', () => {
            this.duration = this.audioElement.duration || this.duration;
        });
    }

    ensureAudioContext() {
        if (!this.audioContext) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            this.audioContext = new AudioCtx();
        }
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }
    }

    loadAudioUrl(url, fallbackDuration = 0) {
        this.audioElement.src = url;
        this.duration = fallbackDuration;
        this.currentTime = 0;
        this.audioElement.playbackRate = this.playbackRate;
    }

    play() {
        this.ensureAudioContext();
        this.isPlaying = true;
        this.audioElement.play().catch(e => console.warn('Audio play prevented:', e));
        if (this.onStateChange) this.onStateChange(true);
    }

    pause() {
        this.isPlaying = false;
        this.audioElement.pause();
        if (this.onStateChange) this.onStateChange(false);
    }

    toggle() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    seek(timeSeconds) {
        this.currentTime = Math.max(0, Math.min(timeSeconds, this.duration));
        this.audioElement.currentTime = this.currentTime;
        if (this.onTimeUpdate) this.onTimeUpdate(this.currentTime);
    }

    setPlaybackRate(rate) {
        this.playbackRate = rate;
        this.audioElement.playbackRate = rate;
    }

    setLoopA(time) {
        this.loopA = time;
    }

    setLoopB(time) {
        this.loopB = time;
    }

    clearLoop() {
        this.loopA = null;
        this.loopB = null;
    }

    checkLoop() {
        if (this.loopA !== null && this.loopB !== null && this.loopB > this.loopA) {
            if (this.currentTime >= this.loopB) {
                this.seek(this.loopA);
            }
        }
    }

    /**
     * Synthesize piano note using Web Audio API (additive harmonics + ADSR)
     */
    playSynthNote(midiNumber, duration = 0.5, velocity = 0.8) {
        this.ensureAudioContext();
        const ctx = this.audioContext;
        const now = ctx.currentTime;
        const freq = 440 * Math.pow(2, (midiNumber - 69) / 12);

        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        // Realistic piano harmonic tone
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(freq, now);

        // ADSR Envelope
        const attack = 0.005;
        const decay = duration * 0.8;
        const peakGain = 0.25 * velocity;

        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(peakGain, now + attack);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + attack + decay);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start(now);
        osc.stop(now + attack + decay + 0.05);
    }
}
