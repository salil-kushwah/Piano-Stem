import { RousseauVisualizer, THEMES } from './visualizer.js';
import { AudioPlayer } from './audio-player.js';

class PianoStudioApp {
    constructor() {
        this.visualizer = null;
        this.player = new AudioPlayer();
        this.currentTaskId = null;
        this.currentData = null;
        this.ws = null;
        this.animationFrameId = null;

        this.initElements();
        this.initVisualizer();
        this.bindEvents();
        this.startRenderLoop();
    }

    initElements() {
        this.dropZone = document.getElementById('drop-zone');
        this.fileInput = document.getElementById('file-input');
        this.mixedCheckbox = document.getElementById('mixed-checkbox');

        this.processingOverlay = document.getElementById('processing-overlay');
        this.progressBar = document.getElementById('progress-bar');
        this.progressPercent = document.getElementById('progress-percent');
        this.progressMessage = document.getElementById('progress-message');

        this.studioControls = document.getElementById('studio-controls');
        this.playBtn = document.getElementById('play-btn');
        this.timeDisplay = document.getElementById('time-display');
        this.timeSlider = document.getElementById('time-slider');
        this.speedSelect = document.getElementById('speed-select');
        this.themeSelect = document.getElementById('theme-select');
        this.stemSelect = document.getElementById('stem-select');

        this.downloadMidiBtn = document.getElementById('download-midi-btn');
        this.downloadStemBtn = document.getElementById('download-stem-btn');
        this.newTrackBtn = document.getElementById('new-track-btn');
        this.demoTrackBtn = document.getElementById('demo-track-btn');

        this.loopABtn = document.getElementById('loop-a-btn');
        this.loopBBtn = document.getElementById('loop-b-btn');
        this.loopClearBtn = document.getElementById('loop-clear-btn');
    }

    initVisualizer() {
        const canvas = document.getElementById('visualizer-canvas');
        this.visualizer = new RousseauVisualizer(canvas);

        // Bind audio player callbacks
        this.player.onTimeUpdate = (time) => {
            if (!this.isUserScrubbing) {
                this.timeSlider.value = (time / (this.player.duration || 1)) * 100;
                this.updateTimeDisplay(time, this.player.duration);
            }
        };

        this.player.onStateChange = (isPlaying) => {
            this.playBtn.innerHTML = isPlaying
                ? `<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>`
                : `<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`;
        };
    }

    bindEvents() {
        // Drag and drop
        this.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropZone.classList.add('drag-active');
        });

        this.dropZone.addEventListener('dragleave', () => {
            this.dropZone.classList.remove('drag-active');
        });

        this.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropZone.classList.remove('drag-active');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                this.handleFileUpload(e.dataTransfer.files[0]);
            }
        });

        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                this.handleFileUpload(e.target.files[0]);
            }
        });

        // Demo track
        this.demoTrackBtn.addEventListener('click', () => {
            this.loadDemoTrack();
        });

        // Play/Pause
        this.playBtn.addEventListener('click', () => {
            this.player.toggle();
        });

        // Time scrubbing
        this.timeSlider.addEventListener('input', (e) => {
            this.isUserScrubbing = true;
            const targetTime = (parseFloat(e.target.value) / 100) * this.player.duration;
            this.visualizer.seek(targetTime);
            this.updateTimeDisplay(targetTime, this.player.duration);
        });

        this.timeSlider.addEventListener('change', (e) => {
            this.isUserScrubbing = false;
            const targetTime = (parseFloat(e.target.value) / 100) * this.player.duration;
            this.player.seek(targetTime);
            this.visualizer.seek(targetTime);
        });

        // Speed selector
        this.speedSelect.addEventListener('change', (e) => {
            const speed = parseFloat(e.target.value);
            this.player.setPlaybackRate(speed);
        });

        // Theme selector
        this.themeSelect.addEventListener('change', (e) => {
            this.visualizer.setTheme(e.target.value);
        });

        // Audio stem selector
        this.stemSelect.addEventListener('change', (e) => {
            const mode = e.target.value;
            if (!this.currentData) return;

            const wasPlaying = this.player.isPlaying;
            const curTime = this.player.currentTime;

            const getAudioFilename = (p) => p ? p.replace(/\\/g, '/').split('/').pop() : '';

            if (mode === 'pure_piano' && this.currentData.pure_piano_file) {
                this.player.loadAudioUrl(`/api/audio/${getAudioFilename(this.currentData.pure_piano_file)}`, this.player.duration);
            } else if (mode === 'isolated_stem') {
                this.player.loadAudioUrl(`/api/audio/${getAudioFilename(this.currentData.stem_audio_file)}`, this.player.duration);
            } else if (mode === 'original_mix') {
                this.player.loadAudioUrl(`/api/audio/${getAudioFilename(this.currentData.original_audio_file)}`, this.player.duration);
            }
            this.player.seek(curTime);
            if (wasPlaying) this.player.play();
        });

        // A-B Looping
        this.loopABtn.addEventListener('click', () => {
            this.player.setLoopA(this.player.currentTime);
            this.loopABtn.classList.add('active-loop');
            this.loopABtn.textContent = `A: ${this.formatTime(this.player.currentTime)}`;
        });

        this.loopBBtn.addEventListener('click', () => {
            this.player.setLoopB(this.player.currentTime);
            this.loopBBtn.classList.add('active-loop');
            this.loopBBtn.textContent = `B: ${this.formatTime(this.player.currentTime)}`;
        });

        this.loopClearBtn.addEventListener('click', () => {
            this.player.clearLoop();
            this.loopABtn.classList.remove('active-loop');
            this.loopBBtn.classList.remove('active-loop');
            this.loopABtn.textContent = 'Set [A]';
            this.loopBBtn.textContent = 'Set [B]';
        });

        // Downloads
        this.downloadMidiBtn.addEventListener('click', () => {
            if (this.currentData && this.currentData.midi_file) {
                const fn = this.currentData.midi_file.replace(/\\/g, '/').split('/').pop();
                window.location.href = `/api/midi/${fn}`;
            }
        });

        this.downloadStemBtn.addEventListener('click', () => {
            if (this.currentData && this.currentData.stem_audio_file) {
                const fn = this.currentData.stem_audio_file.replace(/\\/g, '/').split('/').pop();
                window.location.href = `/api/audio/${fn}`;
            }
        });

        // Load new track
        this.newTrackBtn.addEventListener('click', () => {
            this.player.pause();
            this.dropZone.style.display = 'flex';
            this.studioControls.style.display = 'none';
        });

        // Keyboard shortcuts
        window.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
                e.preventDefault();
                this.player.toggle();
            } else if (e.code === 'ArrowLeft') {
                e.preventDefault();
                this.player.seek(this.player.currentTime - 5);
            } else if (e.code === 'ArrowRight') {
                e.preventDefault();
                this.player.seek(this.player.currentTime + 5);
            }
        });
    }

    startRenderLoop() {
        const render = () => {
            if (this.player.isPlaying) {
                this.visualizer.update(this.player.currentTime);
            }
            this.visualizer.render();
            this.animationFrameId = requestAnimationFrame(render);
        };
        this.animationFrameId = requestAnimationFrame(render);
    }

    async handleFileUpload(file) {
        const isMixed = this.mixedCheckbox.checked;

        this.showProcessing("Uploading audio file...", 0);

        const formData = new FormData();
        formData.append("file", file);
        formData.append("is_mixed", isMixed);

        try {
            const resp = await fetch("/api/process", {
                method: "POST",
                body: formData
            });

            if (!resp.ok) {
                throw new Error("Failed to upload audio file.");
            }

            const data = await resp.json();
            this.currentTaskId = data.task_id;
            this.connectWebSocket(data.task_id, data.original_audio_file);
        } catch (err) {
            console.error(err);
            alert("Error: " + err.message);
            this.hideProcessing();
        }
    }

    connectWebSocket(taskId, originalAudioFile) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/progress/${taskId}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.progress !== undefined) {
                this.updateProcessing(msg.message, msg.progress);
            }

            if (msg.status === 'completed' && msg.result) {
                this.ws.close();
                this.currentData = {
                    ...msg.result,
                    original_audio_file: originalAudioFile
                };
                this.loadStudioData(this.currentData);
            } else if (msg.status === 'failed') {
                this.ws.close();
                alert("Processing failed: " + (msg.error || "Unknown error"));
                this.hideProcessing();
            }
        };

        this.ws.onerror = (e) => {
            console.warn("WebSocket error:", e);
        };
    }

    loadStudioData(data) {
        this.hideProcessing();
        this.dropZone.style.display = 'none';
        this.studioControls.style.display = 'flex';

        this.visualizer.loadMidiData(data.visualizer_data);

        // Load isolated piano stem
        const getAudioFilename = (p) => p ? p.replace(/\\/g, '/').split('/').pop() : '';
        const initialAudio = data.stem_audio_file || data.pure_piano_file;
        this.player.loadAudioUrl(`/api/audio/${getAudioFilename(initialAudio)}`, data.visualizer_data.duration);
        this.updateTimeDisplay(0, data.visualizer_data.duration);
        this.timeSlider.value = 0;

        // Auto-play
        setTimeout(() => this.player.play(), 300);
    }

    loadDemoTrack() {
        // Pre-packaged Beethoven Moonlight Sonata demo notes
        const demoNotes = this.generateDemoNotes();
        const demoData = {
            duration: 18.0,
            notes: demoNotes,
            pedals: [
                { time: 0.0, duration: 8.5 },
                { time: 8.6, duration: 9.4 }
            ]
        };

        this.dropZone.style.display = 'none';
        this.studioControls.style.display = 'flex';

        this.visualizer.loadMidiData(demoData);
        this.player.duration = 18.0;
        this.player.currentTime = 0;
        this.updateTimeDisplay(0, 18.0);

        // Custom ticker for demo since no MP3 file uploaded
        this.player.onTimeUpdate = (time) => {
            this.timeSlider.value = (time / 18.0) * 100;
            this.updateTimeDisplay(time, 18.0);
            this.visualizer.update(time);
        };

        let demoStart = null;
        const tickDemo = (timestamp) => {
            if (!this.player.isPlaying) {
                demoStart = null;
                return;
            }
            if (!demoStart) demoStart = timestamp - (this.player.currentTime * 1000);
            const elapsed = (timestamp - demoStart) / 1000;
            if (elapsed >= 18.0) {
                this.player.seek(0);
                demoStart = timestamp;
            } else {
                this.player.currentTime = elapsed;
                this.player.onTimeUpdate(elapsed);
            }
            requestAnimationFrame(tickDemo);
        };

        this.player.onStateChange = (playing) => {
            this.playBtn.innerHTML = playing
                ? `<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>`
                : `<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`;
            if (playing) requestAnimationFrame(tickDemo);
        };

        this.player.play();
    }

    generateDemoNotes() {
        // Beethoven Moonlight Sonata triplets
        const notes = [];
        const arpeggio = [49, 56, 61]; // C#3, G#3, C#4
        let t = 0.0;
        const noteDur = 0.45;
        const gap = 0.35;

        // Bass octave
        notes.push({ midi: 37, name: 'C#2', time: 0.0, duration: 8.0, velocity: 0.7, hand: 'left' });
        notes.push({ midi: 49, name: 'C#3', time: 0.0, duration: 8.0, velocity: 0.65, hand: 'left' });

        for (let i = 0; i < 24; i++) {
            const pitch = arpeggio[i % 3] + (i > 12 ? 3 : 0);
            notes.push({
                midi: pitch,
                name: 'Note',
                time: roundTime(t),
                duration: noteDur,
                velocity: 0.6 + (Math.sin(i) * 0.15),
                hand: 'right'
            });

            // Melody entrance
            if (i === 6) {
                notes.push({ midi: 68, name: 'G#4', time: roundTime(t), duration: 2.5, velocity: 0.9, hand: 'right' });
            }
            if (i === 15) {
                notes.push({ midi: 68, name: 'G#4', time: roundTime(t), duration: 2.2, velocity: 0.9, hand: 'right' });
            }
            if (i === 18) {
                notes.push({ midi: 69, name: 'A4', time: roundTime(t), duration: 1.8, velocity: 0.92, hand: 'right' });
            }
            t += gap;
        }

        // Measure 2 bass
        notes.push({ midi: 35, name: 'B1', time: 8.4, duration: 8.0, velocity: 0.7, hand: 'left' });
        notes.push({ midi: 47, name: 'B2', time: 8.4, duration: 8.0, velocity: 0.65, hand: 'left' });

        return notes;
    }

    showProcessing(message, percent = 0) {
        this.processingOverlay.style.display = 'flex';
        this.updateProcessing(message, percent);
    }

    updateProcessing(message, percent) {
        this.progressMessage.textContent = message;
        this.progressBar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
        this.progressPercent.textContent = `${percent}%`;
    }

    hideProcessing() {
        this.processingOverlay.style.display = 'none';
    }

    formatTime(sec) {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    }

    updateTimeDisplay(current, duration) {
        this.timeDisplay.textContent = `${this.formatTime(current)} / ${this.formatTime(duration || 0)}`;
    }
}

function roundTime(val) {
    return Math.round(val * 100) / 100;
}

window.addEventListener('DOMContentLoaded', () => {
    window.pianoStudio = new PianoStudioApp();
});
