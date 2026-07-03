import sys
import os
import time
import random
import bisect
from pathlib import Path
from typing import Dict, List, Any, Optional

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal, QThread, QRectF
from PyQt6.QtGui import (
    QPainter, QColor, QLinearGradient, QPen, QBrush,
    QFont, QDragEnterEvent, QDropEvent, QPixmap, QMouseEvent
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QSlider, QLabel, QComboBox,
    QProgressBar, QFileDialog, QFrame, QMessageBox, QSizePolicy,
    QDialog, QGridLayout
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from backend.config import STEMS_DIR, MIDI_DIR
from backend.pipeline import ProcessingPipeline
from backend.midi_parser import filter_and_export_midi
from backend.synth import synthesize_midi_to_audio
from backend.chord_namer import detect_chord_at_time

# Piano Layout Constants (88 Keys: A0 (21) to C8 (108))
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
IS_BLACK_KEY = [False, True, False, True, False, False, True, False, True, False, True, False]

# Visualizer Color Themes — Left Hand (Blue/Cyan) vs Right Hand (Pink/Purple) distinction
THEMES = {
    'Rousseau Classic': {
        'left_top': QColor(0, 140, 255),       # Vivid Azure Blue
        'left_bottom': QColor(0, 70, 200),     # Deep Royal Blue
        'right_top': QColor(255, 0, 128),      # Vivid Hot Pink
        'right_bottom': QColor(180, 0, 120),   # Deep Magenta
        'glow_left': QColor(0, 140, 255, 180),
        'glow_right': QColor(255, 0, 128, 180),
        'chord_glow': QColor(241, 196, 15, 140),  # Golden amber chord aura
        'accent': QColor(0, 229, 255),
        'active_key_left': QColor(150, 210, 255),   # Soft blue highlight
        'active_key_right': QColor(255, 170, 210),  # Soft pink highlight
    },
    'Neon Synthwave': {
        'left_top': QColor(0, 200, 255),       # Electric Cyan
        'left_bottom': QColor(0, 100, 220),    # Deep Electric Blue
        'right_top': QColor(255, 80, 200),     # Neon Pink
        'right_bottom': QColor(160, 30, 255),  # Vivid Purple
        'glow_left': QColor(0, 200, 255, 180),
        'glow_right': QColor(255, 80, 200, 180),
        'chord_glow': QColor(255, 215, 0, 140),
        'accent': QColor(255, 64, 160),
        'active_key_left': QColor(160, 230, 255),
        'active_key_right': QColor(255, 180, 230),
    },
    'Emerald & Gold': {
        'left_top': QColor(0, 180, 130),       # Rich Emerald
        'left_bottom': QColor(0, 120, 90),     # Deep Forest
        'right_top': QColor(255, 200, 50),     # Bright Gold
        'right_bottom': QColor(220, 150, 0),   # Deep Amber
        'glow_left': QColor(0, 200, 150, 180),
        'glow_right': QColor(255, 200, 50, 180),
        'chord_glow': QColor(255, 100, 50, 140),
        'accent': QColor(0, 230, 118),
        'active_key_left': QColor(167, 243, 208),
        'active_key_right': QColor(255, 235, 180),
    },
    'Midnight Fire': {
        'left_top': QColor(100, 120, 255),     # Indigo Blue
        'left_bottom': QColor(50, 50, 180),    # Deep Indigo
        'right_top': QColor(255, 90, 50),      # Bright Orange
        'right_bottom': QColor(200, 30, 30),   # Deep Red
        'glow_left': QColor(100, 120, 255, 180),
        'glow_right': QColor(255, 90, 50, 180),
        'chord_glow': QColor(255, 240, 100, 140),
        'accent': QColor(255, 140, 50),
        'active_key_left': QColor(180, 190, 255),
        'active_key_right': QColor(255, 180, 150),
    }
}


class SmoothAudioClock:
    """
    Sub-millisecond continuous monotonic audio clock interpolator.
    Eliminates Windows Media Foundation 100ms coarse time stepping,
    enabling perfectly fluid 60FPS / 120FPS falling tiles without lag or stutter.
    """
    def __init__(self):
        self.last_audio_pos = 0.0
        self.last_perf_time = time.perf_counter()
        self.playback_rate = 1.0
        self.is_playing = False

    def start(self, current_audio_pos: float, rate: float = 1.0):
        self.last_audio_pos = max(0.0, current_audio_pos)
        self.last_perf_time = time.perf_counter()
        self.playback_rate = rate
        self.is_playing = True

    def pause(self, current_audio_pos: float):
        self.last_audio_pos = max(0.0, current_audio_pos)
        self.is_playing = False

    def seek(self, target_pos: float):
        self.last_audio_pos = max(0.0, target_pos)
        self.last_perf_time = time.perf_counter()

    def set_rate(self, rate: float):
        if self.is_playing:
            self.last_audio_pos = self.get_time()
            self.last_perf_time = time.perf_counter()
        self.playback_rate = rate

    def sync_hardware_clock(self, player_pos: float):
        if not self.is_playing:
            self.last_audio_pos = player_pos
            return

        estimated = self.get_time()
        drift = player_pos - estimated
        # If drift is large (> 0.08s, e.g. track scrub, loop wrap, or pipeline pause), re-anchor
        if abs(drift) > 0.08:
            self.last_audio_pos = player_pos
            self.last_perf_time = time.perf_counter()
        elif abs(drift) > 0.015:
            # Gentle continuous phase lock
            self.last_audio_pos += drift * 0.1

    def get_time(self) -> float:
        if not self.is_playing:
            return self.last_audio_pos
        now = time.perf_counter()
        return max(0.0, self.last_audio_pos + (now - self.last_perf_time) * self.playback_rate)


class SparkParticle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'life', 'decay', 'color')
    def __init__(self, x: float, y: float, color: QColor):
        self.x = x
        self.y = y
        self.vx = (random.random() - 0.5) * 5.0
        self.vy = -random.random() * 4.5 - 2.0
        self.life = 1.0
        self.decay = random.random() * 0.05 + 0.04
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.22  # gravity
        self.life -= self.decay


class RousseauVisualizerWidget(QWidget):
    note_clicked = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(380)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        self.all_notes: List[Dict[str, Any]] = []
        self.notes: List[Dict[str, Any]] = []
        self.note_start_times: List[float] = []
        self.pedals: List[Dict[str, Any]] = []
        self.duration = 0.0
        self.current_time = 0.0
        self.velocity_threshold = 40.0

        self.diagnostic_mode = False
        self.selected_note_id: Optional[str] = None

        self.theme_name = 'Rousseau Classic'
        self.theme = THEMES[self.theme_name]

        self.note_fall_speed = 380.0  # pixels / second
        self.keyboard_height = 125
        self.active_keys: Dict[int, float] = {}  # midi -> velocity
        self.is_pedal_active = False
        self.show_fingering = True
        self.show_measure_lines = True
        self.tempo_bpm = 120.0
        self.beat_interval = 0.5
        self.beats_per_measure = 4
        self.key_name = "C Major"
        self.current_chord_info: Optional[Dict[str, Any]] = None
        self.chord_fade_alpha = 0.0
        self.particles: List[SparkParticle] = []

        # 88-Key geometry cache
        self.start_midi = 21  # A0
        self.end_midi = 108   # C8
        self.total_white_keys = 52
        self.white_keys_meta = []
        self.black_keys_meta = []
        self._init_keys_meta()

        # Fast lookup geometry cache (rebuilt on resize)
        self.key_geom: Dict[int, Any] = {}
        self.key_center_x: Dict[int, float] = {}
        self.cached_keyboard_pixmap: Optional[QPixmap] = None

    def _init_keys_meta(self):
        white_idx = 0
        for midi in range(self.start_midi, self.end_midi + 1):
            is_black = IS_BLACK_KEY[midi % 12]
            name = f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"
            if not is_black:
                self.white_keys_meta.append({'midi': midi, 'name': name, 'white_idx': white_idx})
                white_idx += 1
            else:
                self.black_keys_meta.append({'midi': midi, 'name': name, 'prev_white_idx': white_idx - 1})

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rebuild_geometry_cache()

    def _rebuild_geometry_cache(self):
        w = max(100, self.width())
        h = max(100, self.height())
        fall_area = h - self.keyboard_height
        white_w = w / self.total_white_keys
        black_w = white_w * 0.62
        black_h = self.keyboard_height * 0.64

        self.key_geom.clear()
        self.key_center_x.clear()

        # Precompute white keys
        for wk in self.white_keys_meta:
            midi = wk['midi']
            x = wk['white_idx'] * white_w
            nx = x + 1.5
            nw = white_w - 3.0
            cx = x + (white_w / 2.0)
            self.key_geom[midi] = {
                'is_black': False,
                'x': x,
                'nx': nx,
                'nw': nw,
                'w': white_w,
                'h': self.keyboard_height,
                'center_x': cx
            }
            self.key_center_x[midi] = cx

        # Precompute black keys
        for bk in self.black_keys_meta:
            midi = bk['midi']
            x = (bk['prev_white_idx'] + 1) * white_w - (black_w / 2.0)
            nx = x + 1.0
            nw = black_w - 2.0
            cx = x + (black_w / 2.0)
            self.key_geom[midi] = {
                'is_black': True,
                'x': x,
                'nx': nx,
                'nw': nw,
                'w': black_w,
                'h': black_h,
                'center_x': cx
            }
            self.key_center_x[midi] = cx

        self._render_keyboard_pixmap()

    def _render_keyboard_pixmap(self):
        w = max(100, self.width())
        kh = self.keyboard_height
        pixmap = QPixmap(w, kh)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. White Keys (Idle)
        for wk in self.white_keys_meta:
            meta = self.key_geom.get(wk['midi'])
            if not meta:
                continue
            x = meta['x']
            kw = meta['w']

            grad = QLinearGradient(x, 0, x, kh)
            grad.setColorAt(0.0, QColor(248, 250, 252))
            grad.setColorAt(0.9, QColor(220, 224, 230))
            grad.setColorAt(1.0, QColor(140, 140, 145))

            painter.setPen(QPen(QColor(30, 34, 42), 1))
            painter.setBrush(QBrush(grad))
            painter.drawRoundedRect(int(x), 0, int(kw), int(kh), 0, 0)

        # 2. Black Keys (Idle)
        for bk in self.black_keys_meta:
            meta = self.key_geom.get(bk['midi'])
            if not meta:
                continue
            x = meta['x']
            kw = meta['w']
            bh = meta['h']

            grad = QLinearGradient(x, 0, x, bh)
            grad.setColorAt(0.0, QColor(25, 27, 31))
            grad.setColorAt(0.85, QColor(14, 15, 18))
            grad.setColorAt(1.0, QColor(5, 5, 5))

            painter.setPen(QPen(QColor(10, 10, 12), 1))
            painter.setBrush(QBrush(grad))
            painter.drawRoundedRect(int(x), 0, int(kw), int(bh), 2, 2)

        # 3. Red felt strip along top
        painter.fillRect(0, 0, w, 3, QColor(180, 20, 30))
        painter.end()

        self.cached_keyboard_pixmap = pixmap

    def set_theme(self, name: str):
        if name in THEMES:
            self.theme_name = name
            self.theme = THEMES[name]
            self.update()

    def load_data(self, data: Dict[str, Any], min_velocity: float = 40.0, max_duration: Optional[float] = 1.2):
        self.all_notes = data.get("notes", [])
        self.pedals = data.get("pedals", [])
        self.duration = data.get("duration", 0.0)
        self.key_name = data.get("key_signature", "C Major")
        self.tempo_bpm = data.get("tempo_bpm", 120.0)
        self.beat_interval = data.get("beat_interval", 60.0 / max(30.0, self.tempo_bpm))
        self.current_chord_info = None
        self.chord_fade_alpha = 0.0
        self.current_time = 0.0
        self.active_keys.clear()
        self.particles.clear()
        self.velocity_threshold = min_velocity
        self.duration_clamp = max_duration if (max_duration is not None and max_duration > 0) else None
        self._refresh_notes()

    def set_velocity_gate(self, min_vel: float) -> int:
        self.velocity_threshold = min_vel
        return self._refresh_notes()

    def set_duration_clamp(self, max_dur: float) -> int:
        self.duration_clamp = max_dur if max_dur > 0 else None
        return self._refresh_notes()

    def _refresh_notes(self) -> int:
        filtered = []
        for n in self.all_notes:
            vel_raw = n.get("velocity", 0.8)
            vel = vel_raw if vel_raw > 1.0 else (vel_raw * 127.0)
            if vel >= self.velocity_threshold:
                note_copy = dict(n)
                if self.duration_clamp is not None and self.duration_clamp > 0:
                    note_copy["duration"] = min(note_copy["duration"], self.duration_clamp)
                filtered.append(note_copy)

        filtered.sort(key=lambda x: x["time"])
        self.notes = filtered
        self.note_start_times = [n["time"] for n in self.notes]
        self.max_note_dur = max((n["duration"] for n in self.notes), default=2.0)
        self.update()
        return len(self.notes)

    def set_time(self, time_sec: float):
        self.current_time = max(0.0, min(time_sec, self.duration if self.duration > 0 else 9999.0))
        self.active_keys.clear()
        self.active_key_hands = {}  # midi -> hand string (for O(1) lookup in paint)
        fall_area = self.height() - self.keyboard_height

        # Fast localized active notes scan around current_time
        if self.notes:
            idx = bisect.bisect_right(self.note_start_times, self.current_time)
            # Scan backwards for notes that started before current_time but might still be sounding
            max_lookback = getattr(self, 'max_note_dur', 4.0) + 0.1
            start_scan = max(0, bisect.bisect_left(self.note_start_times, self.current_time - max_lookback))
            end_scan = min(len(self.notes), idx + 2)

            for i in range(start_scan, end_scan):
                n = self.notes[i]
                n_start = n["time"]
                n_end = n_start + n["duration"]
                if n_start <= self.current_time <= n_end:
                    midi = n["midi"]
                    self.active_keys[midi] = n.get("velocity", 0.8)
                    self.active_key_hands[midi] = n.get("hand", "right")

                    # Spawn spark particle on note onset
                    if abs(self.current_time - n_start) < 0.035 and random.random() < 0.35:
                        cx = self.key_center_x.get(midi, self.width() / 2.0)
                        self.particles.append(SparkParticle(cx, fall_area, self.theme['accent']))

        # Real-time Rolling Harmonic Chord Recognition
        if len(self.active_keys) >= 2:
            chord = detect_chord_at_time(self.notes, self.current_time, window_back=0.55, key_name=self.key_name)
            if chord:
                self.current_chord_info = chord
                self.chord_fade_alpha = min(1.0, self.chord_fade_alpha + 0.25)
            else:
                self.chord_fade_alpha = max(0.0, self.chord_fade_alpha - 0.08)
        else:
            self.chord_fade_alpha = max(0.0, self.chord_fade_alpha - (0.12 if len(self.active_keys) == 0 else 0.06))
            if self.chord_fade_alpha <= 0.0:
                self.current_chord_info = None

        # Pedal detection (binary search for O(log n) instead of linear scan)
        self.is_pedal_active = False
        for p in self.pedals:
            if p["time"] <= self.current_time <= (p["time"] + p["duration"]):
                self.is_pedal_active = True
                break
            if p["time"] > self.current_time:
                break  # pedals are sorted by time

        # Particle decay — O(n) list comprehension instead of O(n²) remove-in-loop
        if self.particles:
            for p in self.particles:
                p.update()
            self.particles = [p for p in self.particles if p.life > 0]

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        fall_area = h - self.keyboard_height

        # 1. Background
        bg_grad = QLinearGradient(0, 0, 0, fall_area)
        bg_grad.setColorAt(0.0, QColor(7, 8, 10))
        bg_grad.setColorAt(1.0, QColor(14, 17, 23))
        painter.fillRect(0, 0, w, h, bg_grad)

        # 2. Key Lanes
        pen_lane = QPen(QColor(255, 255, 255, 7), 1)
        painter.setPen(pen_lane)
        for wk in self.white_keys_meta:
            meta = self.key_geom.get(wk['midi'])
            if meta:
                x = meta['x']
                painter.drawLine(int(x), 0, int(x), int(fall_area))

        # 2b. Horizontal Measure Bar Lines & Beat Subdivisions (Synthesia & Rousseau Style)
        if self.show_measure_lines and self.beat_interval > 0.05:
            visible_sec = (fall_area / self.note_fall_speed) + 0.5
            min_t = self.current_time - 0.2
            max_t = self.current_time + visible_sec

            b_start = max(0, int(min_t / self.beat_interval))
            b_end = int((max_t / self.beat_interval) + 1)

            for b in range(b_start, b_end):
                t_beat = b * self.beat_interval
                y = fall_area - (t_beat - self.current_time) * self.note_fall_speed
                if 0 <= y <= fall_area:
                    is_measure = (b % self.beats_per_measure == 0)
                    if is_measure:
                        m_num = (b // self.beats_per_measure) + 1
                        # Distinct Measure Bar Line (Subtle Cyan Glow)
                        painter.setPen(QPen(QColor(0, 198, 255, 50), 1.5, Qt.PenStyle.SolidLine))
                        painter.drawLine(0, int(y), int(w), int(y))

                        # Measure Counter Badge on left margin
                        painter.setFont(QFont("Outfit", 8, QFont.Weight.Bold))
                        painter.setPen(QColor(0, 198, 255, 140))
                        painter.drawText(8, int(y - 3), f"m.{m_num}")
                    else:
                        # Faint Beat Subdivision Line
                        painter.setPen(QPen(QColor(255, 255, 255, 12), 1, Qt.PenStyle.DashLine))
                        painter.drawLine(0, int(y), int(w), int(y))

        # 3. Falling Notes (Ultra-smooth binary search range)
        visible_sec = (fall_area / self.note_fall_speed) + 0.5
        min_t = self.current_time - 0.2
        max_t = self.current_time + visible_sec

        if self.notes:
            # Find first note that could be on screen
            lookback = max(4.0, getattr(self, 'max_note_dur', 4.0) + 0.5)
            start_idx = max(0, bisect.bisect_left(self.note_start_times, min_t - lookback))

            for i in range(start_idx, len(self.notes)):
                n = self.notes[i]
                n_start = n["time"]
                n_end = n_start + n["duration"]

                if n_start > max_t:
                    break
                if n_end < min_t:
                    continue

                y_bottom = fall_area - (n_start - self.current_time) * self.note_fall_speed
                y_top = fall_area - (n_end - self.current_time) * self.note_fall_speed
                note_h = max(6.0, y_bottom - y_top)

                if y_top > fall_area or y_bottom < 0:
                    continue

                midi = n["midi"]
                geom = self.key_geom.get(midi)
                if not geom:
                    continue

                nx = geom['nx']
                nw = geom['nw']

                hand = n.get("hand", "right")
                is_chord = n.get("is_chord", False)
                status = n.get("status", "accepted")

                if self.diagnostic_mode and status == "flagged":
                    top_color = QColor(255, 140, 0)
                    bot_color = QColor(190, 60, 0)
                else:
                    top_color = self.theme['left_top'] if hand == "left" else self.theme['right_top']
                    bot_color = self.theme['left_bottom'] if hand == "left" else self.theme['right_bottom']

                note_grad = QLinearGradient(nx, y_top, nx, y_bottom)
                note_grad.setColorAt(0.0, top_color)
                note_grad.setColorAt(1.0, bot_color)

                is_selected = (n.get("id") == self.selected_note_id)
                if is_selected:
                    painter.setPen(QPen(QColor(255, 255, 255, 250), 2.5))
                elif self.diagnostic_mode and status == "flagged":
                    painter.setPen(QPen(QColor(255, 220, 0, 220), 1.5, Qt.PenStyle.DashLine))
                elif is_chord:
                    chord_color = self.theme['chord_glow']
                    painter.setPen(QPen(chord_color, 2.0))
                else:
                    painter.setPen(Qt.PenStyle.NoPen)

                painter.setBrush(QBrush(note_grad))
                painter.drawRoundedRect(int(nx), int(y_top), int(nw), int(note_h), 4, 4)

                # Draw finger number badge if enabled
                finger = n.get("finger")
                if self.show_fingering and finger and note_h >= 14:
                    badge_sz = min(13, int(nw * 0.72))
                    if badge_sz >= 9:
                        bx = int(nx + (nw - badge_sz) / 2)
                        by = int(y_bottom - badge_sz - 2)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
                        painter.drawEllipse(bx, by, badge_sz, badge_sz)
                        painter.setPen(QPen(QColor(255, 255, 255, 240)))
                        f = painter.font()
                        f.setPixelSize(max(7, badge_sz - 3))
                        f.setBold(True)
                        painter.setFont(f)
                        painter.drawText(bx, by, badge_sz, badge_sz, Qt.AlignmentFlag.AlignCenter, str(finger))

        # 4. Particles
        for p in self.particles:
            alpha = max(0, min(255, int(p.life * 255)))
            p_color = QColor(p.color.red(), p.color.green(), p.color.blue(), alpha)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(p_color))
            painter.drawEllipse(int(p.x - 2), int(p.y - 2), 4, 4)

        # 5. Draw Pre-cached Static Keyboard (Single blit: ~0.05ms)
        if self.cached_keyboard_pixmap:
            painter.drawPixmap(0, int(fall_area), self.cached_keyboard_pixmap)

        # 5b. Sustain Pedal Glow Bar
        if self.is_pedal_active:
            pedal_glow = QLinearGradient(0, fall_area - 6, 0, fall_area)
            pedal_glow.setColorAt(0.0, QColor(241, 196, 15, 0))
            pedal_glow.setColorAt(1.0, QColor(241, 196, 15, 160))
            painter.fillRect(0, int(fall_area - 6), w, 6, QBrush(pedal_glow))
            painter.setPen(QPen(QColor(241, 196, 15, 220)))
            f_pedal = painter.font()
            f_pedal.setPixelSize(10)
            f_pedal.setBold(True)
            painter.setFont(f_pedal)
            painter.drawText(w - 140, int(h - 8), "[ SUSTAIN PEDAL ]")

        # 6. Active Key Overlays & Glows (Hand-Colored)
        # Uses pre-built active_key_hands dict from set_time() for O(1) hand lookup
        active_key_hands = getattr(self, 'active_key_hands', {})
        for midi, vel in self.active_keys.items():
            geom = self.key_geom.get(midi)
            if not geom:
                continue

            x = geom['x']
            kw = geom['w']
            is_black = geom['is_black']
            kh = geom['h']

            # O(1) hand lookup from pre-built dict
            key_hand = active_key_hands.get(midi, 'right')

            # Glowing Aura above keyboard (hand-colored)
            glow_h = 35
            glow_grad = QLinearGradient(x, fall_area - glow_h, x, fall_area)
            glow_color = self.theme.get('glow_left' if key_hand == 'left' else 'glow_right',
                                        self.theme.get('glow_left', QColor(0, 198, 255, 180)))
            glow_grad.setColorAt(0.0, QColor(glow_color.red(), glow_color.green(), glow_color.blue(), 0))
            glow_grad.setColorAt(1.0, glow_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow_grad))
            painter.drawRect(int(x), int(fall_area - glow_h), int(kw), glow_h)

            # Key Face Pressed Highlight (hand-colored)
            if is_black:
                key_color = self.theme['left_top'] if key_hand == 'left' else self.theme['right_top']
                press_grad = QLinearGradient(x, fall_area, x, fall_area + kh)
                press_grad.setColorAt(0.0, key_color)
                press_grad.setColorAt(0.85, key_color.darker(130))
                press_grad.setColorAt(1.0, QColor(10, 10, 12))
                painter.setPen(QPen(QColor(10, 10, 12), 1))
                painter.setBrush(QBrush(press_grad))
                painter.drawRoundedRect(int(x), int(fall_area), int(kw), int(kh), 2, 2)
            else:
                key_color = self.theme.get('active_key_left' if key_hand == 'left' else 'active_key_right',
                                           self.theme.get('active_key_left', QColor(180, 240, 255)))
                press_grad = QLinearGradient(x, fall_area, x, h)
                press_grad.setColorAt(0.0, key_color)
                press_grad.setColorAt(0.9, key_color.darker(110))
                press_grad.setColorAt(1.0, QColor(140, 140, 145))
                painter.setPen(QPen(QColor(40, 44, 52), 1))
                painter.setBrush(QBrush(press_grad))
                painter.drawRoundedRect(int(x), int(fall_area), int(kw), int(self.keyboard_height), 0, 0)

        # 7. Sustain Pedal Indicator
        if self.is_pedal_active:
            painter.setFont(QFont("Outfit", 9, QFont.Weight.Bold))
            painter.setPen(self.theme['accent'])
            painter.drawText(20, int(fall_area - 12), "SUSTAIN PEDAL [ON]")

        # 8. Floating Harmony HUD Pill (Glassmorphic Broadcast Overlay)
        if self.current_chord_info and self.chord_fade_alpha > 0.02:
            alpha = int(255 * self.chord_fade_alpha)
            sym = self.current_chord_info.get("symbol", "")
            root = self.current_chord_info.get("root", "")
            qual = self.current_chord_info.get("quality", "")
            is_inv = self.current_chord_info.get("is_inversion", False)

            pill_w = 175
            pill_h = 38
            pill_x = int((w - pill_w) / 2)
            pill_y = 16

            # Glassmorphic rounded pill container
            hud_bg = QColor(10, 14, 22, int(220 * self.chord_fade_alpha))
            hud_border = QColor(0, 198, 255, int(180 * self.chord_fade_alpha))
            painter.setPen(QPen(hud_border, 1.5))
            painter.setBrush(QBrush(hud_bg))
            painter.drawRoundedRect(pill_x, pill_y, pill_w, pill_h, 19, 19)

            # Golden musical note icon
            painter.setFont(QFont("Outfit", 12, QFont.Weight.Bold))
            painter.setPen(QColor(241, 196, 15, alpha))
            painter.drawText(pill_x + 14, pill_y, 22, pill_h, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "♬")

            # Main Chord Symbol Text (Vivid White Glow)
            painter.setFont(QFont("Outfit", 13, QFont.Weight.ExtraBold))
            painter.setPen(QColor(255, 255, 255, alpha))
            painter.drawText(pill_x + 36, pill_y, pill_w - 48, pill_h, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, sym)

            # Subtitle Quality Tag
            if is_inv:
                tag_text = "Inversion"
            elif "m" in qual and "maj" not in qual:
                tag_text = "Minor"
            elif "maj" in qual or qual == "":
                tag_text = "Major"
            else:
                tag_text = qual.upper()

            painter.setFont(QFont("Outfit", 8, QFont.Weight.DemiBold))
            painter.setPen(QColor(0, 198, 255, int(alpha * 0.85)))
            painter.drawText(pill_x, pill_y, pill_w - 14, pill_h, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, tag_text)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
            fall_area = self.height() - self.keyboard_height
            if y < fall_area and self.notes:
                # Find clicked note on waterfall
                for n in reversed(self.notes):
                    n_start = n["time"]
                    n_end = n_start + n["duration"]
                    y_bottom = fall_area - (n_start - self.current_time) * self.note_fall_speed
                    y_top = fall_area - (n_end - self.current_time) * self.note_fall_speed
                    geom = self.key_geom.get(n["midi"])
                    if geom:
                        nx = geom['nx']
                        nw = geom['nw']
                        if (nx - 2) <= x <= (nx + nw + 2) and (y_top - 4) <= y <= (y_bottom + 4):
                            self.selected_note_id = n.get("id")
                            self.update()
                            self.note_clicked.emit(n)
                            return
        super().mousePressEvent(event)


class NoteInspectorDialog(QDialog):
    note_deleted = pyqtSignal(dict)
    note_hand_toggled = pyqtSignal(dict)

    def __init__(self, note: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.note = note
        self.setWindowTitle(f"Note Inspector — {note.get('name', 'Note')}")
        self.setFixedSize(430, 380)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("""
            QDialog {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 8px;
            }
            QLabel {
                color: #e6edf3;
            }
            QPushButton {
                background: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                padding: 6px 12px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #30363d;
                border-color: #8b949e;
            }
            QPushButton#DeleteBtn {
                background: #da3633;
                color: #ffffff;
                border: 1px solid #f85149;
            }
            QPushButton#DeleteBtn:hover {
                background: #b62324;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        # Header Badge
        header = QHBoxLayout()
        name_lbl = QLabel(f"🎹 {note.get('name', '')} (MIDI {note.get('midi', '')})")
        name_lbl.setStyleSheet("font-size: 1.15rem; font-weight: bold; color: #58a6ff;")
        header.addWidget(name_lbl)
        header.addStretch()

        status = note.get("status", "accepted")
        status_color = "#3fb950" if status == "accepted" else "#f0883e"
        status_text = "✓ Verified Real Strike" if status == "accepted" else "⚠ Flagged Ghost"
        badge = QLabel(status_text)
        badge.setStyleSheet(f"background: {status_color}22; color: {status_color}; border: 1px solid {status_color}; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.75rem;")
        header.addWidget(badge)
        layout.addLayout(header)

        # Details Grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)

        def add_row(r, label, val, color="#c9d1d9"):
            l = QLabel(label)
            l.setStyleSheet("color: #8b949e; font-size: 0.82rem;")
            v = QLabel(str(val))
            v.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 0.85rem;")
            grid.addWidget(l, r, 0)
            grid.addWidget(v, r, 1)

        onset_p = note.get("onset_prob", 0.0) * 100
        frame_p = note.get("frame_prob", 0.0) * 100
        dur = note.get("duration", 0.0)
        t = note.get("time", 0.0)
        vel = int(note.get("velocity", 0.0) * 127) if note.get("velocity", 0.0) <= 1.0 else int(note.get("velocity", 0.0))
        hand = note.get("hand", "unknown").capitalize()
        finger = note.get("finger", "Auto")

        add_row(0, "Onset Time / Duration:", f"{t:.3f}s / {dur:.3f}s")
        add_row(1, "Neural Onset Confidence:", f"{onset_p:.1f}%", "#3fb950" if onset_p > 70 else "#f0883e")
        add_row(2, "Frame Sustain Presence:", f"{frame_p:.1f}%")
        add_row(3, "Velocity Dynamics:", f"{vel} / 127")
        add_row(4, "Pedagogical Hand / Finger:", f"{hand} Hand • Finger {finger}")
        add_row(5, "Harmony / Chord:", f"{note.get('chord_symbol', 'None')}")

        ev = note.get("evidence", {})
        row_idx = 6
        if "vocal_bleed" in ev:
            vb = ev["vocal_bleed"]
            v_delta = vb.get("vocal_peak_dbfs", 0) - vb.get("piano_peak_dbfs", 0)
            add_row(row_idx, "Vocal Bleed Contrast:", f"{v_delta:+.1f} dB", "#f0883e" if v_delta > 0 else "#3fb950")
            row_idx += 1
        if "drum_bleed" in ev:
            db = ev["drum_bleed"]
            d_delta = db.get("drum_peak_dbfs", 0) - db.get("piano_peak_dbfs", 0)
            add_row(row_idx, "Drum Bleed Contrast:", f"{d_delta:+.1f} dB", "#f0883e" if d_delta > 0 else "#3fb950")
            row_idx += 1
        if "bass_bleed" in ev:
            bb = ev["bass_bleed"]
            b_delta = bb.get("bass_peak_dbfs", 0) - bb.get("piano_peak_dbfs", 0)
            add_row(row_idx, "Bass Bleed Contrast:", f"{b_delta:+.1f} dB", "#f0883e" if b_delta > 0 else "#3fb950")
            row_idx += 1
        if "treble_comb_contrast" in ev:
            add_row(row_idx, "Treble Comb Contrast:", f"{ev['treble_comb_contrast']:+.1f} dB", "#3fb950" if ev['treble_comb_contrast'] > 2.0 else "#f0883e")
            row_idx += 1

        layout.addLayout(grid)

        # Flag Reason if present
        reason = note.get("rejection_reason")
        if reason:
            reason_box = QLabel(f"<b>Acoustic Evidence Flag:</b> {reason}")
            reason_box.setWordWrap(True)
            reason_box.setStyleSheet("background: rgba(240, 136, 62, 0.15); border: 1px solid rgba(240, 136, 62, 0.35); padding: 6px 10px; border-radius: 6px; color: #ffab70; font-size: 0.78rem;")
            layout.addWidget(reason_box)

        layout.addStretch()

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_del = QPushButton("🗑️ Delete False Note")
        btn_del.setObjectName("DeleteBtn")
        btn_del.setToolTip("Remove this note from the transcription and visualizer")
        btn_del.clicked.connect(self._on_delete)
        btn_layout.addWidget(btn_del)

        btn_hand = QPushButton("✋ Toggle Hand")
        btn_hand.setToolTip("Switch note between Left and Right hand")
        btn_hand.clicked.connect(self._on_toggle_hand)
        btn_layout.addWidget(btn_hand)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

    def _on_delete(self):
        self.note_deleted.emit(self.note)
        self.accept()

    def _on_toggle_hand(self):
        self.note_hand_toggled.emit(self.note)
        self.accept()


class ProcessingWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict)
    failed_signal = pyqtSignal(str)

    def __init__(self, audio_path: str, stem_mode: str = "none", precision_mode: str = "clean", model_engine: str = "studio_master"):
        super().__init__()
        self.audio_path = Path(audio_path)
        self.stem_mode = stem_mode
        self.precision_mode = precision_mode
        self.model_engine = model_engine

    def run(self):
        try:
            task_id = self.audio_path.stem[:8]
            pipeline = ProcessingPipeline(
                task_id=task_id,
                progress_callback=lambda p, msg: self.progress_signal.emit(p, msg)
            )
            result = pipeline.run(
                self.audio_path,
                is_mixed_audio=(self.stem_mode != "none"),
                precision_mode=self.precision_mode,
                model_engine=self.model_engine,
                stem_mode=self.stem_mode
            )
            self.finished_signal.emit(result)
        except Exception as e:
            self.failed_signal.emit(str(e))


class PianoStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Piano Stem Studio (RTX Accelerated)")
        self.resize(1280, 820)
        self.setMinimumSize(960, 640)
        self.setAcceptDrops(True)

        self.current_data = None
        self.current_audio_path = None
        self.loop_a = None
        self.loop_b = None
        self.max_duration_clamp: Optional[float] = 1.2

        # Audio Setup & Sub-millisecond Monotonic Clock
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        self.clock = SmoothAudioClock()
        self.player.errorOccurred.connect(self._on_media_error)

        # 60FPS High-Precision Animation Timer
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)  # 60 FPS
        self.timer.timeout.connect(self.on_timer_tick)

        self.init_ui()
        self.apply_dark_style()

    def _on_media_error(self, error, msg: str = ""):
        """Handle media playback errors with user-friendly messages."""
        error_messages = {
            QMediaPlayer.Error.ResourceError: "Could not open the audio file. It may be corrupted or in use.",
            QMediaPlayer.Error.FormatError: "Unsupported audio format. Try converting to WAV or MP3.",
            QMediaPlayer.Error.AccessDeniedError: "Access denied to the audio file. Check file permissions.",
        }
        friendly_msg = error_messages.get(error, f"Media playback error: {msg}")
        print(f"[Media Player] Error: {friendly_msg}")
        QMessageBox.warning(self, "Audio Playback Error", friendly_msg)

    def init_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # -------------------------------------------------------------
        # 1. Header Toolbar (Parented strictly to central to avoid fullscreen shift)
        # -------------------------------------------------------------
        header = QFrame(central)
        header.setFixedHeight(56)
        header.setObjectName("HeaderFrame")
        header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(16, 0, 16, 0)
        h_layout.setSpacing(10)

        title_label = QLabel("Piano Stem Studio")
        title_label.setObjectName("TitleLabel")
        h_layout.addWidget(title_label)

        badge = QLabel("RTX 3050")
        badge.setObjectName("BadgeLabel")
        h_layout.addWidget(badge)

        self.key_badge = QLabel("🎼 Key: Auto")
        self.key_badge.setStyleSheet("color: #00c6ff; background: rgba(0, 198, 255, 0.12); padding: 4px 10px; border-radius: 6px; font-weight: bold; border: 1px solid rgba(0, 198, 255, 0.3);")
        h_layout.addWidget(self.key_badge)

        h_layout.addStretch()

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(THEMES.keys()))
        self.theme_combo.setMaximumWidth(165)
        self.theme_combo.setToolTip("Change visualizer color palette")
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        h_layout.addWidget(self.theme_combo)

        # Stem Isolation Mode Selector (BS-RoFormer-SW vs HTDemucs 6S vs UVR MDX-Net vs Direct Solo)
        self.stem_combo = QComboBox()
        self.stem_combo.addItems([
            "✨ Auto-Detect Mix Profile (Recommended)",
            "🎸 Regime A: Full Band Mix (BS-RoFormer 6S + Bleed Guard)",
            "🎙️ Regime B: Piano & Vocal Duet (UVR InstVoc — Zero Cannibalization)",
            "🎛️ Legacy: HTDemucs 6S",
            "⚡ Solo Piano (Direct Acoustic — No Demix)"
        ])
        self.stem_combo.setMaximumWidth(325)
        self.stem_combo.setToolTip(
            "Isolation Engine:\n"
            "- ✨ Auto-Detect Mix Profile: Analyzes acoustic spectrum, rhythm groove, and vocal sibilance to select the optimal pipeline automatically.\n"
            "- 🎸 Regime A: Full Band Mix: 6-stem BS-RoFormer-SW separation + Vocal, Drum, and Bass cross-stem bleed matrix + Cymbal comb killer.\n"
            "- 🎙️ Regime B: Piano & Vocal Duet: High-Purity 2-stem vocal extraction. Prevents 6-stem models from cannibalizing piano bass octaves and hammer transients.\n"
            "- 🎛️ Legacy: HTDemucs 6S: Fast fallback demixer.\n"
            "- ⚡ Solo Piano: Direct acoustic transcription with 100% transient fidelity."
        )
        h_layout.addWidget(self.stem_combo)

        self.engine_combo = QComboBox()
        self.engine_combo.addItems([
            "🎯 Studio Master (Transkun SOTA)",
            "Spotify Basic Pitch"
        ])
        self.engine_combo.setMaximumWidth(220)
        self.engine_combo.setToolTip("AI Engine:\n- 🎯 Studio Master: Transkun 2.0 Non-Hierarchical Transformer + Semi-CRF at 44.1kHz (SOTA, Recommended).\n- Spotify Basic Pitch: Lightweight onset/frame neural network.")
        h_layout.addWidget(self.engine_combo)

        self.accuracy_combo = QComboBox()
        self.accuracy_combo.addItems([
            "Clean / Strict",
            "Balanced",
            "High Sensitivity"
        ])
        self.accuracy_combo.setMaximumWidth(135)
        self.accuracy_combo.setToolTip("Transcription Profile: Clean filters out vocal reverb and bleed artifacts.")
        h_layout.addWidget(self.accuracy_combo)

        self.btn_open = QPushButton("Open File")
        self.btn_open.setToolTip("Open audio file (*.mp3, *.wav, *.flac)")
        self.btn_open.clicked.connect(self.browse_file)
        h_layout.addWidget(self.btn_open)

        self.btn_demo = QPushButton("Demo")
        self.btn_demo.setToolTip("Play Beethoven Moonlight Sonata visualizer demo")
        self.btn_demo.clicked.connect(self.load_demo)
        h_layout.addWidget(self.btn_demo)

        self.btn_fullscreen = QPushButton("⛶ Fullscreen")
        self.btn_fullscreen.setToolTip("Toggle Fullscreen Mode (Press F11 or Esc)")
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        h_layout.addWidget(self.btn_fullscreen)

        layout.addWidget(header)

        # -------------------------------------------------------------
        # 2. Main Visualizer Area
        # -------------------------------------------------------------
        self.visualizer = RousseauVisualizerWidget(central)
        self.visualizer.note_clicked.connect(self.open_note_inspector)
        layout.addWidget(self.visualizer, 1)

        # Progress overlay bar (hidden by default)
        self.progress_frame = QFrame(central)
        self.progress_frame.setObjectName("ProgressFrame")
        p_layout = QVBoxLayout(self.progress_frame)
        self.progress_label = QLabel("Ready")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        p_layout.addWidget(self.progress_label)
        p_layout.addWidget(self.progress_bar)
        self.progress_frame.hide()
        layout.addWidget(self.progress_frame)

        # -------------------------------------------------------------
        # 3. Pianist Note Noise Gate & Note Length Filter Toolbar
        # -------------------------------------------------------------
        filter_bar = QFrame(central)
        filter_bar.setFixedHeight(44)
        filter_bar.setObjectName("FilterBarFrame")
        filter_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        fb_layout = QHBoxLayout(filter_bar)
        fb_layout.setContentsMargins(16, 0, 16, 0)
        fb_layout.setSpacing(12)

        gate_icon = QLabel("⚡ Noise Gate:")
        gate_icon.setStyleSheet("font-weight: bold; color: #00c6ff;")
        fb_layout.addWidget(gate_icon)

        self.vel_gate_slider = QSlider(Qt.Orientation.Horizontal)
        self.vel_gate_slider.setRange(0, 95)
        self.vel_gate_slider.setValue(40)
        self.vel_gate_slider.setFixedWidth(120)
        self.vel_gate_slider.setToolTip("Drag to instantly eliminate quiet background bleed or ghost notes")
        self.vel_gate_slider.valueChanged.connect(self.on_velocity_gate_changed)
        fb_layout.addWidget(self.vel_gate_slider)

        self.vel_gate_label = QLabel("Min Vel: 40")
        self.vel_gate_label.setFixedWidth(75)
        fb_layout.addWidget(self.vel_gate_label)

        dur_icon = QLabel("⏱️ Note Length:")
        dur_icon.setStyleSheet("font-weight: bold; color: #00c6ff;")
        fb_layout.addWidget(dur_icon)

        self.dur_slider = QSlider(Qt.Orientation.Horizontal)
        self.dur_slider.setRange(4, 35)  # 0.4s to 3.5s (35 = Full)
        self.dur_slider.setValue(12)      # 1.2s default (middle of the bell curve)
        self.dur_slider.setFixedWidth(110)
        self.dur_slider.setToolTip("Adjust chord and note visual & export length clamp (0.4s to 3.5s, or Full)")
        self.dur_slider.valueChanged.connect(self.on_duration_slider_changed)
        fb_layout.addWidget(self.dur_slider)

        self.dur_label = QLabel("1.2s")
        self.dur_label.setFixedWidth(45)
        fb_layout.addWidget(self.dur_label)

        self.notes_stat_label = QLabel("Active: 0 / 0 Notes")
        self.notes_stat_label.setFixedWidth(135)
        fb_layout.addWidget(self.notes_stat_label)

        self.btn_apply_filter = QPushButton("Apply to Audio")
        self.btn_apply_filter.setToolTip("Re-synthesize the 17MB Concert Grand sound using only the currently active filtered notes (preserving natural audio duration)")
        self.btn_apply_filter.clicked.connect(self.apply_filter_to_audio)
        fb_layout.addWidget(self.btn_apply_filter)

        self.btn_fingering = QPushButton("🖐️ Fingering: ON")
        self.btn_fingering.setToolTip("Toggle ergonomic 1-2-3-4-5 pianist finger numbers on falling notes")
        self.btn_fingering.clicked.connect(self.toggle_fingering)
        fb_layout.addWidget(self.btn_fingering)

        self.btn_grid = QPushButton("📏 Bar Grid: ON")
        self.btn_grid.setToolTip("Toggle horizontal measure bar lines and beat subdivisions")
        self.btn_grid.clicked.connect(self.toggle_grid)
        fb_layout.addWidget(self.btn_grid)

        self.btn_export_xml = QPushButton("🎼 Sheet Music (MusicXML)")
        self.btn_export_xml.setToolTip("Export transcription as standard 2-staff Grand Staff MusicXML for MuseScore, Sibelius, or Finale")
        self.btn_export_xml.clicked.connect(self.export_musicxml)
        fb_layout.addWidget(self.btn_export_xml)

        self.layer_combo = QComboBox()
        self.layer_combo.addItems([
            "🎧 Layer: Verified Master",
            "🤖 Layer: Raw AI (Unfiltered)",
            "🔍 Layer: Diagnostic Mode"
        ])
        self.layer_combo.setMaximumWidth(210)
        self.layer_combo.setToolTip(
            "Visualization & Audit Layer:\n"
            "- 🎧 Layer: Verified Master: Acoustically vetted notes after harmonic deconvolution\n"
            "- 🤖 Layer: Raw AI (Unfiltered): Direct neural model detections before filters\n"
            "- 🔍 Layer: Diagnostic Mode: Flags ghost candidates with dashed borders and hover reasons"
        )
        self.layer_combo.currentTextChanged.connect(self.on_layer_changed)
        fb_layout.addWidget(self.layer_combo)

        fb_layout.addStretch()
        layout.addWidget(filter_bar)

        # -------------------------------------------------------------
        # 4. Bottom Controls Toolbar
        # -------------------------------------------------------------
        footer = QFrame(central)
        footer.setFixedHeight(70)
        footer.setObjectName("FooterFrame")
        footer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        f_layout = QHBoxLayout(footer)
        f_layout.setContentsMargins(16, 0, 16, 0)
        f_layout.setSpacing(12)

        self.btn_play = QPushButton("Play")
        self.btn_play.setObjectName("PlayBtn")
        self.btn_play.setFixedSize(46, 46)
        self.btn_play.clicked.connect(self.toggle_play)
        f_layout.addWidget(self.btn_play)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setFixedWidth(95)
        f_layout.addWidget(self.time_label)

        self.metro_badge = QLabel("● 120 BPM")
        self.metro_badge.setToolTip("Auto-detected song tempo & visual metronome")
        self.metro_badge.setStyleSheet("color: #00c6ff; background: rgba(0, 198, 255, 0.12); padding: 4px 8px; border-radius: 4px; font-weight: bold; border: 1px solid rgba(0, 198, 255, 0.25);")
        f_layout.addWidget(self.metro_badge)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(self.on_slider_moved)
        f_layout.addWidget(self.slider, 1)

        # Audio source switcher
        self.source_combo = QComboBox()
        self.source_combo.addItems([
            "🎹 Concert Grand (SF2)",
            "🤖 Raw AI Grand (SF2)",
            "🎧 Isolated Piano Stem (Enhanced)",
            "🎤 Original Song Mix"
        ])
        self.source_combo.setMaximumWidth(200)
        self.source_combo.setToolTip("Toggle audio source between pure Concert Grand piano, isolated stem, or original song with vocals")
        self.source_combo.currentTextChanged.connect(self.on_source_changed)
        f_layout.addWidget(self.source_combo)

        self.btn_loop_a = QPushButton("Set [A]")
        self.btn_loop_a.clicked.connect(self.set_loop_a)
        f_layout.addWidget(self.btn_loop_a)

        self.btn_loop_b = QPushButton("Set [B]")
        self.btn_loop_b.clicked.connect(self.set_loop_b)
        f_layout.addWidget(self.btn_loop_b)

        self.btn_loop_clear = QPushButton("Clear")
        self.btn_loop_clear.clicked.connect(self.clear_loop)
        f_layout.addWidget(self.btn_loop_clear)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "0.75x", "1.0x", "1.25x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.setMaximumWidth(75)
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)
        f_layout.addWidget(self.speed_combo)

        self.btn_export_midi = QPushButton("Export MIDI")
        self.btn_export_midi.clicked.connect(self.export_midi)
        f_layout.addWidget(self.btn_export_midi)

        self.btn_export_wav = QPushButton("Export WAV")
        self.btn_export_wav.clicked.connect(self.export_wav)
        f_layout.addWidget(self.btn_export_wav)

        layout.addWidget(footer)

    def apply_dark_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #07080a;
            }
            #HeaderFrame {
                background-color: rgba(14, 17, 23, 0.96);
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            }
            #FilterBarFrame {
                background-color: rgba(12, 14, 20, 0.98);
                border-top: 1px solid rgba(0, 198, 255, 0.2);
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            }
            #FooterFrame {
                background-color: rgba(14, 17, 23, 0.96);
                border-top: 1px solid rgba(255, 255, 255, 0.08);
            }
            #TitleLabel {
                font-size: 16px;
                font-weight: 800;
                color: #f0f2f5;
                font-family: 'Outfit', 'Inter', sans-serif;
            }
            #BadgeLabel {
                background-color: rgba(0, 198, 255, 0.15);
                color: #00c6ff;
                border: 1px solid rgba(0, 198, 255, 0.3);
                border-radius: 4px;
                padding: 2px 7px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton {
                background-color: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.12);
                color: #f0f2f5;
                border-radius: 6px;
                padding: 5px 12px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.12);
                border-color: #00c6ff;
            }
            #PlayBtn {
                background-color: #00c6ff;
                color: #07080a;
                font-size: 14px;
                font-weight: bold;
                border-radius: 23px;
            }
            #PlayBtn:hover {
                background-color: #33d3ff;
            }
            QComboBox {
                color: #f0f2f5;
                font-size: 12px;
                background-color: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 6px;
                padding: 4px 8px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: rgba(255, 255, 255, 0.15);
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #00c6ff;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
            #ProgressFrame {
                background-color: rgba(7, 8, 10, 0.95);
                border-top: 1px solid rgba(0, 198, 255, 0.3);
                padding: 8px 24px;
            }
            QLabel {
                color: #f0f2f5;
            }
        """)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal()
            self.btn_fullscreen.setText("⛶ Fullscreen")
        else:
            super().keyPressEvent(event)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.btn_fullscreen.setText("⛶ Fullscreen")
        else:
            self.showFullScreen()
            self.btn_fullscreen.setText("🗗 Windowed")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                self.process_audio(path)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Music File", "", "Audio Files (*.mp3 *.wav *.flac *.m4a)"
        )
        if file_path:
            self.process_audio(file_path)

    def process_audio(self, file_path: str):
        if hasattr(self, 'worker') and self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(
                self,
                "Processing in Progress",
                "A song is currently being processed. Please wait for it to finish."
            )
            return

        self.current_audio_path = file_path
        self.progress_frame.show()
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting AI pipeline...")

        stem_text = self.stem_combo.currentText()
        if "Auto-Detect" in stem_text:
            stem_mode = "auto"
        elif "Full Band" in stem_text or "BS-RoFormer" in stem_text:
            stem_mode = "bs_roformer"
        elif "Duet" in stem_text or "MDX" in stem_text or "Vocals" in stem_text:
            stem_mode = "mdx_inst"
        elif "HTDemucs" in stem_text:
            stem_mode = "demucs_6s"
        else:
            stem_mode = "none"

        acc_text = self.accuracy_combo.currentText()
        if "Clean" in acc_text:
            precision_mode = "clean"
        elif "Sensitivity" in acc_text:
            precision_mode = "sensitive"
        else:
            precision_mode = "balanced"

        engine_text = self.engine_combo.currentText()
        if "Studio Master" in engine_text or "Dual-AI" in engine_text:
            model_engine = "studio_master"
        elif "Spotify" in engine_text:
            model_engine = "basic_pitch"
        else:
            model_engine = "bytedance"

        self.worker = ProcessingWorker(
            file_path,
            stem_mode=stem_mode,
            precision_mode=precision_mode,
            model_engine=model_engine
        )
        self.worker.progress_signal.connect(self.on_worker_progress)
        self.worker.finished_signal.connect(self.on_worker_finished)
        self.worker.failed_signal.connect(self.on_worker_failed)
        self.worker.start()

    def on_worker_progress(self, progress: int, msg: str):
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"{msg} ({progress}%)")

    def on_worker_finished(self, result: Dict[str, Any]):
        self.progress_frame.hide()
        self.current_data = result

        # Load MIDI data into Visualizer
        gate_val = self.vel_gate_slider.value()
        self.visualizer.load_data(
            result["visualizer_data"],
            min_velocity=gate_val,
            max_duration=self.max_duration_clamp
        )

        active_cnt = len(self.visualizer.notes)
        total_cnt = len(self.visualizer.all_notes)
        self.notes_stat_label.setText(f"Active: {active_cnt} / {total_cnt} Notes")

        # Update Key Signature Badge
        key_name = result.get("visualizer_data", {}).get("key_signature", "Auto")
        self.key_badge.setText(f"🎼 Key: {key_name}")

        # Update Tempo Metronome Badge
        tempo = result.get("visualizer_data", {}).get("tempo_bpm", 120.0)
        self.metro_badge.setText(f"● {tempo:.1f} BPM")

        # Cache available audio sources
        self.audio_sources = {
            "🎹 Concert Grand (SF2 Master)": result.get("pure_piano_file"),
            "🤖 Raw AI Grand (SF2)": result.get("raw_piano_file"),
            "🎧 Isolated Piano Stem (Enhanced)": result.get("stem_audio_file"),
            "🎙️ Isolated Vocal Stem": result.get("vocal_stem_file"),
            "🥁 Isolated Drum Stem": result.get("drums_stem_file"),
            "🎸 Isolated Bass Stem": result.get("bass_stem_file"),
            "📻 Original Song Mix": result.get("original_audio_file")
        }

        # Filter down to only sources that exist on disk
        available_sources = [k for k, v in self.audio_sources.items() if v and os.path.exists(v)]

        self.source_combo.blockSignals(True)
        cur_text = self.source_combo.currentText()
        self.source_combo.clear()
        self.source_combo.addItems(available_sources)
        if cur_text in available_sources:
            self.source_combo.setCurrentText(cur_text)
        self.source_combo.blockSignals(False)

        # Load selected audio into native QMediaPlayer
        chosen_source = self.source_combo.currentText()
        target_path = self.audio_sources.get(chosen_source) or result.get("pure_piano_file") or result.get("stem_audio_file")
        if target_path and os.path.exists(target_path):
            self.player.setSource(QUrl.fromLocalFile(str(target_path)))

        self.clock.start(0.0, rate=self.player.playbackRate())
        self.timer.start()
        self.btn_play.setText("Pause")
        self.player.play()

    def on_velocity_gate_changed(self, val: int):
        active_cnt = self.visualizer.set_velocity_gate(val)
        total_cnt = len(self.visualizer.all_notes)
        self.vel_gate_label.setText(f"Min Vel: {val}")
        self.notes_stat_label.setText(f"Active: {active_cnt} / {total_cnt} Notes")

    def on_duration_slider_changed(self, val: int):
        if val >= 35:
            self.max_duration_clamp = None
            self.dur_label.setText("Full")
        else:
            self.max_duration_clamp = val / 10.0
            self.dur_label.setText(f"{self.max_duration_clamp:.1f}s")
        active_cnt = self.visualizer.set_duration_clamp(self.max_duration_clamp or 0.0)
        total_cnt = len(self.visualizer.all_notes)
        self.notes_stat_label.setText(f"Active: {active_cnt} / {total_cnt} Notes")

    def apply_filter_to_audio(self):
        if not self.current_data:
            QMessageBox.information(self, "Filter", "No song processed yet. Load a song first!")
            return

        gate_val = self.vel_gate_slider.value()
        max_dur = self.max_duration_clamp
        orig_midi = Path(self.current_data["midi_file"])
        filtered_midi = orig_midi.parent / f"{orig_midi.stem}_filtered.mid"
        filtered_pure = Path(self.current_data["pure_piano_file"])

        filter_and_export_midi(
            orig_midi,
            filtered_midi,
            min_velocity=gate_val,
            max_duration=max_dur
        )
        synthesize_midi_to_audio(filtered_midi, filtered_pure)

        was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        cur_pos = self.player.position()
        self.player.setSource(QUrl.fromLocalFile(str(filtered_pure)))
        self.player.setPosition(cur_pos)
        self.clock.seek(cur_pos / 1000.0)
        if was_playing:
            self.player.play()

        dur_text = f"max {max_dur:.1f}s" if max_dur else "natural full length"
        QMessageBox.information(
            self,
            "Filter Applied",
            f"Concert Grand soundfont audio updated!\nActive notes: {len(self.visualizer.notes)}\nLength: {dur_text}."
        )

    def toggle_fingering(self):
        self.visualizer.show_fingering = not self.visualizer.show_fingering
        txt = "🖐️ Fingering: ON" if self.visualizer.show_fingering else "🖐️ Fingering: OFF"
        self.btn_fingering.setText(txt)
        self.visualizer.update()

    def export_musicxml(self):
        if not self.visualizer.notes:
            QMessageBox.information(self, "No Notes", "Please transcribe or load an audio file first.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Sheet Music (MusicXML)",
            str(MIDI_DIR / "sheet_music.musicxml"),
            "MusicXML Files (*.musicxml *.xml)"
        )
        if out_path:
            try:
                from backend.musicxml_exporter import export_to_musicxml
                key_name = self.current_data.get("visualizer_data", {}).get("key_signature", self.visualizer.key_name) if self.current_data else self.visualizer.key_name
                tempo = self.current_data.get("visualizer_data", {}).get("tempo_bpm", self.visualizer.tempo_bpm) if self.current_data else self.visualizer.tempo_bpm
                title = Path(self.current_audio_path).stem if self.current_audio_path else "Piano Stem Studio"
                export_to_musicxml(
                    self.visualizer.notes,
                    Path(out_path),
                    title=title,
                    key_name=key_name,
                    tempo_bpm=tempo
                )
                QMessageBox.information(self, "Export Successful", f"Grand staff sheet music exported to:\n{out_path}\n\nYou can now open it in MuseScore, Sibelius, Dorico, or Finale.")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Could not export MusicXML: {e}")

    def toggle_grid(self):
        self.visualizer.show_measure_lines = not self.visualizer.show_measure_lines
        txt = "📏 Bar Grid: ON" if self.visualizer.show_measure_lines else "📏 Bar Grid: OFF"
        self.btn_grid.setText(txt)
        self.visualizer.update()

    def on_layer_changed(self, text: str):
        if not self.current_data:
            return
        gate_val = self.vel_gate_slider.value()
        if "Raw" in text:
            raw_data = self.current_data.get("raw_visualizer_data")
            if raw_data:
                self.visualizer.diagnostic_mode = False
                self.visualizer.load_data(raw_data, min_velocity=gate_val, max_duration=self.max_duration_clamp)
        elif "Diagnostic" in text:
            master_data = self.current_data.get("visualizer_data")
            if master_data:
                self.visualizer.diagnostic_mode = True
                self.visualizer.load_data(master_data, min_velocity=gate_val, max_duration=self.max_duration_clamp)
        else:
            master_data = self.current_data.get("visualizer_data")
            if master_data:
                self.visualizer.diagnostic_mode = False
                self.visualizer.load_data(master_data, min_velocity=gate_val, max_duration=self.max_duration_clamp)

        active_cnt = len(self.visualizer.notes)
        total_cnt = len(self.visualizer.all_notes)
        self.notes_stat_label.setText(f"Active: {active_cnt} / {total_cnt} Notes")
        self.visualizer.update()

    def open_note_inspector(self, note: dict):
        dialog = NoteInspectorDialog(note, self)
        dialog.note_deleted.connect(self.on_note_deleted)
        dialog.note_hand_toggled.connect(self.on_note_hand_toggled)
        dialog.exec()

    def on_note_deleted(self, note: dict):
        note_id = note.get("id")
        self.visualizer.all_notes = [n for n in self.visualizer.all_notes if n.get("id") != note_id]
        if self.current_data and "visualizer_data" in self.current_data:
            v_notes = self.current_data["visualizer_data"].get("notes", [])
            self.current_data["visualizer_data"]["notes"] = [n for n in v_notes if n.get("id") != note_id]
        active_cnt = self.visualizer._refresh_notes()
        total_cnt = len(self.visualizer.all_notes)
        self.notes_stat_label.setText(f"Active: {active_cnt} / {total_cnt} Notes")
        self.visualizer.selected_note_id = None
        self.visualizer.update()

    def on_note_hand_toggled(self, note: dict):
        note_id = note.get("id")
        for n in self.visualizer.all_notes:
            if n.get("id") == note_id:
                curr = n.get("hand", "right")
                n["hand"] = "left" if curr == "right" else "right"
                break
        self.visualizer._refresh_notes()
        self.visualizer.update()

    def on_source_changed(self, text: str):
        if not hasattr(self, 'audio_sources') or text not in self.audio_sources:
            return
        target = self.audio_sources[text]
        if not target or not os.path.exists(target):
            return
        was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        cur_pos = self.player.position()
        self.player.setSource(QUrl.fromLocalFile(str(target)))
        self.player.setPosition(cur_pos)
        self.clock.seek(cur_pos / 1000.0)
        if was_playing:
            self.player.play()

    def on_worker_failed(self, error_msg: str):
        self.progress_frame.hide()
        QMessageBox.critical(self, "Pipeline Error", f"Processing failed:\n{error_msg}")

    def on_timer_tick(self):
        dur_ms = self.player.duration()
        pos_ms = self.player.position()

        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.clock.sync_hardware_clock(pos_ms / 1000.0)
            cur_time = self.clock.get_time()

            # Loop check
            if self.loop_a is not None and self.loop_b is not None:
                if cur_time >= self.loop_b:
                    self.player.setPosition(int(self.loop_a * 1000))
                    self.clock.seek(self.loop_a)
                    return

            self.visualizer.set_time(cur_time)

            # Visual Metronome Beat Pulse
            bi = self.visualizer.beat_interval
            if bi > 0.05:
                beat_num = int(cur_time / bi)
                is_downbeat = (beat_num % self.visualizer.beats_per_measure) == 0
                if is_downbeat:
                    self.metro_badge.setStyleSheet("color: #ffffff; background: rgba(0, 198, 255, 0.40); padding: 4px 8px; border-radius: 4px; font-weight: bold; border: 1px solid #00c6ff;")
                else:
                    self.metro_badge.setStyleSheet("color: #00c6ff; background: rgba(0, 198, 255, 0.12); padding: 4px 8px; border-radius: 4px; font-weight: bold; border: 1px solid rgba(0, 198, 255, 0.25);")

            if dur_ms > 0:
                self.slider.blockSignals(True)
                self.slider.setValue(int((cur_time / (dur_ms / 1000.0)) * 1000))
                self.slider.blockSignals(False)
                self.time_label.setText(f"{self.fmt_time(cur_time)} / {self.fmt_time(dur_ms / 1000.0)}")
        else:
            if dur_ms > 0:
                cur_time = pos_ms / 1000.0
                self.time_label.setText(f"{self.fmt_time(cur_time)} / {self.fmt_time(dur_ms / 1000.0)}")

    def fmt_time(self, sec: float) -> str:
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}:{s:02d}"

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.btn_play.setText("Play")
            self.clock.pause(self.player.position() / 1000.0)
        else:
            self.player.play()
            self.btn_play.setText("Pause")
            self.clock.start(self.player.position() / 1000.0, rate=self.player.playbackRate())
            self.timer.start()

    def on_slider_moved(self, val: int):
        if self.player.duration() > 0:
            target_sec = (val / 1000.0) * (self.player.duration() / 1000.0)
            self.player.setPosition(int(target_sec * 1000))
            self.clock.seek(target_sec)
            self.visualizer.set_time(target_sec)

    def on_speed_changed(self, text: str):
        speed = float(text.replace('x', ''))
        self.player.setPlaybackRate(speed)
        self.clock.set_rate(speed)

    def on_theme_changed(self, text: str):
        self.visualizer.set_theme(text)

    def set_loop_a(self):
        pos = self.clock.get_time()
        bi = self.visualizer.beat_interval
        if bi > 0.1:
            nearest = round(pos / bi) * bi
            if abs(pos - nearest) <= 0.18:
                pos = nearest
        self.loop_a = pos
        self.btn_loop_a.setText(f"A: {self.fmt_time(pos)}")

    def set_loop_b(self):
        pos = self.clock.get_time()
        bi = self.visualizer.beat_interval
        if bi > 0.1:
            nearest = round(pos / bi) * bi
            if abs(pos - nearest) <= 0.18:
                pos = nearest
        self.loop_b = pos
        self.btn_loop_b.setText(f"B: {self.fmt_time(pos)}")

    def clear_loop(self):
        self.loop_a = None
        self.loop_b = None
        self.btn_loop_a.setText("Set [A]")
        self.btn_loop_b.setText("Set [B]")

    def export_midi(self):
        if not self.current_data:
            QMessageBox.information(self, "Export", "No MIDI available yet. Process a song first!")
            return
        orig_midi = Path(self.current_data["midi_file"])
        gate_val = self.vel_gate_slider.value()

        dest, _ = QFileDialog.getSaveFileName(self, "Save Clean MIDI File", orig_midi.name, "MIDI Files (*.mid)")
        if dest:
            filter_and_export_midi(
                orig_midi,
                Path(dest),
                min_velocity=gate_val,
                max_duration=self.max_duration_clamp
            )
            dur_str = f"max {self.max_duration_clamp:.1f}s" if self.max_duration_clamp else "natural full length"
            QMessageBox.information(
                self,
                "Export",
                f"Clean MIDI exported to:\n{dest}\n\nSettings: Min Vel {gate_val}, Length: {dur_str}."
            )

    def export_wav(self):
        if not self.current_data or not self.current_data.get("pure_piano_file"):
            QMessageBox.information(self, "Export", "No piano audio available yet. Process a song first!")
            return
        pure_path = Path(self.current_data["pure_piano_file"])
        if not pure_path.exists():
            QMessageBox.warning(self, "File Not Found", "The piano audio file was not found on disk.")
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Save Concert Grand Piano Audio", pure_path.name, "WAV Audio (*.wav)")
        if dest:
            import shutil
            shutil.copy(pure_path, dest)
            QMessageBox.information(self, "Export", f"Concert Grand piano audio exported to:\n{dest}")

    def load_demo(self):
        notes = []
        t = 0.0
        arpeggio = [49, 56, 61]
        for i in range(28):
            pitch = arpeggio[i % 3] + (3 if i > 12 else 0)
            notes.append({'midi': pitch, 'time': t, 'duration': 0.35, 'velocity': 0.8, 'hand': 'right'})
            if i in [6, 15, 24]:
                notes.append({'midi': 68, 'time': t, 'duration': 1.2, 'velocity': 0.9, 'hand': 'right'})
            t += 0.35

        demo_data = {
            'duration': 12.0,
            'notes': notes,
            'pedals': [{'time': 0.0, 'duration': 12.0}]
        }
        self.visualizer.load_data(demo_data)
        self.notes_stat_label.setText(f"Active: {len(notes)} / {len(notes)} Notes")
        self.clock.start(0.0)
        self.timer.start()

    def closeEvent(self, event):
        self.timer.stop()
        self.player.stop()
        if hasattr(self, 'worker') and self.worker is not None and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(1000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = PianoStudioWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
