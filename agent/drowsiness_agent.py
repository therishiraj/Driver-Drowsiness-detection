"""
DrowsinessAgent — Intelligent Agent Core
=========================================
Architecture:
  PERCEPTION  → Captures & preprocesses webcam frames, detects facial landmarks
  REASON      → Computes Eye Aspect Ratio (EAR), classifies drowsiness state via CNN
  ACTION      → Triggers audio-visual alerts, logs events
  FEEDBACK    → Maintains rolling state history, adjusts thresholds adaptively

This follows the classic sense-think-act loop used in intelligent agent design.
"""

import cv2
import numpy as np
import time
import threading
from collections import deque
from utils.face_detector import FaceDetector
from utils.eye_processor import EyeProcessor
from utils.cnn_classifier import CNNClassifier
from utils.alert_system import AlertSystem


class AgentState:
    """Encapsulates the full agent state at any point in time."""
    AWAKE = "AWAKE"
    DROWSY = "DROWSY"
    SLEEPING = "SLEEPING"
    NO_FACE = "NO_FACE"


class DrowsinessAgent:
    """
    Intelligent Agent for real-time driver drowsiness detection.

    Perception  : FaceDetector + EyeProcessor extract eye regions from frames
    Reason      : CNNClassifier infers eye state; EAR threshold validates
    Action      : AlertSystem fires audio + on-screen warnings
    Feedback    : Rolling EAR history + consecutive frame counters adapt sensitivity
    """

    # ── Tuneable Thresholds ─────────────────────────────────────────────────
    EAR_THRESHOLD = 0.25          # Below this → eye considered closed
    CONSECUTIVE_FRAMES = 20       # Frames with closed eyes → DROWSY alert
    SLEEPING_FRAMES = 48          # Extended closure → SLEEPING alert
    EAR_HISTORY_SIZE = 90         # Rolling window for EAR (≈3 s at 30 fps)
    BLINK_COOLDOWN_S = 0.15       # Ignore closures shorter than this (blinks)

    def __init__(self):
        # ── Sub-systems ─────────────────────────────────────────────────────
        self.face_detector = FaceDetector()
        self.eye_processor = EyeProcessor()
        self.cnn_classifier = CNNClassifier()
        self.alert_system = AlertSystem()

        # ── Internal state ──────────────────────────────────────────────────
        self._state = AgentState.NO_FACE
        self._ear_history = deque(maxlen=self.EAR_HISTORY_SIZE)
        self._consecutive_closed = 0
        self._total_blinks = 0
        self._blink_start = None
        self._alert_active = False
        self._session_start = time.time()
        self._last_alert_time = 0
        self._frame_count = 0
        self._fps_counter = deque(maxlen=30)
        self._lock = threading.Lock()

        # ── Metrics ─────────────────────────────────────────────────────────
        self._perception_log = deque(maxlen=300)   # last 300 frames of events

    # ════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ════════════════════════════════════════════════════════════════════════

    def process_frame(self, frame: np.ndarray):
        """
        Full agent cycle: PERCEIVE → REASON → ACT → log FEEDBACK.
        Returns (annotated_frame, state_dict).
        """
        t_start = time.perf_counter()
        self._frame_count += 1

        # ── PERCEPTION ──────────────────────────────────────────────────────
        perception = self._perceive(frame)

        # ── REASONING ───────────────────────────────────────────────────────
        reasoning = self._reason(perception)

        # ── ACTION ──────────────────────────────────────────────────────────
        self._act(reasoning)

        # ── FEEDBACK / ANNOTATION ───────────────────────────────────────────
        annotated = self._annotate(frame.copy(), perception, reasoning)

        # FPS tracking
        elapsed = time.perf_counter() - t_start
        self._fps_counter.append(elapsed)

        self._perception_log.append({
            'frame': self._frame_count,
            'state': self._state,
            'ear': perception.get('ear'),
            'timestamp': time.time()
        })

        return annotated, self.get_state()

    def get_state(self) -> dict:
        """Return serialisable agent state for the frontend."""
        fps = 1.0 / (sum(self._fps_counter) / len(self._fps_counter)) if self._fps_counter else 0
        avg_ear = (sum(e for e in self._ear_history if e is not None) /
                   max(1, sum(1 for e in self._ear_history if e is not None)))
        return {
            'state': self._state,
            'ear': round(avg_ear, 3),
            'consecutive_closed': self._consecutive_closed,
            'total_blinks': self._total_blinks,
            'alert_active': self._alert_active,
            'session_duration': round(time.time() - self._session_start, 1),
            'frame_count': self._frame_count,
            'fps': round(fps, 1),
            'alert_threshold_frames': self.CONSECUTIVE_FRAMES,
        }

    def reset(self):
        """Reset agent internal counters (keeps model loaded)."""
        self._state = AgentState.NO_FACE
        self._ear_history.clear()
        self._consecutive_closed = 0
        self._total_blinks = 0
        self._blink_start = None
        self._alert_active = False
        self._session_start = time.time()
        self._last_alert_time = 0
        self._frame_count = 0
        self._perception_log.clear()
        self.alert_system.stop_alert()

    # ════════════════════════════════════════════════════════════════════════
    # PERCEPTION
    # ════════════════════════════════════════════════════════════════════════

    def _perceive(self, frame: np.ndarray) -> dict:
        """
        Extract sensory data from raw frame.
        Returns dict with: face_detected, landmarks, left_eye, right_eye, ear
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Adaptive histogram equalisation for varied lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        faces, landmarks = self.face_detector.detect(gray_eq)

        if not faces:
            self._ear_history.append(None)
            return {'face_detected': False, 'faces': [], 'landmarks': None,
                    'left_eye': None, 'right_eye': None, 'ear': None,
                    'gray': gray_eq}

        # Use the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        lm = landmarks[faces.index(face)] if landmarks else None

        left_eye, right_eye, ear = self.eye_processor.extract_eyes(gray_eq, lm, face)
        self._ear_history.append(ear)

        return {
            'face_detected': True,
            'faces': faces,
            'face': face,
            'landmarks': lm,
            'left_eye': left_eye,
            'right_eye': right_eye,
            'ear': ear,
            'gray': gray_eq
        }

    # ════════════════════════════════════════════════════════════════════════
    # REASONING
    # ════════════════════════════════════════════════════════════════════════

    def _reason(self, perception: dict) -> dict:
        """
        Combine EAR threshold + CNN classification to determine drowsiness.
        Returns dict with: eye_state, cnn_confidence, ear_closed, new_state
        """
        if not perception['face_detected']:
            self._consecutive_closed = 0
            self._state = AgentState.NO_FACE
            return {'eye_state': 'unknown', 'cnn_confidence': 0.0,
                    'ear_closed': False, 'new_state': AgentState.NO_FACE}

        ear = perception['ear']
        ear_closed = (ear is not None) and (ear < self.EAR_THRESHOLD)

        # CNN inference on eye crops
        cnn_label, cnn_conf = 'open', 0.5
        if perception['left_eye'] is not None:
            cnn_label, cnn_conf = self.cnn_classifier.predict(
                perception['left_eye'], perception['right_eye']
            )

        # Fuse EAR + CNN: both must agree for high-confidence closed detection
        eyes_closed = ear_closed and (cnn_label == 'closed')

        # Update consecutive counter
        if eyes_closed:
            self._consecutive_closed += 1
            if self._blink_start is None:
                self._blink_start = time.time()
        else:
            # Count as blink only if closure was brief (< cooldown)
            if self._blink_start is not None:
                duration = time.time() - self._blink_start
                if duration < self.BLINK_COOLDOWN_S:
                    self._total_blinks += 1
                self._blink_start = None
            self._consecutive_closed = 0

        # Determine new state
        if self._consecutive_closed >= self.SLEEPING_FRAMES:
            new_state = AgentState.SLEEPING
        elif self._consecutive_closed >= self.CONSECUTIVE_FRAMES:
            new_state = AgentState.DROWSY
        else:
            new_state = AgentState.AWAKE

        self._state = new_state
        return {
            'eye_state': cnn_label,
            'cnn_confidence': round(cnn_conf, 3),
            'ear_closed': ear_closed,
            'eyes_closed': eyes_closed,
            'new_state': new_state,
            'consecutive': self._consecutive_closed,
        }

    # ════════════════════════════════════════════════════════════════════════
    # ACTION
    # ════════════════════════════════════════════════════════════════════════

    def _act(self, reasoning: dict):
        """
        Fire alerts based on reasoning output.
        Dual-channel: audio beep + on-screen warning flag.
        """
        state = reasoning['new_state']
        now = time.time()

        if state in (AgentState.DROWSY, AgentState.SLEEPING):
            if not self._alert_active or (now - self._last_alert_time) > 2.0:
                self.alert_system.trigger(severity=state)
                self._alert_active = True
                self._last_alert_time = now
        else:
            if self._alert_active:
                self.alert_system.stop_alert()
                self._alert_active = False

    # ════════════════════════════════════════════════════════════════════════
    # FEEDBACK / ANNOTATION
    # ════════════════════════════════════════════════════════════════════════

    def _annotate(self, frame: np.ndarray, perception: dict, reasoning: dict) -> np.ndarray:
        """Draw agent state overlay on frame for visual feedback."""
        state = reasoning['new_state']
        h, w = frame.shape[:2]

        # ── State colour ────────────────────────────────────────────────────
        colours = {
            AgentState.AWAKE:    (0, 200, 0),
            AgentState.DROWSY:   (0, 165, 255),
            AgentState.SLEEPING: (0, 0, 220),
            AgentState.NO_FACE:  (180, 180, 180),
        }
        colour = colours.get(state, (255, 255, 255))

        # ── Alert banner ────────────────────────────────────────────────────
        if state == AgentState.DROWSY:
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 165, 255), -1)
            cv2.putText(frame, '⚠ DROWSINESS DETECTED — WAKE UP!', (10, 42),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
        elif state == AgentState.SLEEPING:
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 200), -1)
            cv2.putText(frame, '🚨 SLEEPING AT WHEEL — PULL OVER!', (10, 42),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

        # ── Face bounding box ───────────────────────────────────────────────
        if perception['face_detected']:
            x, y, fw, fh = perception['face']
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), colour, 2)

        # ── HUD overlay (bottom panel) ──────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 90), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        ear = perception.get('ear')
        ear_str = f'{ear:.3f}' if ear is not None else '---'
        fps = 1.0 / (sum(self._fps_counter) / max(1, len(self._fps_counter))) if self._fps_counter else 0

        hud_items = [
            (f'STATE: {state}', colour),
            (f'EAR: {ear_str}', (200, 200, 200)),
            (f'CLOSED: {self._consecutive_closed}/{self.CONSECUTIVE_FRAMES}', (200, 200, 200)),
            (f'BLINKS: {self._total_blinks}  FPS: {fps:.0f}', (150, 150, 150)),
        ]
        for i, (text, col) in enumerate(hud_items):
            y_pos = h - 85 + i * 21
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

        # EAR progress bar
        if ear is not None:
            bar_w = int(min(ear / 0.4, 1.0) * 200)
            cv2.rectangle(frame, (w - 220, h - 30), (w - 20, h - 15), (60, 60, 60), -1)
            cv2.rectangle(frame, (w - 220, h - 30), (w - 220 + bar_w, h - 15), colour, -1)
            cv2.putText(frame, 'EAR', (w - 220, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        return frame
