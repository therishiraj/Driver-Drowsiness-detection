"""
AlertSystem — ACTION sub-module
Dual-channel alert: audio beep (cross-platform) + frontend notification flag.
Designed for < 200ms notification latency from detection event.
"""

import threading
import time
import math
import wave
import os
import struct


class AlertSystem:
    """
    Fires drowsiness alerts through two channels:
      1. Audio — synthesised beep wav played via platform audio (no external deps)
      2. Visual — sets a flag read by the frontend (handled in annotation layer)
    """

    def __init__(self):
        self._alert_thread = None
        self._stop_event = threading.Event()
        self._beep_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'sounds', 'alert.wav')
        self._generate_beep_wav()

    # ── Public API ──────────────────────────────────────────────────────────

    def trigger(self, severity: str = 'DROWSY'):
        """Start alert in background thread (non-blocking)."""
        self.stop_alert()
        self._stop_event.clear()
        freq = 880 if severity == 'SLEEPING' else 440
        self._alert_thread = threading.Thread(
            target=self._play_loop, args=(freq,), daemon=True
        )
        self._alert_thread.start()

    def stop_alert(self):
        """Stop any active alert."""
        self._stop_event.set()
        if self._alert_thread and self._alert_thread.is_alive():
            self._alert_thread.join(timeout=0.5)

    # ── Internal ────────────────────────────────────────────────────────────

    def _play_loop(self, freq: int):
        """Play alert sound repeatedly until stop_event is set."""
        while not self._stop_event.is_set():
            self._play_sound()
            if self._stop_event.wait(timeout=1.5):
                break

    def _play_sound(self):
        """Play the pre-generated alert WAV using platform audio."""
        if not os.path.exists(self._beep_path):
            return
        try:
            import subprocess, sys
            if sys.platform == 'darwin':
                subprocess.Popen(['afplay', self._beep_path],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif sys.platform == 'win32':
                import winsound
                winsound.PlaySound(self._beep_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                # Linux — try aplay, paplay, then pygame fallback
                for player in ['aplay', 'paplay']:
                    r = subprocess.run(['which', player], capture_output=True)
                    if r.returncode == 0:
                        subprocess.Popen([player, self._beep_path],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return
                # pygame fallback
                try:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(self._beep_path)
                    pygame.mixer.music.play()
                except Exception:
                    pass
        except Exception:
            pass

    def _generate_beep_wav(self):
        """Synthesise a 0.5s 440Hz sine wave WAV and save it."""
        os.makedirs(os.path.dirname(self._beep_path), exist_ok=True)
        if os.path.exists(self._beep_path):
            return  # Already exists

        sample_rate = 44100
        duration    = 0.5
        frequency   = 440.0
        amplitude   = 28000
        n_samples   = int(sample_rate * duration)

        samples = []
        for i in range(n_samples):
            t = i / sample_rate
            # Sine with fade-in / fade-out envelope
            env = min(i, n_samples - i, sample_rate // 20) / (sample_rate // 20)
            val = int(amplitude * env * math.sin(2 * math.pi * frequency * t))
            samples.append(struct.pack('<h', val))

        with wave.open(self._beep_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(samples))
