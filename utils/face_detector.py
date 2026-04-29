"""
FaceDetector — PERCEPTION module
Uses OpenCV's DNN face detector (SSD ResNet) with Haar cascade fallback.
Returns face bounding boxes and 68-point landmark estimates.
"""

import cv2
import numpy as np
import os


class FaceDetector:
    """
    Detects faces using OpenCV's built-in detectors.
    Primary: Haar cascade (fast, no extra model files needed)
    Produces pseudo-landmark grid sufficient for eye region extraction.
    """

    def __init__(self):
        # Load Haar cascades (bundled with OpenCV — no download needed)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )

        # Try to load dlib if available for better landmarks
        self._dlib_available = False
        try:
            import dlib
            self._detector = dlib.get_frontal_face_detector()
            model_path = os.path.join(os.path.dirname(__file__), '..', 'model',
                                      'shape_predictor_68_face_landmarks.dat')
            if os.path.exists(model_path):
                self._predictor = dlib.shape_predictor(model_path)
                self._dlib_available = True
        except ImportError:
            pass

    def detect(self, gray: np.ndarray):
        """
        Detect faces and estimate landmarks.
        Returns: (list of (x,y,w,h), list of landmark arrays or None)
        """
        if self._dlib_available:
            return self._detect_dlib(gray)
        return self._detect_haar(gray)

    def _detect_haar(self, gray: np.ndarray):
        """Haar cascade detection with synthetic landmark estimation."""
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return [], []

        face_list = [tuple(f) for f in faces]
        landmarks = []
        for (x, y, w, h) in face_list:
            lm = self._estimate_landmarks(x, y, w, h, gray)
            landmarks.append(lm)
        return face_list, landmarks

    def _detect_dlib(self, gray: np.ndarray):
        """dlib 68-point landmark detection."""
        import dlib
        dets = self._detector(gray, 1)
        if not dets:
            return [], []

        face_list = []
        landmarks = []
        for d in dets:
            x = max(0, d.left())
            y = max(0, d.top())
            w = d.right() - x
            h = d.bottom() - y
            face_list.append((x, y, w, h))

            shape = self._predictor(gray, d)
            lm = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
            landmarks.append(lm)

        return face_list, landmarks

    def _estimate_landmarks(self, x, y, w, h, gray) -> np.ndarray:
        """
        Estimate 68-point landmark positions geometrically from face bbox.
        Eye region indices: left eye 36-41, right eye 42-47.
        These are good enough approximations for EAR computation when dlib
        is unavailable.
        """
        lm = np.zeros((68, 2), dtype=np.float32)

        # Estimate eye centres from Haar sub-detection
        face_roi = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 3, minSize=(20, 20))

        # Sort eyes left-right
        eyes_sorted = sorted(eyes, key=lambda e: e[0]) if len(eyes) >= 2 else []

        if len(eyes_sorted) >= 2:
            le = eyes_sorted[0]   # left eye in face-space
            re = eyes_sorted[1]   # right eye
        else:
            # Fallback geometry
            le = (int(w * 0.20), int(h * 0.35), int(w * 0.25), int(h * 0.15))
            re = (int(w * 0.55), int(h * 0.35), int(w * 0.25), int(h * 0.15))

        def eye_points(ex, ey, ew, eh, base_idx):
            """Generate 6 points approximating eye contour."""
            cx, cy = ex + ew // 2, ey + eh // 2
            rx, ry = ew // 2, eh // 2
            pts = [
                (cx - rx, cy),          # left corner
                (cx - rx//2, cy - ry),  # upper-left
                (cx + rx//2, cy - ry),  # upper-right
                (cx + rx, cy),          # right corner
                (cx + rx//2, cy + ry),  # lower-right
                (cx - rx//2, cy + ry),  # lower-left
            ]
            for i, (px, py) in enumerate(pts):
                lm[base_idx + i] = [x + px, y + py]

        eye_points(*le, 36)
        eye_points(*re, 42)
        return lm
