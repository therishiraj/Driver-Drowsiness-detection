"""
EyeProcessor — PERCEPTION sub-module
Computes Eye Aspect Ratio (EAR) and extracts normalised eye crops.
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist


class EyeProcessor:
    """
    Processes facial landmarks to extract eye regions and compute EAR.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Where p1..p6 are the 6 landmark points around each eye (Soukupová & Čech, 2016).
    """

    # dlib landmark indices
    LEFT_EYE_IDX  = list(range(36, 42))
    RIGHT_EYE_IDX = list(range(42, 48))
    EYE_CROP_SIZE = (64, 32)  # width × height for CNN input

    def extract_eyes(self, gray: np.ndarray, landmarks, face_bbox):
        """
        Extract eye crops and compute fused EAR.
        Returns: (left_eye_crop, right_eye_crop, ear_value)
        """
        if landmarks is None:
            return None, None, None

        left_pts  = landmarks[self.LEFT_EYE_IDX]
        right_pts = landmarks[self.RIGHT_EYE_IDX]

        left_ear  = self._eye_aspect_ratio(left_pts)
        right_ear = self._eye_aspect_ratio(right_pts)
        ear = (left_ear + right_ear) / 2.0

        left_crop  = self._crop_eye(gray, left_pts)
        right_crop = self._crop_eye(gray, right_pts)

        return left_crop, right_crop, ear

    @staticmethod
    def _eye_aspect_ratio(eye_pts: np.ndarray) -> float:
        """Compute EAR from 6 landmark points."""
        # Vertical distances
        A = dist.euclidean(eye_pts[1], eye_pts[5])
        B = dist.euclidean(eye_pts[2], eye_pts[4])
        # Horizontal distance
        C = dist.euclidean(eye_pts[0], eye_pts[3])
        if C < 1e-6:
            return 0.0
        return (A + B) / (2.0 * C)

    def _crop_eye(self, gray: np.ndarray, eye_pts: np.ndarray):
        """
        Crop eye region with padding, resize to fixed size.
        Applies CLAHE for lighting normalisation.
        """
        x_min = int(np.min(eye_pts[:, 0])) - 5
        x_max = int(np.max(eye_pts[:, 0])) + 5
        y_min = int(np.min(eye_pts[:, 1])) - 5
        y_max = int(np.max(eye_pts[:, 1])) + 5

        # Clamp to image bounds
        h, w = gray.shape
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)

        if x_max <= x_min or y_max <= y_min:
            return None

        crop = gray[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, self.EYE_CROP_SIZE)
        # Normalise lighting
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        crop = clahe.apply(crop)
        return crop
