#!/usr/bin/env python3
"""
tracker.py — Real-time pupil and iris tracker using OpenCV.

Dark-pupil detection with near-infrared illumination:
  1. Face detection → eye ROI extraction (Haar cascades)
  2. Iris boundary detection (Hough circles)
  3. Dark-pupil segmentation (adaptive threshold + contour + ellipse fit)
  4. Pupil-to-iris ratio (PIR) computation

Hardware: Raspberry Pi NoIR Camera V3 / Arducam IMX462 / Jetson IMX219-NoIR
Illumination: 940nm IR LED (invisible to human eye, safe for sustained use)
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from .analysis import PupilSample

# ── Re-export for convenience ─────────────────────────────────────────────
__all__ = ["PupilTracker", "TrackingState"]


@dataclass
class TrackingState:
    """Tracks best detection across frames for stability."""
    pupil_center: Optional[Tuple[int, int]] = None
    pupil_radius_px: float = 0.0
    iris_center: Optional[Tuple[int, int]] = None
    iris_radius_px: float = 0.0
    eye_roi: Optional[Tuple[int, int, int, int]] = None
    lost_count: int = 0
    frame_count: int = 0

    @property
    def pupil_diameter_px(self) -> float:
        return 2.0 * self.pupil_radius_px if self.pupil_radius_px > 0 else 0.0

    @property
    def iris_diameter_px(self) -> float:
        return 2.0 * self.iris_radius_px if self.iris_radius_px > 0 else 0.0


class PupilTracker:
    """Real-time pupil tracker for Arroyo pupillary testing.

    Under 940nm IR illumination, the pupil appears as the darkest region
    in the eye image (dark-pupil method). The iris is detected via edge-
    based Hough circle fitting, providing the calibration ruler for the
    pupil-to-iris ratio (PIR) method.
    """

    def __init__(self,
                 use_face_detection: bool = True,
                 eye_side: str = "right",
                 min_pupil_radius: int = 5,
                 max_pupil_radius: int = 80,
                 min_iris_radius: int = 25,
                 max_iris_radius: int = 150,
                 pupil_threshold_offset: int = 15,
                 tracking_smoothing: float = 0.7,
                 lost_threshold: int = 15):
        self.use_face_detection = use_face_detection
        self.eye_side = eye_side
        self.min_pupil_r = min_pupil_radius
        self.max_pupil_r = max_pupil_radius
        self.min_iris_r = min_iris_radius
        self.max_iris_r = max_iris_radius
        self.pupil_thresh_offset = pupil_threshold_offset
        self.smoothing = tracking_smoothing
        self.lost_threshold = lost_threshold
        self.state = TrackingState()
        self._face_cascade = None
        self._eye_cascade = None
        if use_face_detection:
            self._init_cascades()

    def _init_cascades(self):
        try:
            face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            eye_xml = cv2.data.haarcascades + 'haarcascade_eye.xml'
            import os
            if os.path.exists(face_xml):
                self._face_cascade = cv2.CascadeClassifier(face_xml)
            if os.path.exists(eye_xml):
                self._eye_cascade = cv2.CascadeClassifier(eye_xml)
        except Exception:
            self.use_face_detection = False

    def process_frame(self, frame: np.ndarray, t: float) -> PupilSample:
        self.state.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        gray = cv2.equalizeHist(gray)

        roi = self._get_eye_roi(gray)
        if roi is None:
            return self._lost_sample(t)

        x, y, w, h = roi
        eye_img = gray[y:y+h, x:x+w]
        self.state.eye_roi = roi

        self._detect_iris(eye_img)
        pupil_ok = self._detect_pupil(eye_img)

        if not pupil_ok:
            self.state.lost_count += 1
            if self.state.lost_count > self.lost_threshold:
                self.state.pupil_radius_px = 0.0
            return PupilSample(t=t, pupil_px=0, iris_px=0, valid=False)

        self.state.lost_count = 0
        return PupilSample(
            t=t,
            pupil_px=self.state.pupil_diameter_px,
            iris_px=self.state.iris_diameter_px,
            valid=self.state.iris_diameter_px > 0,
        )

    def _get_eye_roi(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = gray.shape
        if self.use_face_detection and self._face_cascade is not None:
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                eye_y = fy + int(fh * 0.15)
                eye_h = int(fh * 0.35)
                if self.eye_side == "right":
                    eye_x = fx + int(fw * 0.1)
                else:
                    eye_x = fx + int(fw * 0.55)
                eye_w = int(fw * 0.35)
                eye_x = max(0, min(eye_x, w - eye_w))
                eye_y = max(0, min(eye_y, h - eye_h))
                return (eye_x, eye_y, eye_w, eye_h)
            if self._eye_cascade is not None:
                eyes = self._eye_cascade.detectMultiScale(
                    gray, scaleFactor=1.3, minNeighbors=3,
                    minSize=(30, 30), maxSize=(200, 200))
                if len(eyes) > 0:
                    eye = min(eyes, key=lambda e: e[0]) if self.eye_side == "right" \
                        else max(eyes, key=lambda e: e[0])
                    ex, ey, ew, eh = eye
                    pad = 10
                    return (max(0, ex-pad), max(0, ey-pad),
                            min(w, ew+2*pad), min(h, eh+2*pad))
        if self.state.eye_roi is not None:
            return self.state.eye_roi
        return (0, 0, w, h)

    def _detect_iris(self, eye_img: np.ndarray) -> bool:
        h, w = eye_img.shape
        if h < 20 or w < 20:
            return False
        blur = cv2.GaussianBlur(eye_img, (5, 5), 0)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=min(w, h) * 0.5,
            param1=80, param2=30,
            minRadius=self.min_iris_r, maxRadius=self.max_iris_r)
        if circles is not None and len(circles) > 0:
            cx, cy, r = circles[0][0]
            if self.state.iris_radius_px > 0 and self.smoothing > 0:
                r = self.smoothing * self.state.iris_radius_px + (1 - self.smoothing) * r
            self.state.iris_center = (int(cx), int(cy))
            self.state.iris_radius_px = float(r)
            return True
        return self.state.iris_radius_px > 0

    def _detect_pupil(self, eye_img: np.ndarray) -> bool:
        h, w = eye_img.shape
        if h < 20 or w < 20:
            return False
        mean_val = float(np.mean(eye_img))
        thresh_val = max(10, int(mean_val - self.pupil_thresh_offset))
        _, binary = cv2.threshold(eye_img, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_r, best_cx, best_cy, best_score = 0.0, 0, 0, 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            r_min_sq = 3.14159 * self.min_pupil_r ** 2
            r_max_sq = 3.14159 * self.max_pupil_r ** 2
            if area < r_min_sq or area > r_max_sq:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < self.min_pupil_r or radius > self.max_pupil_r:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 1:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.4:
                continue
            x0 = max(0, int(cx - radius)); x1 = int(cx + radius)
            y0 = max(0, int(cy - radius)); y1 = int(cy + radius)
            darkness = 1.0 - (float(np.mean(eye_img[y0:y1, x0:x1])) / 255.0)
            score = circularity * darkness * min(radius / self.max_pupil_r, 1.0)
            if score > best_score:
                best_score = score
                best_r = radius
                best_cx, best_cy = int(cx), int(cy)

        if best_r > 0:
            if self.state.pupil_radius_px > 0 and self.smoothing > 0:
                best_r = self.smoothing * self.state.pupil_radius_px + (1 - self.smoothing) * best_r
            self.state.pupil_center = (best_cx, best_cy)
            self.state.pupil_radius_px = float(best_r)
            return True
        return self.state.pupil_radius_px > 0 and self.state.lost_count < self.lost_threshold

    def _lost_sample(self, t: float) -> PupilSample:
        self.state.lost_count += 1
        if self.state.lost_count > self.lost_threshold:
            self.state.pupil_radius_px = 0.0
        return PupilSample(t=t, pupil_px=0, iris_px=0, valid=False)

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        vis = frame.copy()
        if self.state.eye_roi is not None:
            x, y, w, h = self.state.eye_roi
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
            if self.state.iris_radius_px > 0 and self.state.iris_center:
                ic = (x + self.state.iris_center[0], y + self.state.iris_center[1])
                cv2.circle(vis, ic, int(self.state.iris_radius_px), (0, 200, 255), 1)
            if self.state.pupil_radius_px > 0 and self.state.pupil_center:
                pc = (x + self.state.pupil_center[0], y + self.state.pupil_center[1])
                cv2.circle(vis, pc, int(self.state.pupil_radius_px), (0, 0, 255), 2)
                pir = self.state.pupil_diameter_px / max(self.state.iris_diameter_px, 1)
                cv2.putText(vis, f"PIR={pir:.3f} d={self.state.pupil_diameter_px:.1f}px",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return vis

    def reset(self):
        self.state = TrackingState()