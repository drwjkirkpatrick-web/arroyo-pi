#!/usr/bin/env python3
"""
main.py — Arroyo Meter CLI for Raspberry Pi / Jetson Orin Nano.

Usage:
  python3 -m arroyopi.main live          # Live camera exam (requires camera + IR LED)
  python3 -m arroyopi.main demo           # Synthetic demo (no hardware needed)
  python3 -m arroyopi.main demo --grade fatigue  # Specific demo pattern
  python3 -m arroyopi.main calibrate      # Show camera preview with tracking overlay

The exam protocol follows Arroyo's 1924 test:
  1. Dim room, eyes adapt ~1 min
  2. Camera aimed at patient's eye (or face for face-detection mode)
  3. 5 s pre-stimulus baseline
  4. Light on (penlight across eye at ~45°, ~20 cm) — press SPACE
  5. Hold 90 s, tracking pupil diameter frame-by-frame
  6. Analysis + graded pupillogram + CSV export
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

from .analysis import (
    PupilSample, EyeMetrics, SessionReport, analyze_eye, analyze_session,
    synth_eye, GRADE_LABELS, DEFAULT_IRIS_MM,
)
# Tracker imports cv2 lazily — only needed for live/calibrate modes

# ── Demo mode ─────────────────────────────────────────────────────────────
DEMO_PATTERNS = {
    "healthy": {"escape_hold_s": None, "osc_hz": 0.0},
    "mild": {"escape_hold_s": 15.0, "osc_hz": 0.0},
    "fatigue": {"escape_hold_s": 10.0, "osc_hz": 0.3},
    "exhaustion": {"escape_hold_s": 2.0, "osc_hz": 0.4},
}


def run_demo(grade: str = "fatigue", output_dir: str = ".") -> SessionReport:
    """Run a synthetic demo exam (no camera hardware needed)."""
    pattern = DEMO_PATTERNS.get(grade, DEMO_PATTERNS["fatigue"])
    print(f"\n{'='*60}")
    print(f"  Arroyo Meter — DEMO MODE ({grade})")
    print(f"{'='*60}")
    print(f"  Generating synthetic pupillogram...")
    print(f"  Pattern: escape_hold={pattern['escape_hold_s']}s, "
          f"osc={pattern['osc_hz']}Hz")

    left_samples, t0 = synth_eye(
        escape_hold_s=pattern["escape_hold_s"],
        osc_hz=pattern["osc_hz"],
        seed=42,
    )
    right_samples, _ = synth_eye(
        escape_hold_s=pattern["escape_hold_s"],
        osc_hz=pattern["osc_hz"],
        seed=99,
    )

    report = analyze_session(left_samples, right_samples, t0)

    _print_report(report)
    _export_csv(left_samples, right_samples, t0, report, output_dir, demo=True)
    return report


# ── Live mode ─────────────────────────────────────────────────────────────
def run_live(eye_side: str = "right", duration: int = 90,
             iris_mm: float = DEFAULT_IRIS_MM, output_dir: str = ".",
             show_preview: bool = True, camera_index: int = 0,
             use_face_detection: bool = True) -> SessionReport:
    """Run a live camera exam."""
    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV (cv2) is required for live mode. Install with:")
        print("  pip3 install opencv-python-headless  (or opencv-python for GUI)")
        sys.exit(1)

    from .tracker import PupilTracker

    print(f"\n{'='*60}")
    print(f"  Arroyo Meter — LIVE EXAM")
    print(f"{'='*60}")
    print(f"  Eye: {eye_side} | Duration: {duration}s | Iris: {iris_mm}mm")
    print(f"  Face detection: {'ON' if use_face_detection else 'OFF (macro lens)'}")
    print()

    # Try camera backends in order of preference
    cap = _open_camera(camera_index)
    if cap is None:
        print("ERROR: Could not open camera. Check CSI/USB connection.")
        print("For Jetson CSI: use GStreamer pipeline (see hardware/README.md)")
        sys.exit(1)

    tracker = PupilTracker(
        use_face_detection=use_face_detection,
        eye_side=eye_side,
    )

    samples: List[PupilSample] = []
    t_start = time.time()
    t_light_on: Optional[float] = None
    light_on_pressed = False

    print("  PRE-STIMULUS BASELINE (5 s)")
    print("  → Shine penlight across eye at ~45° and press SPACE")
    print("  → Press 'q' to abort")

    baseline_s = 5.0
    recording = True

    while recording:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Frame read error")
            continue

        t = time.time() - t_start
        sample = tracker.process_frame(frame, t)
        samples.append(sample)

        if show_preview:
            vis = tracker.draw_overlay(frame)
            # Status text
            status = "BASELINE" if t_light_on is None else f"LIGHT ON ({t - t_light_on:.1f}s)"
            color = (0, 255, 0) if t_light_on is None else (0, 200, 255)
            cv2.putText(vis, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if t_light_on is not None and (t - t_light_on) >= duration:
                cv2.putText(vis, "COMPLETE", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)

            cv2.imshow("Arroyo Meter", vis)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            t_light_on = time.time() - t_start
            light_on_pressed = True
            print(f"  → LIGHT ON at t={t_light_on:.2f}s")
        elif key == ord('q'):
            recording = False

        # Auto-end after duration
        if t_light_on is not None and (t - t_light_on) >= duration:
            recording = False

        # Auto-start if baseline passes and light not pressed (optional)
        if t_light_on is None and t >= baseline_s and not light_on_pressed:
            pass  # Wait for user to press SPACE

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    if t_light_on is None:
        print("\n  Exam aborted — no light stimulus was recorded.")
        return SessionReport()

    print(f"\n  Recorded {len(samples)} samples over {samples[-1].t:.1f}s")
    print("  Analyzing...")

    report = analyze_session(samples, [sample] if False else None, t_light_on,
                             iris_mm=iris_mm, fs_hint=30.0)
    # For single-eye mode, put samples in the correct eye
    if eye_side == "right":
        report = SessionReport(right=analyze_eye(samples, t_light_on, iris_mm),
                               overall_grade="", overall_label="")
        report.overall_grade = report.right.grade
        report.overall_label = report.right.grade_label
    else:
        report = SessionReport(left=analyze_eye(samples, t_light_on, iris_mm),
                               overall_grade="", overall_label="")
        report.overall_grade = report.left.grade
        report.overall_label = report.left.grade_label

    _print_report(report)
    _export_csv(samples, None, t_light_on, report, output_dir)
    return report


def _open_camera(index: int):
    """Open camera with fallback for different platforms."""
    import cv2

    # Try GStreamer pipeline for Jetson CSI camera
    try:
        pipeline = (
            f"nvarguscamerasrc sensor-id={index} ! "
            "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"  Camera: Jetson CSI (GStreamer, sensor {index})")
            return cap
    except Exception:
        pass

    # Fallback: standard VideoCapture (USB webcam or Pi camera V4L2)
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"  Camera: V4L2/USB (index {index})")
        return cap

    return None


# ── Calibrate mode ────────────────────────────────────────────────────────
def run_calibrate(eye_side: str = "right", camera_index: int = 0,
                  use_face_detection: bool = True):
    """Show live camera with tracking overlay for positioning."""
    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV (cv2) is required for calibrate mode.")
        sys.exit(1)

    from .tracker import PupilTracker

    print(f"\n  Arroyo Meter — CALIBRATION PREVIEW")
    print(f"  Aim camera at eye. Press 'q' to exit.\n")

    cap = _open_camera(camera_index)
    if cap is None:
        print("ERROR: Could not open camera.")
        sys.exit(1)

    tracker = PupilTracker(use_face_detection=use_face_detection, eye_side=eye_side)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        vis = tracker.draw_overlay(frame)
        cv2.putText(vis, "CALIBRATION — press 'q' to exit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Arroyo Meter — Calibrate", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Output ────────────────────────────────────────────────────────────────
def _print_report(report: SessionReport):
    print(f"\n{'─'*60}")
    print(f"  RESULTS")
    print(f"{'─'*60}")

    for label, eye in [("LEFT", report.left), ("RIGHT", report.right)]:
        if eye is None:
            continue
        print(f"\n  {label} EYE:")
        print(f"    Grade: {eye.grade} — {eye.grade_label}")
        print(f"    Baseline: {eye.baseline_mm:.2f} mm (PIR {eye.baseline_pir:.3f})")
        print(f"    Min diameter: {eye.min_diameter_mm:.2f} mm")
        print(f"    Constriction: {eye.constriction_percent:.1f}%")
        if eye.latency_s is not None:
            print(f"    Latency: {eye.latency_s:.3f} s")
        if eye.hold_time_s is not None:
            print(f"    Hold time: {eye.hold_time_s:.1f} s")
        else:
            print(f"    Hold time: HELD (full recording)")
        print(f"    Max redilation: {eye.max_redilation_percent:.1f}%")
        print(f"    Oscillations: {eye.oscillation_count} ({eye.oscillation_hz:.2f} Hz)")
        print(f"    Holding index: {eye.holding_index:.2f}")
        print(f"    Valid frames: {eye.valid_sample_fraction*100:.0f}%")
        if eye.quality_warnings:
            for w in eye.quality_warnings:
                print(f"    ⚠ {w}")

    if report.left and report.right:
        print(f"\n  SYMMETRY:")
        if report.hold_delta_s is not None:
            print(f"    Hold delta: {report.hold_delta_s:.1f} s")
        if report.amplitude_delta_percent is not None:
            print(f"    Amplitude delta: {report.amplitude_delta_percent:.1f}%")

    print(f"\n  OVERALL: {report.overall_grade} — {report.overall_label}")
    print(f"{'─'*60}\n")


def _export_csv(left_samples: List[PupilSample], right_samples: Optional[List[PupilSample]],
                t0: float, report: SessionReport, output_dir: str, demo: bool = False):
    """Export pupillogram data and report to CSV/JSON."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "demo" if demo else "exam"
    csv_path = os.path.join(output_dir, f"arroyo_{prefix}_{ts}.csv")
    json_path = os.path.join(output_dir, f"arroyo_{prefix}_{ts}.json")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "left_pupil_px", "left_iris_px", "left_valid",
                         "right_pupil_px", "right_iris_px", "right_valid"])
        left = left_samples or []
        right = right_samples or []
        for i in range(max(len(left), len(right))):
            ls = left[i] if i < len(left) else PupilSample(0, 0, 0, False)
            rs = right[i] if i < len(right) else PupilSample(0, 0, 0, False)
            writer.writerow([f"{ls.t:.4f}", f"{ls.pupil_px:.2f}", f"{ls.iris_px:.2f}",
                             int(ls.valid), f"{rs.pupil_px:.2f}", f"{rs.iris_px:.2f}",
                             int(rs.valid)])

    def eye_to_dict(eye: Optional[EyeMetrics]) -> Optional[dict]:
        if eye is None:
            return None
        return {
            "grade": eye.grade, "grade_label": eye.grade_label,
            "baseline_mm": round(eye.baseline_mm, 3),
            "min_diameter_mm": round(eye.min_diameter_mm, 3),
            "constriction_percent": round(eye.constriction_percent, 2),
            "latency_s": eye.latency_s, "hold_time_s": eye.hold_time_s,
            "max_redilation_percent": round(eye.max_redilation_percent, 2),
            "oscillation_count": eye.oscillation_count,
            "oscillation_hz": round(eye.oscillation_hz, 3),
            "holding_index": round(eye.holding_index, 3),
            "valid_sample_fraction": round(eye.valid_sample_fraction, 3),
            "quality_warnings": eye.quality_warnings,
        }

    with open(json_path, 'w') as f:
        json.dump({
            "timestamp": ts, "t_light_on": t0,
            "left": eye_to_dict(report.left),
            "right": eye_to_dict(report.right),
            "overall_grade": report.overall_grade,
            "overall_label": report.overall_label,
        }, f, indent=2)

    print(f"  Exported: {csv_path}")
    print(f"  Exported: {json_path}")


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog="arroyopi",
        description="Arroyo Asthenocoria Meter — pupillography on Pi/Jetson",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # demo
    p_demo = sub.add_parser("demo", help="Synthetic demo (no hardware)")
    p_demo.add_argument("--grade", choices=list(DEMO_PATTERNS.keys()),
                        default="fatigue", help="Demo pattern grade")
    p_demo.add_argument("--output-dir", default=".", help="Output directory")

    # live
    p_live = sub.add_parser("live", help="Live camera exam")
    p_live.add_argument("--eye", choices=["left", "right"], default="right")
    p_live.add_argument("--duration", type=int, default=90, help="Recording seconds")
    p_live.add_argument("--iris-mm", type=float, default=DEFAULT_IRIS_MM)
    p_live.add_argument("--output-dir", default=".")
    p_live.add_argument("--no-preview", action="store_true", help="Disable GUI preview")
    p_live.add_argument("--camera", type=int, default=0, help="Camera index")
    p_live.add_argument("--no-face-detection", action="store_true",
                        help="Disable face detection (macro lens mode)")

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Camera preview with tracking overlay")
    p_cal.add_argument("--eye", choices=["left", "right"], default="right")
    p_cal.add_argument("--camera", type=int, default=0)
    p_cal.add_argument("--no-face-detection", action="store_true")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo(grade=args.grade, output_dir=args.output_dir)
    elif args.command == "live":
        run_live(
            eye_side=args.eye, duration=args.duration,
            iris_mm=args.iris_mm, output_dir=args.output_dir,
            show_preview=not args.no_preview, camera_index=args.camera,
            use_face_detection=not args.no_face_detection,
        )
    elif args.command == "calibrate":
        run_calibrate(
            eye_side=args.eye, camera_index=args.camera,
            use_face_detection=not args.no_face_detection,
        )


if __name__ == "__main__":
    main()