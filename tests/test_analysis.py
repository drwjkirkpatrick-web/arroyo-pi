#!/usr/bin/env python3
"""
test_analysis.py — Verification suite for the Arroyo Pi analysis core.

Each test builds a synthetic pupillogram with known ground truth and asserts
the analyzer recovers it. Mirrors the arroyo-visionos reference_core suite.

Run:  python3 -m pytest tests/test_analysis.py -v
  or: python3 tests/test_analysis.py
"""
import math
import unittest
from arroyopi.analysis import (
    PupilSample, analyze_eye, analyze_session, synth_eye,
    DEFAULT_IRIS_MM, ESCAPE_THRESHOLD,
)


class TestSynthEye(unittest.TestCase):
    """Verify the synthetic generator produces analyzable data."""

    def test_healthy_generation(self):
        samples, t0 = synth_eye(escape_hold_s=None, seed=42)
        self.assertEqual(t0, 5.0)
        self.assertGreater(len(samples), 2000)
        # All samples should be valid
        self.assertTrue(all(s.valid for s in samples))

    def test_fatigue_generation(self):
        samples, t0 = synth_eye(escape_hold_s=8.0, osc_hz=0.3, seed=42)
        self.assertGreater(len(samples), 2000)
        # PIR should vary (not flat)
        pirs = [s.pupil_px / s.iris_px for s in samples if s.valid]
        self.assertGreater(max(pirs) - min(pirs), 0.05)

    def test_blink_invalidation(self):
        samples, _ = synth_eye(seed=42, blink_frames=(100, 101, 102))
        self.assertFalse(samples[100].valid)
        self.assertFalse(samples[101].valid)
        self.assertFalse(samples[102].valid)


class TestHealthyPupil(unittest.TestCase):
    """Constriction that holds for the full window → healthy grade."""

    def setUp(self):
        self.samples, self.t0 = synth_eye(escape_hold_s=None, seed=42)
        self.res = analyze_eye(self.samples, self.t0)

    def test_no_escape(self):
        self.assertFalse(self.res.escaped)
        self.assertIsNone(self.res.escape_time_s)
        self.assertIn(self.res.grade, ("held", "healthy"))

    def test_constriction(self):
        self.assertGreater(self.res.constriction_percent, 30.0)
        self.assertLess(self.res.constriction_percent, 55.0)

    def test_valid_baseline(self):
        self.assertGreater(self.res.baseline_mm, 2.0)
        self.assertLess(self.res.baseline_mm, 6.0)

    def test_holding_index_high(self):
        self.assertGreater(self.res.holding_index, 0.80)


class TestMildEscape(unittest.TestCase):
    """Escape after ~15s → mild disruption band."""

    def setUp(self):
        self.samples, self.t0 = synth_eye(escape_hold_s=15.0, seed=42)
        self.res = analyze_eye(self.samples, self.t0)

    def test_escape_detected(self):
        self.assertTrue(self.res.escaped)
        self.assertIsNotNone(self.res.escape_time_s)

    def test_grade_mild(self):
        self.assertEqual(self.res.grade, "mild")

    def test_escape_time_in_band(self):
        self.assertGreaterEqual(self.res.escape_time_s, 12.0)
        self.assertLessEqual(self.res.escape_time_s, 22.0)


class TestFatigueOscillation(unittest.TestCase):
    """Escape after ~7s hold with oscillation → fatigue band (5-10s hold)."""

    def setUp(self):
        # hold=10 with osc=0.3 gives hold_time ~7.4s → fatigue band
        # (oscillation amplitude 25% crosses 15% escape threshold slightly
        #  before the logistic escape, so effective hold is ~7s)
        self.samples, self.t0 = synth_eye(escape_hold_s=10.0, osc_hz=0.3, seed=42)
        self.res = analyze_eye(self.samples, self.t0)

    def test_escape_detected(self):
        self.assertTrue(self.res.escaped)
        self.assertIsNotNone(self.res.escape_time_s)

    def test_grade_fatigue(self):
        self.assertEqual(self.res.grade, "fatigue")

    def test_oscillations(self):
        self.assertGreater(self.res.oscillation_count, 5)
        self.assertGreater(self.res.oscillation_hz, 0.1)


class TestExhaustion(unittest.TestCase):
    """Rapid escape < 5s → exhaustion band."""

    def setUp(self):
        self.samples, self.t0 = synth_eye(escape_hold_s=2.0, osc_hz=0.4, seed=42)
        self.res = analyze_eye(self.samples, self.t0)

    def test_escape_detected(self):
        self.assertTrue(self.res.escaped)
        self.assertIsNotNone(self.res.escape_time_s)

    def test_grade_exhaustion(self):
        self.assertEqual(self.res.grade, "exhaustion")

    def test_rapid_escape(self):
        self.assertLess(self.res.escape_time_s, 8.0)


class TestNoReflex(unittest.TestCase):
    """Flat pupil (no constriction) → noReflex grade."""

    def test_flat_pupil(self):
        samples, t0 = synth_eye(
            escape_hold_s=None, constriction_frac=0.001, seed=42)
        res = analyze_eye(samples, t0)
        self.assertEqual(res.grade, "noReflex")


class TestInsufficientData(unittest.TestCase):
    """Too few or mostly-invalid samples → insufficientData."""

    def test_empty(self):
        res = analyze_eye([], 0.0)
        self.assertEqual(res.grade, "insufficientData")

    def test_too_few(self):
        samples = [PupilSample(t=0.0, pupil_px=50, iris_px=120),
                   PupilSample(t=0.1, pupil_px=48, iris_px=120)]
        res = analyze_eye(samples, 0.0)
        self.assertEqual(res.grade, "insufficientData")

    def test_mostly_invalid(self):
        samples, t0 = synth_eye(seed=42)
        # Invalidate 70% of samples
        for i in range(0, len(samples), 10):
            for j in range(7):
                if i + j < len(samples):
                    samples[i + j] = PupilSample(
                        t=samples[i + j].t, pupil_px=0, iris_px=0, valid=False)
        res = analyze_eye(samples, t0)
        self.assertEqual(res.grade, "insufficientData")


class TestSessionSymmetry(unittest.TestCase):
    """Two-eye session analysis."""

    def test_both_healthy(self):
        left, t0 = synth_eye(escape_hold_s=None, seed=42)
        right, _ = synth_eye(escape_hold_s=None, seed=99)
        report = analyze_session(left, right, t0)
        self.assertIn(report.overall_grade, ("held", "healthy"))

    def test_disagreeing_eyes(self):
        left, t0 = synth_eye(escape_hold_s=None, seed=42)
        right, _ = synth_eye(escape_hold_s=2.0, osc_hz=0.4, seed=99)
        report = analyze_session(left, right, t0)
        self.assertEqual(report.overall_grade, "exhaustion")

    def test_both_fatigue(self):
        left, t0 = synth_eye(escape_hold_s=10.0, osc_hz=0.3, seed=42)
        right, _ = synth_eye(escape_hold_s=10.0, osc_hz=0.3, seed=99)
        report = analyze_session(left, right, t0)
        self.assertEqual(report.overall_grade, "fatigue")


class TestQualityWarnings(unittest.TestCase):
    """Quality warnings for problematic recordings."""

    def test_low_sample_rate(self):
        samples, t0 = synth_eye(fs=10.0, seed=42)
        res = analyze_eye(samples, t0, fs_hint=10.0)
        # Should still analyze but may have warnings
        # Low sample rate warning is in the post-stimulus rate check
        # With 10 fps, the rate is 10 which is < 15
        self.assertTrue(any("sample rate" in w for w in res.quality_warnings))

    def test_short_recording(self):
        samples, t0 = synth_eye(post_s=20.0, seed=42)
        res = analyze_eye(samples, t0)
        self.assertTrue(any("shorter than 30" in w for w in res.quality_warnings))


if __name__ == "__main__":
    unittest.main(verbosity=2)