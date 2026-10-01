#!/usr/bin/env python3
"""
analysis.py — Arroyo asthenocoria analysis core (Python-native port).

Direct descendant of arroyo-visionos/reference_core/arroyo_core.py, adapted
for real-time embedded use on Raspberry Pi / Jetson Orin Nano. The algorithm
is unchanged; only the data interface differs (PupilSample with pupil_px /
iris_px from the live tracker instead of pre-calibrated mm).

Clinical background: RESEARCH.md and the original arroyo-visionos RESEARCH.md.
- Arroyo's test (1924): sustained side-light across the eye; a healthy pupil
  holds constriction; the positive "asthenocaria" sign is redilation or
  pulsation despite the continuing light.
- Grading thresholds are functional-medicine bands, NOT clinically validated.
  The program never diagnoses — it quantifies pupillary behavior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── Constants (match arroyo-visionos reference_core) ──────────────────────
ESCAPE_THRESHOLD = 0.15          # redilation ≥ 15% of amplitude = escape
ESCAPE_SUSTAIN_S = 2.0           # must persist ≥ 2 s
LATENCY_DROP = 0.05              # 5% drop below baseline = constriction onset
LATENCY_SUSTAIN_SAMPLES = 3      # consecutive samples to confirm latency
MAX_CONSTRICTION_WINDOW_S = 5.0  # search window for peak constriction
BASELINE_WINDOW_S = 2.0          # pre-stimulus baseline window
MIN_AMPLITUDE_FRACTION = 0.05    # min constriction for valid reflex
OSC_MIN_AMPLITUDE = 0.10         # min swing for fasciculation counting
MIN_VALID_SAMPLE_FRACTION = 0.50 # <50% valid frames = reject
HOLD_HEALTHY_S = 20.0
HOLD_MILD_S = 10.0
HOLD_FATIGUE_S = 5.0
PULSATION_ESCAPE_S = 5.0
PULSATION_OSC_HZ = 0.3
DEFAULT_IRIS_MM = 11.7           # adult-average horizontal visible iris diameter
DETREND_WINDOW_S = 3.0

GRADE_LABELS = {
    "held": "Constriction maintained — negative Arroyo sign (healthy)",
    "healthy": "Held constriction ≥ 20 s — healthy band",
    "mild": "Escape after 10–20 s — mild disruption band",
    "fatigue": "Escape after 5–10 s — fatigue band",
    "exhaustion": "Escape < 5 s — exhaustion band",
    "noReflex": "No reliable constriction detected",
    "insufficientData": "Insufficient data for analysis",
    "unknown": "Not measurable",
}


# ── Data types ────────────────────────────────────────────────────────────
@dataclass
class PupilSample:
    """One frame's measurement for one eye."""
    t: float
    pupil_px: float
    iris_px: float
    valid: bool = True


@dataclass
class EyeMetrics:
    """Complete analysis result for one eye."""
    baseline_pir: float = 0.0
    min_pir: float = 0.0
    latency_s: Optional[float] = None
    t_constriction_s: Optional[float] = None
    amplitude_pir: float = 0.0
    constriction_percent: float = 0.0
    escaped: bool = False
    escape_time_s: Optional[float] = None
    hold_time_s: Optional[float] = None
    max_redilation_percent: float = 0.0
    holding_index: float = 0.0
    oscillation_count: int = 0
    oscillation_hz: float = 0.0
    baseline_mm: float = 0.0
    min_diameter_mm: float = 0.0
    valid_sample_fraction: float = 1.0
    grade: str = "unknown"
    grade_label: str = ""
    quality_warnings: List[str] = field(default_factory=list)


@dataclass
class SessionReport:
    """Both eyes + symmetry comparison."""
    left: Optional[EyeMetrics] = None
    right: Optional[EyeMetrics] = None
    hold_delta_s: Optional[float] = None
    amplitude_delta_percent: Optional[float] = None
    overall_grade: str = "unknown"
    overall_label: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────
def median(values: List[float]) -> float:
    if not values:
        raise ValueError("median of empty list")
    xs = sorted(values)
    n = len(xs)
    mid = n // 2
    return xs[mid] if n % 2 == 1 else 0.5 * (xs[mid - 1] + xs[mid])


def running_median(xs: List[float], window: int) -> List[float]:
    n = len(xs)
    out = [0.0] * n
    half = max(1, window // 2)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = median(xs[lo:hi])
    return out


# ── Core analysis ─────────────────────────────────────────────────────────
def _interpolate_invalid(samples: List[PupilSample]) -> List[Tuple[float, float]]:
    """Convert samples to clean (t, PIR) series with linear interpolation of gaps."""
    pts = [(s.t, s.pupil_px / s.iris_px) for s in samples
           if s.valid and s.iris_px > 0]
    if len(pts) < 2:
        return pts
    filled: List[Tuple[float, float]] = []
    for i in range(len(pts) - 1):
        (t0, p0), (t1, p1) = pts[i], pts[i + 1]
        filled.append((t0, p0))
        for s in samples:
            if t0 < s.t < t1 and (not s.valid or s.iris_px <= 0):
                frac = (s.t - t0) / (t1 - t0) if t1 > t0 else 0.0
                filled.append((s.t, p0 + frac * (p1 - p0)))
    filled.append(pts[-1])
    filled.sort(key=lambda p: p[0])
    return filled


def _count_oscillations(pir_post: List[float], duration_s: float,
                        fs_hint: float, amplitude: float) -> Tuple[int, float]:
    if duration_s <= 0 or amplitude <= 0 or len(pir_post) < 4:
        return 0, 0.0
    window = max(3, int(round(DETREND_WINDOW_S * fs_hint)))
    trend = running_median(pir_post, window)
    resid = [p - tr for p, tr in zip(pir_post, trend)]
    band = OSC_MIN_AMPLITUDE * amplitude
    swings = 0
    direction = 0
    for r in resid:
        if direction == 0:
            if r >= band:
                direction = -1
            elif r <= -band:
                direction = 1
        elif direction == 1:
            if r >= band:
                swings += 1
                direction = -1
        else:
            if r <= -band:
                swings += 1
                direction = 1
    cycles = swings // 2
    return cycles, cycles / duration_s if duration_s > 0 else 0.0


def grade_eye(hold_time_s: Optional[float], escaped: bool,
              oscillation_hz: float, recording_s: float) -> Tuple[str, str]:
    if hold_time_s is None:
        return ("held", f"Constriction maintained for the full {recording_s:.0f} s "
                         "— negative Arroyo sign (healthy band).")
    if hold_time_s >= HOLD_HEALTHY_S:
        return ("healthy", f"Held constriction {hold_time_s:.0f} s — healthy band "
                           f"(≥ {HOLD_HEALTHY_S:.0f} s).")
    if hold_time_s >= HOLD_MILD_S:
        return ("mild", f"Constriction failed after {hold_time_s:.0f} s — mild "
                        f"disruption band ({HOLD_MILD_S:.0f}–{HOLD_HEALTHY_S:.0f} s).")
    if hold_time_s >= HOLD_FATIGUE_S:
        return ("fatigue", f"Constriction failed after {hold_time_s:.0f} s — fatigue "
                           f"band ({HOLD_FATIGUE_S:.0f}–{HOLD_MILD_S:.0f} s).")
    if oscillation_hz >= PULSATION_OSC_HZ:
        return ("exhaustion", "Immediate pulsation and dilation — exhaustion band "
                              "(< 5 s with oscillations ≥ 0.3 Hz).")
    return ("exhaustion", f"Constriction failed within {hold_time_s:.1f} s — "
                          "exhaustion band (< 5 s).")


def analyze_eye(samples: List[PupilSample], t0: float,
                iris_mm: float = DEFAULT_IRIS_MM,
                fs_hint: float = 30.0) -> EyeMetrics:
    """Analyze one eye's pupillogram. t0 = stimulus onset time (s)."""
    m = EyeMetrics()
    if not samples:
        m.grade = "insufficientData"
        m.grade_label = "No samples recorded."
        return m

    total = len(samples)
    valid_count = sum(1 for s in samples if s.valid and s.iris_px > 0)
    m.valid_sample_fraction = valid_count / total
    if m.valid_sample_fraction < MIN_VALID_SAMPLE_FRACTION:
        m.grade = "insufficientData"
        m.grade_label = "Less than half of the frames contained a valid pupil."
        return m

    series = _interpolate_invalid(samples)
    if len(series) < 5:
        m.grade = "insufficientData"
        m.grade_label = "Too few samples to analyze."
        return m
    ts = [p[0] for p in series]
    pir = [p[1] for p in series]

    # baseline
    base_pts = [p for p in series if t0 - BASELINE_WINDOW_S <= p[0] < t0]
    if not base_pts:
        base_pts = [p for p in series if p[0] < ts[0] + BASELINE_WINDOW_S]
        if base_pts:
            m.quality_warnings.append("no pre-stimulus baseline; used first 2 s")
    m.baseline_pir = median([p[1] for p in base_pts]) if base_pts else pir[0]

    # max constriction
    win = [p for p in series if t0 <= p[0] <= t0 + MAX_CONSTRICTION_WINDOW_S]
    if not win:
        win = [p for p in series if p[0] >= t0]
    if not win:
        m.grade = "insufficientData"
        m.grade_label = "No post-stimulus samples."
        return m
    min_pt = min(win, key=lambda p: p[1])
    m.min_pir = min_pt[1]
    t_min = min_pt[0]

    m.amplitude_pir = m.baseline_pir - m.min_pir
    m.constriction_percent = (
        100.0 * m.amplitude_pir / m.baseline_pir
        if m.baseline_pir > 0 else 0.0)
    m.baseline_mm = m.baseline_pir * iris_mm
    m.min_diameter_mm = m.min_pir * iris_mm

    if m.amplitude_pir < MIN_AMPLITUDE_FRACTION * m.baseline_pir:
        m.grade = "noReflex"
        m.grade_label = f"No reliable constriction (amplitude {m.constriction_percent:.1f}%)."
        return m

    # latency
    for i, (t, p) in enumerate(series):
        if t < t0:
            continue
        if p <= m.baseline_pir * (1.0 - LATENCY_DROP):
            if all(series[j][1] <= m.baseline_pir * (1.0 - LATENCY_DROP)
                   for j in range(i, min(i + LATENCY_SUSTAIN_SAMPLES, len(series)))):
                m.latency_s = t - t0
                break
    m.t_constriction_s = t_min - t0

    # escape
    post = [p for p in series if p[0] >= t_min]
    if not post:
        m.grade = "insufficientData"
        m.grade_label = "Recording ended at maximum constriction."
        return m
    rel = [(p[0], (p[1] - m.min_pir) / m.amplitude_pir) for p in post]
    t_end = rel[-1][0]

    m.escaped = False
    m.escape_time_s = None
    for i, (t, r) in enumerate(rel):
        if r >= ESCAPE_THRESHOLD:
            t_sustain_end = t + ESCAPE_SUSTAIN_S
            window = [rr for (tt, rr) in rel if t <= tt <= t_sustain_end]
            if window and min(window) >= ESCAPE_THRESHOLD:
                m.escaped = True
                m.escape_time_s = t - t0
                break

    if m.escaped and m.escape_time_s is not None:
        m.hold_time_s = m.escape_time_s - (t_min - t0)
    else:
        m.hold_time_s = None

    m.max_redilation_percent = 100.0 * max(r for _, r in rel)
    within = sum(1 for _, r in rel if r < ESCAPE_THRESHOLD)
    m.holding_index = within / len(rel)

    m.oscillation_count, m.oscillation_hz = _count_oscillations(
        [p[1] for p in post], t_end - t_min, fs_hint, m.amplitude_pir)

    m.grade, m.grade_label = grade_eye(
        m.hold_time_s, m.escaped, m.oscillation_hz, t_end - t_min)

    # quality warnings
    post_stim = [s for s in samples if s.t >= t0]
    span = post_stim[-1].t - t0 if post_stim else 0.0
    if span < 30.0:
        m.quality_warnings.append("post-stimulus recording shorter than 30 s")
    if span > 0:
        rate = len(post_stim) / span
        if rate < 15.0:
            m.quality_warnings.append("low sample rate (< 15 fps)")
    for a, b in zip(samples, samples[1:]):
        if b.t - a.t > 1.0:
            m.quality_warnings.append("gap > 1 s in the recording")
            break

    return m


def analyze_session(left: Optional[List[PupilSample]],
                    right: Optional[List[PupilSample]], t0: float,
                    iris_mm: float = DEFAULT_IRIS_MM,
                    fs_hint: float = 30.0) -> SessionReport:
    """Analyze both eyes and compare symmetry."""
    report = SessionReport()
    report.left = analyze_eye(left, t0, iris_mm, fs_hint) if left else None
    report.right = analyze_eye(right, t0, iris_mm, fs_hint) if right else None

    graded = [m for m in (report.left, report.right)
              if m and m.grade not in ("noReflex", "insufficientData", "unknown")]
    if report.left and report.right:
        if report.left.hold_time_s is not None and report.right.hold_time_s is not None:
            report.hold_delta_s = abs(report.left.hold_time_s - report.right.hold_time_s)
        report.amplitude_delta_percent = abs(
            report.left.constriction_percent - report.right.constriction_percent)

    if not graded:
        report.overall_grade = "insufficientData"
        report.overall_label = "No valid eye measurements in this session."
        return report

    order = ["held", "healthy", "mild", "fatigue", "exhaustion"]
    worst = graded[0]
    for g in graded[1:]:
        if order.index(g.grade) > order.index(worst.grade):
            worst = g
    report.overall_grade = worst.grade
    report.overall_label = worst.grade_label
    return report


# ── Synthetic generator (for demo/test mode without hardware) ─────────────
def synth_eye(escape_hold_s: Optional[float] = None,
              osc_hz: float = 0.0, latency_s: float = 0.22,
              t_constriction_s: float = 1.9,
              constriction_frac: float = 0.45,
              pre_s: float = 5.0, post_s: float = 90.0,
              fs: float = 30.0, iris_px: float = 120.0,
              iris_mm: float = DEFAULT_IRIS_MM,
              noise: float = 0.004, seed: int = 42,
              blink_frames: Tuple[int, ...] = ()) -> Tuple[List[PupilSample], float]:
    """Generate one eye's synthetic Arroyo recording for demo mode."""
    import random
    rng = random.Random(seed)
    t0 = pre_s
    baseline_pir = 0.33
    amp = baseline_pir * constriction_frac
    n = int((pre_s + post_s) * fs)
    samples: List[PupilSample] = []
    for k in range(n):
        t = k / fs
        pir_val = baseline_pir + rng.gauss(0, noise * baseline_pir)
        if t >= t0:
            dt = t - t0
            if dt >= latency_s:
                tau = (t_constriction_s - latency_s) / 3.0
                c = 1.0 - math.exp(-(dt - latency_s) / tau)
                pir_val = baseline_pir - amp * min(1.0, c)
            if escape_hold_s is not None:
                t_esc = t_constriction_s + escape_hold_s
                if dt >= t_esc:
                    rise = 1.0 / (1.0 + math.exp(-0.35 * (dt - t_esc - 6.0)))
                    pir_val += amp * 0.75 * rise
            if osc_hz > 0 and dt >= t_constriction_s:
                pir_val += amp * 0.25 * math.sin(2 * math.pi * osc_hz * (dt - t_constriction_s))
        pupil_px = pir_val * iris_px
        samples.append(PupilSample(t=t, pupil_px=max(1.0, pupil_px),
                                    iris_px=iris_px, valid=True))
    for bf in blink_frames:
        if 0 <= bf < len(samples):
            samples[bf] = PupilSample(t=samples[bf].t, pupil_px=0, iris_px=0, valid=False)
    return samples, t0