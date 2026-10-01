# 🔦 Arroyo-Pi — Asthenocoria Meter

**A 1924 physical sign, measured with $60 of hardware.**

Arroyo-Pi turns the century-old Arroyo pupillary test — the *asthenocoria*
sign — into a quantified pupillogram on a Raspberry Pi or Jetson Orin Nano.
A 940nm infrared LED (invisible to the human eye) illuminates the pupil while
an OpenCV tracker measures how long constriction holds under sustained light.

> Healthy pupils hold their ground. Arroyo's sign is the pupil that fights —
> constricting, slipping, pulsing — and dilates anyway, in spite of the light.

This is the hardware successor to [arroyo-visionos](https://github.com/drwjkirkpatrick-web/arroyo-visionos)
(Apple Vision Pro). Same analysis algorithm, but no enterprise entitlements,
no $3,500 headset, no platform restrictions — just a Pi, a NoIR camera, and
an IR LED.

---

## What it measures

Each exam produces a full pupillogram and metrics per eye:

- **Baseline diameter** (mm, iris-calibrated via PIR method)
- **Constriction amplitude** (mm and %)
- **Latency** — light-on to constriction onset
- **Time-to-escape** — *the core Arroyo metric*: seconds from light-on until
  the pupil sustains a redilation beyond 15% of its constricted range
- **Pulsations** — count and rate of miosis/mydriasis oscillations
- **Max redilation & holding index**
- **Left/right symmetry** (consensual reflex comparison)

### Grading overlay

| Time to escape | Functional-medicine reading |
|---|---|
| Held (none within window) | Healthy sustained constriction |
| ≥ 20 s | Healthy |
| 10–20 s | Mild disruption |
| 5–10 s | Fatigue pattern |
| < 5 s | Exhaustion pattern |

**These thresholds are functional-medicine convention, not clinically
validated.** Pupillary escape is also a normal physiological phenomenon.
This is a screening and trend instrument — it never diagnoses.

## Quick start

### Demo mode (no hardware needed)

```bash
git clone https://github.com/drwjkirkpatrick-web/arroyo-pi.git
cd arroyo-pi
PYTHONPATH=src python3 -m arroyopi.main demo --grade fatigue
```

Produces a synthetic pupillogram, analysis, and CSV/JSON export — exercises
the full analysis pipeline without a camera.

### Live exam (requires camera + IR LED)

```bash
pip3 install opencv-python-headless numpy

# Calibrate camera positioning
python3 -m arroyopi.main calibrate --eye right

# Run exam
python3 -m arroyopi.main live --eye right --duration 90
```

See [`hardware/README.md`](hardware/README.md) for the full bill of materials,
wiring, and camera setup instructions.

## Architecture

```
src/arroyopi/
  analysis.py     Arroyo analysis core (escape detection, grading, oscillation)
  tracker.py      OpenCV dark-pupil + iris tracker (Haar cascades, Hough, contours)
  main.py         CLI: demo / live / calibrate modes
tests/
  test_analysis.py    25 unit tests (synthetic pupillograms, all grades, edge cases)
docs/
  arroyo_pi_research_paper.tex   LaTeX research paper with grounded citations
  figures/                       Generated PDF charts (pupillograms, architecture, grading)
hardware/
  README.md       Bill of materials, wiring, camera setup
```

## Hardware

| Component | Example | Cost |
|---|---|---|
| Camera | Arducam IMX462 NoIR (Pi) or IMX219 NoIR (Jetson) | ~$25–30 |
| IR LED | 940nm, 3W (invisible to human eye) | ~$3 |
| Computing | Raspberry Pi 4/5 or Jetson Orin Nano | $35–150 |
| **Total** | | **~$60–120** |

The 940nm wavelength is critical: it is invisible to the human retina, so the
IR illumination does not trigger pupil constriction — the camera can see the
pupil in a darkened room without interfering with the test.

## The story behind the name

Carlos F. Arroyo (1892–1928) described the sign in 1924 and named it
*asthenocaria* — from the Greek *asthenēs*, weak — the weak, "lazy" iris that
constricts to light but cannot hold the contraction. The eponym survives in
medical dictionaries as **Arroyo's sign, synonym: asthenocoria**.

## License

MIT © 2026 Walker Kirkpatrick