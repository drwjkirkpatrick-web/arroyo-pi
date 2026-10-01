# Hardware Guide — Arroyo-Pi Pupillography System

## Overview

The Arroyo-Pi system captures sustained pupillary constriction dynamics using
near-infrared (NIR) illumination and a NoIR (infrared-sensitive) camera. The
hardware is designed to be low-cost (~$60–120 total), portable, and safe for
sustained use near the eye.

## Bill of Materials

### Camera (choose one)

| Component | Price | Notes |
|---|---|---|
| **Arducam IMX462 NoIR** (B0333) | ~$30 | Best for Pi. Ultra-low-light (0.01 lux), NoIR (no IR-cut filter), 1080p@50fps, M16 lens f/0.95. [17] |
| Raspberry Pi Camera Module 3 NoIR | ~$25 | Official Pi NoIR camera, 12MP, autofocus. Good but less low-light sensitivity than IMX462. |
| **IMX219 NoIR** (Waveshare/Arducam) | ~$20 | Best for Jetson Orin Nano. 8MP, native driver support, NoIR variant available. |
| Arducam IMX477 NoIR | ~$50 | 12MP HQ camera, Jetson-compatible with driver install. Higher quality but more expensive. |

### Illumination

| Component | Price | Notes |
|---|---|---|
| **940nm IR LED (3W, 5mm or SMD)** | ~$2–5 | Completely invisible to the human eye — does NOT trigger pupil constriction. This is critical for the Arroyo test. [18][19] |
| 850nm IR LED (alternative) | ~$2 | Faint red glow visible to humans — usable but less ideal than 940nm. |
| Constant-current LED driver | ~$3 | Or use a 100Ω resistor with 3.3V/5V supply. |

> **Why 940nm?** The human retina is insensitive to 940nm near-infrared light.
> The pupil will not constrict in response to IR illumination, so the camera
> can see the pupil in a darkened room without interfering with the test.
> 850nm has a faint visible red glow; 940nm is completely invisible. [18][19]

### Computing

| Component | Price | Notes |
|---|---|---|
| **Raspberry Pi 4B (4GB)** or **Pi 5** | $35–60 | Runs OpenCV pupil tracking at 30fps. |
| **Jetson Orin Nano 8GB** | $150 | Already available in this lab. GPU-accelerated OpenCV. |
| MicroSD card (32GB+) | $8 | |

### Optics (optional, for macro/close-up eye imaging)

| Component | Price | Notes |
|---|---|---|
| 20D aspheric condensing lens | $5–10 | For retinal imaging (fundus). Not needed for pupillography. [14] |
| Adjustable macro lens (M12) | $5 | If using face-detection mode at 30–50cm distance. |

### Enclosure and Mounting

| Component | Price | Notes |
|---|---|---|
| 3D-printed camera mount | ~$2 | STL files in `hardware/` (coming soon). |
| Tripod or flex arm | $10 | For stable positioning. |
| Jumper wires (for LED) | $2 | |

**Total estimated cost: $60–120** (depending on computing platform)

## Camera Setup

### Raspberry Pi (IMX462)

```bash
# Enable camera interface
sudo raspi-config  # → Interface Options → Camera → Enable

# Test camera
libcamera-hello --camera 0

# The arroyo-pi software uses V4L2 backend by default:
python3 -m arroyopi.main calibrate --eye right
```

### Jetson Orin Nano (IMX219)

```bash
# Enable IMX219 camera driver
sudo /opt/nvidia/jetson-io/jetson-io.py
# → Configure Jetson 24-pin CSI Connector → Camera IMX219 Dual

# Verify
lsmod | grep imx219
v4l2-ctl --list-devices

# The software auto-detects GStreamer (nvarguscamerasrc) for CSI cameras:
python3 -m arroyopi.main calibrate --eye right
```

## IR LED Wiring

```
3.3V (Pi Pin 1) ──[100Ω resistor]──[940nm IR LED (+)]──[IR LED (−)]── GND (Pin 6)
```

Or use a constant-current driver for stable output. The LED should be
positioned to illuminate the eye from below or to the side at a shallow
angle (10–25° off-axis), not directly into the eye. [19]

## Safety Notes

- **IR exposure limits**: Irradiance below 10 mW/cm² is considered safe for
  chronic IR exposure in the 720–1400nm range (Sliney & Myron, 1980). [19]
- Use a 3W LED at 20–30cm distance — well within safe limits.
- The system is for clinical screening by a trained practitioner, not
  self-administered home use without supervision.
- All patient data stays on-device; exports are explicit and manual.

## Operating Modes

### Face-detection mode (default)
Camera positioned 30–80cm from the patient's face. Haar cascade detects
the face, extracts the eye ROI automatically. Works with standard lens.

### Macro lens mode (`--no-face-detection`)
Camera positioned close (10–20cm) to the eye. No face detection needed;
the full frame is the eye image. Requires macro/close-focus lens.

## Recommended Test Environment

- Dim or darkened room (pupils start semi-dilated after ~1 min adaptation)
- Patient seated, chin rest or head support recommended for stability
- Examiner holds external penlight (white light) at 45° to the eye, ~20cm
- IR LED illuminates the eye from below/side for camera visibility
- The penlight (visible white) is the stimulus; the IR LED is for imaging only