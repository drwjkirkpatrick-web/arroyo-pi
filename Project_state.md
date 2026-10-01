# Project State: Arroyo-Pi (Pupillography on Pi/Jetson)

> **Last updated:** 2026-09-30
> **Current phase:** build
> **Overall health:** green

---

## 1. Goal (1–2 sentences)
A Raspberry Pi / Jetson Orin Nano pupillography system that quantifies Arroyo's
1924 asthenocoria (adrenal pupillary) test using 940nm IR illumination, a NoIR
camera, OpenCV pupil tracking, and a Python analysis core. Hardware successor to
arroyo-visionos — same algorithm, no Apple platform restrictions.

## 2. Current Status
### Done
- [x] Analysis core ported from arroyo-visionos (analysis.py — 381 lines)
- [x] OpenCV pupil/iris tracker (tracker.py — dark-pupil + Hough iris + PIR)
- [x] CLI app with demo/live/calibrate modes (main.py)
- [x] 25 unit tests — all pass (synthetic pupillograms, all grades, edge cases)
- [x] Hardware bill of materials and setup guide (hardware/README.md)
- [x] 4 generated PDF figures (pupillograms, escape detection, architecture, grading)
- [x] LaTeX research paper — 13 pages, 31 grounded citations, compiled to PDF
- [x] pyproject.toml, README.md, .gitignore
- [x] Project_state.md
- [x] Git init + commit (15f3fb8) + GitHub repo + push to main

### In Progress
- [ ] Nothing mid-flight

### Not Started
- [ ] Live camera testing with actual IR LED hardware
- [ ] 3D-printable camera mount STL

## 3. Architecture & Key Decisions
| Decision | Rationale | Date |
|---|---|---|
| 940nm IR illumination | Invisible to human retina — does not trigger pupil constriction, safe for sustained use | 2026-09-30 |
| Dark-pupil detection method | Under IR illumination, pupil is darkest region in eye image; well-documented in eye-tracking literature | 2026-09-30 |
| PIR calibration (pupil-to-iris ratio, HVID 11.7mm) | No hardware calibrator needed; iris diameter constant during light response; validated in smartphone pupillometry | 2026-09-30 |
| Arducam IMX462 NoIR for Pi / IMX219 NoIR for Jetson | IMX462: ultra-low-light (0.01 lux), NoIR, 1080p@50fps. IMX219: native Jetson driver support. Both < $30. | 2026-09-30 |
| Algorithm identical to arroyo-visionos | Same escape threshold (15%), sustain (2s), grading bands — verified by 25 tests | 2026-09-30 |
| Python-only (no Swift) | Runs natively on Pi/Jetson; no Mac required; OpenCV available via pip | 2026-09-30 |

## 4. Blockers & Risks
- **Risk:** Consumer NoIR camera under 940nm may have lower sensitivity than
  at 850nm → mitigation: IMX462 has good NIR sensitivity; 850nm fallback.
- **Risk:** Dark-pupil detection on very dark irises → mitigation: quality
  warnings in analysis; face-detection ROI narrows search area.
- **Risk:** Patient motion artifacts → mitigation: chin rest recommended;
  tracker smoothing (0.7 factor); invalid-frame interpolation.

## 5. Next Step (only ONE)
> **Next:** Verify LaTeX paper compiled successfully, then git init + commit
> + create GitHub repo + push.

## 6. Environment & Tooling Notes
- Project root: `~/projects/arroyo-pi/`
- Python: system python3 (stdlib + numpy + opencv)
- Tests: `PYTHONPATH=src python3 tests/test_analysis.py` (25 tests, all pass)
- Demo: `PYTHONPATH=src python3 -m arroyopi.main demo --grade fatigue`
- LaTeX: pdflatex available (texlive-latex-extra, texlive-science installed)
- Figures: generated with matplotlib 3.11.2, saved as PDF in docs/figures/
- Citation ledger: `~/.hermes/cache/citations/arroyo-pi-ledger.json` (31 sources)

## 7. Recent Session Log
- 2026-09-30: Reviewed arroyo-visionos repo (analysis core, RESEARCH.md, tests).
  Researched hardware (940nm IR, IMX462, IMX219, Pi/Jetson camera pipelines).
  Built arroyo-pi: analysis.py, tracker.py, main.py, 25 tests (all pass),
  hardware BOM, 4 PDF figures, README, pyproject.toml. LaTeX paper delegated
  to subagent for compilation.

## 8. References
- Parent project: `~/projects/arroyo-visionos/` (Apple Vision Pro version)
- Citation ledger: `~/.hermes/cache/citations/arroyo-pi-ledger.json` (31 sources)
- Figures: `docs/figures/fig1_grades.pdf`, `fig2_escape_detection.pdf`,
  `fig3_architecture.pdf`, `fig4_grading_bands.pdf`