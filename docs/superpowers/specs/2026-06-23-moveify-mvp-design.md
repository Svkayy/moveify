# moveify MVP Buildout — Design Spec

**Date:** 2026-06-23
**Status:** Approved (pending written-spec review)
**Repo:** `Svkayy/moveify`
**Working branch:** `mvp-buildout`

## Context

`moveify` is a dance-sync analysis tool: given two dance videos (a learner and a
reference), it runs MediaPipe pose estimation on both, aligns them by audio
cross-correlation, compares limb angles frame-by-frame, and outputs a sync score,
a side-by-side comparison video, a JSON report, and a matplotlib graph. It has a
CLI (`dance.py`) and a Tkinter GUI (`dance_gui.py`).

A codebase assessment found ~70% works (pose extraction, audio cross-correlation,
angle comparison, scoring, comparison video, report, graph are all real), but:

- **Headline bug:** the audio offset is computed (`dance.py:290`) but never
  applied — the "synchronized" comparison video and the frame-by-frame
  comparison are NOT time-aligned. The core promise is half-wired.
- README overpromises an unimplemented "real-time beat-synced coaching" feature.
- `true_multi_person_tracker.py` (858 lines, the most sophisticated code) is
  standalone, not integrated.
- Dead code: unused `soundfile` import, an orphaned `crosscorrelate.praat` never
  called, and `test_pose_tracker.py` is a stub (documentation, not a test).
- Tests are plain functions (not pytest); 2 are real unit tests (angle calc,
  sync score), 2 are integration tests that silently skip without video files.
- No demo videos shipped.

Distinct tech stack from anomalai: MediaPipe + librosa/scipy (audio DSP) +
OpenCV + matplotlib + Tkinter — no cloud, no CoreML.

## Goal

Bring `moveify` to a solid, portfolio-credible MVP.

**Definition of done:**
1. Runs end-to-end on macOS producing a **truly time-aligned** dance-sync
   comparison (the offset bug fixed).
2. A **Gemini AI coaching review** generated from the sync results, env-configured
   and degrading gracefully without a key.
3. A real **pytest** suite (unit + the offset regression + AI reviewer mocked).
4. An accurate **README** with a Mermaid architecture + tech-stack diagram.
5. Demo proof on real clips in `docs/demo/`.
6. Multi-person tracker and dead code removed.

## Guiding principle: close holes, don't camouflage

Fix the real bug, complete the real integration, remove dead code. The Gemini
coach is real (env-driven, graceful fallback), never faked. Demo artifacts come
from the pipeline actually running.

## Out of scope

- Real-time webcam coaching, multi-dancer tracking (the multi-person tracker is
  cut, not integrated).
- The Praat audio path.
- Exercising Gemini against a committed key (the coach degrades to a templated
  summary without one; live key only used at demo-capture time).

## Decisions (from brainstorming)

- **AI coach input:** TEXT from the structured sync data (overall %, per-limb
  scores, worst-synced moments). No multimodal frames for the MVP.
- **GUI freeze:** FIX it — run analysis on a background thread so Tkinter stays
  responsive with live progress.
- **Multi-person tracker:** CUT (`true_multi_person_tracker.py` + stub
  `test_pose_tracker.py`).

## Approach (run-first, then test & polish)

### WS1 — Fix the core sync bug (apply the audio offset)
The offset returned by audio cross-correlation must be applied so the
later-starting video is trimmed/shifted before (a) the frame-by-frame angle
comparison and (b) writing the side-by-side comparison video. Add a regression
test that proves a known synthetic offset is applied (the aligned series line up).

### WS2 — Gemini AI coach/reviewer (`coach.py`)
New module that takes the structured sync results and returns a short coaching
review (overall grade, which limbs/moments lagged, 2–3 actionable tips). Reads
`GEMINI_API_KEY` and `GEMINI_MODEL` (default `gemini-2.5-flash-lite`) from env via
python-dotenv. When no key/SDK is available, returns a deterministic templated
summary built from the numbers (graceful fallback, clearly labeled). Wire the
review into the CLI output, the JSON report, and the GUI. Add `.env.example`.

### WS3 — Cut cruft
Delete `true_multi_person_tracker.py` and `test_pose_tracker.py`. Remove the
unused `soundfile` import. Remove `crosscorrelate.praat` and any references (or
move under a clearly-labeled `extras/` and note it as out-of-scope — prefer
deletion). Trim the README's unimplemented "real-time coaching" claim.

### WS4 — Real tests (pytest)
Convert the existing plain-function tests to pytest. Keep the 2 real unit tests
(angle calculation, sync score). Add: offset-application regression (WS1),
per-limb scoring, and the AI reviewer with Gemini **mocked** (no live calls).
Add `pytest.ini` and a CI workflow running the non-model tests.

### WS5 — README + tech diagram
Rewrite to match reality: MediaPipe pose → audio cross-correlation alignment →
limb-angle scoring → Gemini coach → comparison video/report/graph. Mermaid
architecture + tech-stack diagram. Accurate setup, usage, testing, structure.

### WS6 — Demo proof
Source two short, license-safe dance clips (same routine, two takes), run the
pipeline, and capture the comparison video, the sync graph, and the AI coaching
review into `docs/demo/`. Live Gemini key used only for this capture (not
committed).

### WS7 — Verify + push
Tests green, app runs, demo captured → commit and push to `Svkayy/moveify`,
open a PR. Includes the GUI threading fix.

## Components & boundaries (target structure)

- `dance.py` — core pipeline; fix offset application; emit structured results for
  the coach.
- `coach.py` *(new)* — Gemini coaching review + templated fallback (env-driven).
- `dance_gui.py` — Tkinter GUI; background-thread the analysis; show the review.
- `pose_tracker.py` — webcam pose demo (kept, standalone).
- `.env.example` *(new)* — `GEMINI_API_KEY`, `GEMINI_MODEL`.
- `tests/` — pytest: angle, sync score, offset regression, per-limb, coach (mocked).
- `pytest.ini`, `.github/workflows/tests.yml` *(new)*.
- `README.md` *(rewrite)* + `docs/diagrams` (Mermaid in README) + `docs/demo/`.
- **Removed:** `true_multi_person_tracker.py`, `test_pose_tracker.py`,
  `crosscorrelate.praat`.

## Risks & open questions

- **Offset direction/units:** cross-correlation lag is in audio samples; applying
  it to video needs conversion to frames (lag_seconds × fps). The regression test
  must pin the conversion so a real offset aligns correctly.
- **Demo clips:** must be short and license-safe; pose estimation needs clearly
  visible full-body dancers. If sourced clips track poorly, fall back to two takes
  of a simple movement.
- **Gemini coach without key:** the templated fallback must read as a real summary,
  not a placeholder.
