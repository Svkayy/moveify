# moveify MVP Buildout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `moveify` to a portfolio-credible MVP: a truly time-aligned dance-sync comparison (fix the offset bug), a Gemini AI coach, a real pytest suite, an accurate README with diagrams, and demo proof on real clips. Multi-person tracker + dead code removed.

**Architecture:** A `DanceSyncAnalyzer` (in `dance.py`) runs MediaPipe pose on two videos, aligns them via audio cross-correlation, compares limb angles, and scores sync. A new `coach.py` turns the structured results into a Gemini coaching review (with a templated fallback). A Tkinter GUI (`dance_gui.py`) drives it on a background thread.

**Tech Stack:** Python 3.11, MediaPipe, OpenCV, librosa + scipy (audio DSP), numpy, matplotlib, Tkinter, google-generativeai (Gemini), pytest.

## Global Constraints

- **Python 3.11** exactly.
- **Platform: macOS** target.
- **No live external calls in tests** — Gemini is always mocked. No secrets committed.
- **Close holes, don't camouflage** — fix the real offset bug; the coach is real with a graceful templated fallback when no key; never fabricate output.
- **Gemini config** is env-driven: `GEMINI_API_KEY`, `GEMINI_MODEL` (default `gemini-2.5-flash-lite`), read via python-dotenv. `.env` is git-ignored, never committed.
- **Working branch:** `mvp-buildout`. Commit after every task.
- The audio cross-correlation lag from `find_audio_offset` is in **audio samples**; converting to video frames is `round(offset_samples / sr * fps)`.

## File Structure

- `dance.py` *(modify)* — apply the audio offset; expose a pure `apply_offset(...)`; emit structured results for the coach.
- `coach.py` *(new)* — Gemini coaching review + deterministic templated fallback; env-driven.
- `dance_gui.py` *(modify)* — run analysis on a background thread; show the coach review.
- `.env.example` *(new)* — `GEMINI_API_KEY`, `GEMINI_MODEL`.
- `tests/` *(new)* — `test_dance.py` (angle, sync score, offset regression, per-limb), `test_coach.py` (mocked Gemini + fallback), `conftest.py` if needed.
- `pytest.ini` *(new)*, `.github/workflows/tests.yml` *(new)*.
- `README.md` *(rewrite)* + `docs/demo/` *(new)*.
- **Delete:** `true_multi_person_tracker.py`, `test_pose_tracker.py`, `crosscorrelate.praat`.

---

## Task 1: Cut the multi-person tracker and dead code

**Files:**
- Delete: `true_multi_person_tracker.py`, `test_pose_tracker.py`, `crosscorrelate.praat`
- Modify: `dance.py` (remove unused `soundfile` import), `README.md` (remove the unimplemented "real-time beat-synced coaching" claim — minimal touch; full rewrite is Task 7)

- [ ] **Step 1: Confirm nothing imports the files being deleted**

Run: `grep -rnE "true_multi_person_tracker|crosscorrelate|import soundfile|soundfile" --include='*.py' . | grep -v tests/`
Expected: only the `import soundfile` line in `dance.py` and (possibly) self-references. If `true_multi_person_tracker` is imported anywhere, STOP and report.

- [ ] **Step 2: Delete the files**

```bash
git rm true_multi_person_tracker.py test_pose_tracker.py crosscorrelate.praat
```

- [ ] **Step 3: Remove the unused soundfile import in dance.py**

Find `import soundfile` (near the top imports) and delete that line.

- [ ] **Step 4: Verify dance.py still imports**

Run: `python -c "import ast; ast.parse(open('dance.py').read()); print('dance.py parses')"`
Expected: `dance.py parses` (full import needs deps; this is a syntax check until the venv is ready).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: cut standalone multi-person tracker and dead code"
```

---

## Task 2: pytest scaffolding — port the real unit tests

**Files:**
- Create: `tests/test_dance.py`, `pytest.ini`
- Reference (do not delete yet): `test_dance_sync.py` (port its 2 real unit tests, then remove it)

**Interfaces:**
- Consumes: `DanceSyncAnalyzer.calculate_angle(p1, p2, p3)` and `calculate_sync_score(angles1, angles2)` from `dance.py` (read their real signatures first).

- [ ] **Step 1: Read the real signatures**

Read `dance.py` `calculate_angle` (~line 123), `calculate_limb_angles` (~144), `calculate_sync_score` (~163). Note exact input shapes (e.g. `calculate_angle` takes three landmark dicts with x/y keys; `calculate_sync_score` takes two equal-length angle lists, returns 0–100).

- [ ] **Step 2: Write pytest unit tests mirroring the existing real ones**

```python
# tests/test_dance.py
import math
from dance import DanceSyncAnalyzer

def _p(x, y):  # minimal landmark dict; adjust keys to match calculate_angle
    return {"x": x, "y": y, "z": 0.0, "visibility": 1.0}

def test_calculate_angle_right_angle():
    a = DanceSyncAnalyzer()
    # vertex at origin, arms along +x and +y => 90 degrees
    ang = a.calculate_angle(_p(1, 0), _p(0, 0), _p(0, 1))
    assert abs(ang - 90.0) < 1.0

def test_sync_score_identical_is_max():
    a = DanceSyncAnalyzer()
    angles = [10.0, 20.0, 30.0, 40.0]
    assert a.calculate_sync_score(angles, angles) >= 99.0

def test_sync_score_opposite_is_low():
    a = DanceSyncAnalyzer()
    s = a.calculate_sync_score([0.0, 0.0, 0.0], [180.0, 180.0, 180.0])
    assert s <= 50.0
```

(Adjust `_p` keys and the exact assertions to the real implementation read in Step 1. If `DanceSyncAnalyzer.__init__` loads MediaPipe eagerly and that's slow/heavy, note it as a concern — Task 3 may need a lazy-init split.)

- [ ] **Step 3: Write pytest.ini**

```ini
[pytest]
testpaths = tests
addopts = -q
markers =
    integration: needs real video files (skipped by default)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_dance.py -v`
Expected: PASS (3 tests). If `import dance` is heavy because `__init__` builds MediaPipe, see Step 5.

- [ ] **Step 5: If needed, make MediaPipe init lazy**

If importing/instantiating `DanceSyncAnalyzer` triggers a heavy MediaPipe model load that the angle/score tests don't need, refactor `__init__` so the `mp.solutions.pose.Pose` object is built lazily on first use (a `_pose` property), preserving behavior. Re-run Step 4.

- [ ] **Step 6: Remove the old plain-function test file**

```bash
git rm test_dance_sync.py
```

- [ ] **Step 7: Commit**

```bash
git add tests/test_dance.py pytest.ini dance.py
git commit -m "test: port dance unit tests to pytest"
```

---

## Task 3: Fix the audio-offset application (the headline bug)

**Files:**
- Modify: `dance.py` — add `apply_offset(...)`; call it in `analyze_dance_sync`; honor it in `create_comparison_video`
- Test: `tests/test_dance.py` (add the offset regression)

**Interfaces:**
- Produces: `apply_offset(series1: list, series2: list, offset_frames: int) -> tuple[list, list]` — trims the leading `offset_frames` from whichever series starts earlier so the two align; pure, no video. `offset_frames > 0` means series1 starts earlier (trim its head); `< 0` trims series2's head. Both returned series have equal length (truncate the longer tail).

- [ ] **Step 1: Write the failing regression test**

```python
# add to tests/test_dance.py
from dance import apply_offset

def test_apply_offset_aligns_shifted_series():
    base = [float(i) for i in range(10)]
    shifted = [-1.0, -1.0, -1.0] + base   # series2 is 'base' delayed by 3 frames
    # series1=base starts 3 frames earlier => offset_frames = +3
    a1, a2 = apply_offset(base, shifted, 3)
    assert a1[0] == a2[0]                  # aligned at the same content
    assert len(a1) == len(a2)
    assert a1[:5] == a2[:5]

def test_apply_offset_zero_is_noop_to_min_length():
    a1, a2 = apply_offset([1.0,2.0,3.0], [1.0,2.0], 0)
    assert len(a1) == len(a2) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_dance.py -k apply_offset -v`
Expected: FAIL — `ImportError: cannot import name 'apply_offset'`

- [ ] **Step 3: Implement `apply_offset`**

```python
# dance.py (module-level pure function)
def apply_offset(series1, series2, offset_frames):
    """Trim the leading frames of whichever series starts earlier so the two
    align, then truncate both to equal length. offset_frames>0 => series1 leads."""
    if offset_frames > 0:
        series1 = series1[offset_frames:]
    elif offset_frames < 0:
        series2 = series2[-offset_frames:]
    n = min(len(series1), len(series2))
    return series1[:n], series2[:n]
```

- [ ] **Step 4: Wire it into `analyze_dance_sync`**

In `analyze_dance_sync` (dance.py ~277), after computing `offset` (samples) and extracting `landmarks1`/`landmarks2`, convert and apply:

```python
# offset is in audio samples; convert to video frames using video1's fps
fps = self._video_fps(video1_path)   # add a small helper using cv2.VideoCapture CAP_PROP_FPS
offset_frames = round(offset / sr1 * fps)
landmarks1, landmarks2 = apply_offset(landmarks1, landmarks2, offset_frames)
```

Add the `_video_fps(path)` helper (open with cv2.VideoCapture, read `CAP_PROP_FPS`, default 30.0 if 0/unavailable, release). Ensure the angle comparison and `create_comparison_video` operate on the aligned data (pass `offset_frames` into `create_comparison_video` so the side-by-side frames are also aligned — skip the leading frames of the earlier video there too).

- [ ] **Step 5: Run the suite**

Run: `python -m pytest tests/test_dance.py -v`
Expected: PASS (unit + offset regression).

- [ ] **Step 6: Commit**

```bash
git add dance.py tests/test_dance.py
git commit -m "fix: apply audio offset to align pose series and comparison video"
```

---

## Task 4: Gemini AI coach (`coach.py`)

**Files:**
- Create: `coach.py`, `.env.example`
- Modify: `dance.py` (call the coach in `analyze_dance_sync`, add review to results/report + CLI print)
- Test: `tests/test_coach.py`

**Interfaces:**
- Produces: `generate_review(results: dict) -> dict` returning `{"review": str, "source": "gemini"|"fallback"}`. `results` contains `overall_sync` (float), `limb_performance` (dict label→score), and optionally `worst_moments` (list). Uses `GEMINI_API_KEY`/`GEMINI_MODEL` from env; falls back to a deterministic templated review when no key or the SDK/call fails.
- Produces: `build_fallback_review(results: dict) -> str` (pure; used by the fallback path and directly testable).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_coach.py
import coach

RESULTS = {"overall_sync": 72.5,
           "limb_performance": {"left_arm": 55.0, "right_arm": 88.0, "torso": 91.0}}

def test_fallback_review_is_deterministic_and_mentions_worst_limb():
    text = coach.build_fallback_review(RESULTS)
    assert "left_arm" in text          # the weakest limb is called out
    assert "72" in text or "73" in text  # overall score referenced

def test_generate_review_uses_fallback_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = coach.generate_review(RESULTS)
    assert out["source"] == "fallback"
    assert isinstance(out["review"], str) and out["review"]

def test_generate_review_uses_gemini_when_key_present(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(coach, "_gemini_complete",
                        lambda prompt: "Great work! Focus on your left arm timing.",
                        raising=False)
    out = coach.generate_review(RESULTS)
    assert out["source"] == "gemini"
    assert "left arm" in out["review"].lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_coach.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'coach'`

- [ ] **Step 3: Implement `coach.py`**

```python
# coach.py
from __future__ import annotations
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def build_fallback_review(results: dict) -> str:
    overall = results.get("overall_sync", 0.0)
    limbs = results.get("limb_performance", {}) or {}
    worst = min(limbs, key=limbs.get) if limbs else None
    best = max(limbs, key=limbs.get) if limbs else None
    lines = [f"Overall sync: {overall:.0f}%."]
    if worst:
        lines.append(f"Weakest limb: {worst} ({limbs[worst]:.0f}%) — focus your practice there.")
    if best:
        lines.append(f"Strongest limb: {best} ({limbs[best]:.0f}%) — keep it up.")
    return " ".join(lines)

def _gemini_complete(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
    resp = model.generate_content(prompt)
    return resp.text

def generate_review(results: dict) -> dict:
    if not os.getenv("GEMINI_API_KEY"):
        return {"review": build_fallback_review(results), "source": "fallback"}
    prompt = (
        "You are a supportive dance coach. Based on these sync metrics, give a "
        "short (3-4 sentence) review with one specific tip. Metrics: "
        f"overall={results.get('overall_sync')}, per_limb={results.get('limb_performance')}."
    )
    try:
        return {"review": _gemini_complete(prompt).strip(), "source": "gemini"}
    except Exception:
        return {"review": build_fallback_review(results), "source": "fallback"}
```

- [ ] **Step 4: Write `.env.example`**

```bash
# .env.example — copy to .env. Optional: enables the Gemini AI coach.
# Without a key, moveify uses a templated review built from the sync numbers.
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash-lite
```

- [ ] **Step 5: Wire the coach into `analyze_dance_sync`**

In `dance.py`, after building the results dict (with `overall_sync` and `limb_performance` — confirm the real keys; rename in the coach call if they differ), add:

```python
import coach
results["coach_review"] = coach.generate_review({
    "overall_sync": results["average_sync_score"],          # use the real key
    "limb_performance": results["limb_performance"],         # use the real key
})
```

Print the review in `main()` after the score summary, and ensure `save_analysis_report` includes `coach_review`.

- [ ] **Step 6: Run the suite**

Run: `python -m pytest -q`
Expected: PASS (all tests).

- [ ] **Step 7: Commit**

```bash
git add coach.py .env.example dance.py tests/test_coach.py
git commit -m "feat: add Gemini AI dance coach with templated fallback"
```

---

## Task 5: GUI threading + show the coach review

**Files:**
- Modify: `dance_gui.py`

- [ ] **Step 1: Read the run flow**

Read `dance_gui.py` `start_analysis` (~283) and `run_analysis` (~305). It calls `subprocess.Popen` on the main thread, blocking Tkinter.

- [ ] **Step 2: Run analysis on a background thread**

Wrap the body of `run_analysis` in a `threading.Thread(target=..., daemon=True)` started from `start_analysis`. Marshal log/UI updates back to the Tk main loop with `self.root.after(0, lambda: ...)` (Tkinter is not thread-safe). Disable the Run button while a thread is active; re-enable on completion.

- [ ] **Step 3: Surface the coach review in the GUI**

After the subprocess finishes, read the JSON report's `coach_review.review` (if present) and append it to the log area under a "Coach review" header.

- [ ] **Step 4: Manual smoke (document, do not block CI)**

Note in the report: launching `python dance_gui.py` shows a responsive window; (full run needs videos + deps). No automated GUI test.

- [ ] **Step 5: Commit**

```bash
git add dance_gui.py
git commit -m "fix: run GUI analysis on a background thread and show coach review"
```

---

## Task 6: CI + per-limb test

**Files:**
- Create: `.github/workflows/tests.yml`
- Modify: `tests/test_dance.py` (add a per-limb performance test)

- [ ] **Step 1: Add a per-limb test**

```python
# add to tests/test_dance.py
from dance import DanceSyncAnalyzer

def test_limb_performance_keys_and_range():
    a = DanceSyncAnalyzer()
    # two identical angle-series per frame => high per-limb scores
    perf = a.analyze_limb_performance([[10.0,20.0]], [[10.0,20.0]])  # adjust to real signature
    assert isinstance(perf, dict) and len(perf) >= 1
    for v in perf.values():
        assert 0.0 <= float(v if not isinstance(v, dict) else v.get("score", 0)) <= 100.0
```

(Read `analyze_limb_performance` at `dance.py:358` first and adapt the call/assertion to its real input/output shape.)

- [ ] **Step 2: Write the CI workflow**

```yaml
# .github/workflows/tests.yml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt pytest
      - run: python -m pytest -m "not integration" -q
```

- [ ] **Step 3: Run the suite**

Run: `python -m pytest -m "not integration" -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_dance.py .github/workflows/tests.yml
git commit -m "test: add per-limb test and CI workflow"
```

---

## Task 7: README rewrite + Mermaid diagrams

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README**

Sections: tagline; **Demo** (embed `docs/demo/` artifacts once Task 8 produces them — use the real paths); **Architecture** (Mermaid flowchart: two videos → MediaPipe pose → audio cross-correlation alignment → limb-angle scoring → Gemini coach → comparison video/report/graph); **Tech Stack** (Mermaid or table: MediaPipe, OpenCV, librosa+scipy, matplotlib, Tkinter, google-generativeai); **Setup** (`pip install -r requirements.txt`, `cp .env.example .env` optional for the coach); **Usage** (CLI `python dance.py v1 v2`, GUI `python dance_gui.py`); **Testing** (`pytest -m "not integration"`); **Project structure**. Remove all mention of unimplemented real-time coaching and the deleted multi-person tracker.

- [ ] **Step 2: Verify referenced demo paths exist** (after Task 8) or use placeholders that Task 8 fills.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README with accurate features + diagrams"
```

---

## Task 8: Demo capture (controller/user checkpoint)

**Files:**
- Create: `docs/demo/` (comparison video frame(s), sync graph, coach review)

> Controller sources two short, license-safe dance clips. Live Gemini key (optional) used only here, not committed.

- [ ] **Step 1: Obtain two short clips** with clearly visible full-body dancers (same routine, two takes). Place under a local `samples/` (git-ignored).
- [ ] **Step 2: Run** `python dance.py samples/a.mp4 samples/b.mp4 --output-dir docs/demo`.
- [ ] **Step 3: Capture** the comparison video (or a representative frame), the `sync_scores.png` graph, and the coach review (with a live key for the Gemini path, else the fallback). Save into `docs/demo/`.
- [ ] **Step 4: Commit** (proof only; no key, no raw sample videos unless small + license-safe).

```bash
git add docs/demo
git commit -m "docs: add demo proof (real dance clips)"
```

---

## Task 9: Final verification + push

- [ ] **Step 1:** `python -m pytest -m "not integration"` → all pass.
- [ ] **Step 2:** `python -c "import dance, coach"` → no error.
- [ ] **Step 3:** `grep -rnE "true_multi_person_tracker|soundfile|real-time.*coaching" --include='*.py' .` → empty.
- [ ] **Step 4:** `git push -u origin mvp-buildout`
- [ ] **Step 5:** `gh pr create --title "moveify MVP buildout" --body "..."`

---

## Self-Review

**Spec coverage:** WS1 (offset fix) → Task 3. WS2 (coach) → Task 4. WS3 (cut cruft) → Task 1. WS4 (pytest) → Tasks 2,3,4,6. WS5 (README+diagram) → Task 7. WS6 (demo) → Task 8. WS7 (verify+push, GUI threading) → Tasks 5,9. ✓

**Placeholder scan:** No TBD/implement-later. Where the plan's example test values or keys depend on real `dance.py` shapes (`calculate_angle` keys, `average_sync_score`/`limb_performance` key names, `analyze_limb_performance` signature), each such task starts with a "read the real signature" step and says to adapt — intentional, not a placeholder.

**Type consistency:** `apply_offset` defined/tested in Task 3, used in `analyze_dance_sync`. `generate_review`/`build_fallback_review`/`_gemini_complete` defined in Task 4, consumed by dance.py + GUI (Task 5). `GEMINI_MODEL` default `gemini-2.5-flash-lite` consistent with `.env.example`.
