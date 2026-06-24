"""
Unit tests for DanceSyncAnalyzer math methods.
These tests do NOT trigger MediaPipe/TF-Lite loading (lazy pose init).
"""
import pytest
from dance import DanceSyncAnalyzer, apply_offset


def _p(x, y):
    """Minimal landmark dict matching calculate_angle's expected keys."""
    return {"x": x, "y": y, "z": 0.0, "visibility": 1.0}


def test_calculate_angle_right_angle():
    """Vertex at origin, arms along +x and +y => 90 degrees."""
    a = DanceSyncAnalyzer()
    # p1=(1,0), p2=(0,0) [vertex], p3=(0,1) => 90°
    ang = a.calculate_angle(_p(1, 0), _p(0, 0), _p(0, 1))
    assert abs(ang - 90.0) < 1.0


def test_sync_score_identical_is_max():
    """Identical angle series should yield a score of 100."""
    a = DanceSyncAnalyzer()
    angles = [10.0, 20.0, 30.0, 40.0]
    assert a.calculate_sync_score(angles, angles) >= 99.0


def test_sync_score_90deg_apart_is_50():
    """
    Angles 90° apart produce a score of 50.0.

    The scoring formula uses min(diff, 180-diff)/180 so the floor is 50
    (both 0° and 180° differences map to normalized_diff=0 and 0.5
    respectively, giving scores 100 and 50). With a constant 90° gap the
    normalized_diff is min(90,90)/180 = 0.5, so score = (1-0.5)*100 = 50.
    """
    a = DanceSyncAnalyzer()
    score = a.calculate_sync_score([0.0, 0.0, 0.0], [90.0, 90.0, 90.0])
    assert abs(score - 50.0) < 1.0


def test_apply_offset_aligns_shifted_series():
    # offset > 0 => series1 (video1) starts earlier => trim series1's head
    # series1 has 3 leading frames before content matches series2
    base = [float(i) for i in range(10)]
    shifted = base  # series2 is base (no lead-in)
    series1_with_lead = [-1.0, -1.0, -1.0] + base
    a1, a2 = apply_offset(series1_with_lead, shifted, 3)
    assert a1[0] == a2[0]
    assert len(a1) == len(a2)
    assert a1[:5] == a2[:5]


def test_apply_offset_negative_trims_series2():
    # offset < 0 => series2 (video2) starts earlier/has lead-in => trim series2's head by -offset_frames
    # series2 has 2 leading frames before content matches series1
    a1, a2 = apply_offset([0.,1.,2.,3.,4.], [9.,9.,0.,1.,2.], -2)
    assert len(a1) == len(a2)
    assert a1[:3] == a2[:3]  # [0.,1.,2.] == [0.,1.,2.]


def test_apply_offset_zero_truncates_to_min():
    a1, a2 = apply_offset([1.,2.,3.], [1.,2.], 0)
    assert len(a1) == len(a2) == 2


def test_offset_direction_aligns_real_audio_lag():
    import numpy as np
    a = DanceSyncAnalyzer()
    sr = 1000; D = 50
    audio1 = np.zeros(2000); audio1[500:520] = 1.0          # event at 500
    audio2 = np.zeros(2000); audio2[500+D:520+D] = 1.0       # video2 delayed by D
    off = a.find_audio_offset(audio1, audio2, sr)            # expect negative
    # series2 has D frames of lead-in before the shared content T
    T = list(range(1000)); s1 = list(T); s2 = [-1]*D + list(T)
    o1, o2 = apply_offset(s1, s2, off)
    assert o1[:10] == o2[:10], f"offset {off} did not align (wrong trim direction)"


def test_limb_performance_identical_is_high():
    """Identical frame angles => each limb should be perfectly in sync (~100)."""
    a = DanceSyncAnalyzer()
    frame = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]   # 6 limb angles: Left Arm, Right Arm, Left Leg, Right Leg, Left Torso, Right Torso
    perf = a.analyze_limb_performance([frame], [frame])

    # Check structure: dict with 6 limbs
    assert isinstance(perf, dict) and len(perf) == 6

    # Check limb names and value structure
    expected_limbs = {"Left Arm", "Right Arm", "Left Leg", "Right Leg", "Left Torso", "Right Torso"}
    assert set(perf.keys()) == expected_limbs

    # Check each limb result has required keys and score is in valid range
    for name, limb_result in perf.items():
        assert isinstance(limb_result, dict)
        assert "average_score" in limb_result
        assert "average_difference" in limb_result
        assert "frames_analyzed" in limb_result

        score = limb_result["average_score"]
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 100.0

    # Identical inputs => every limb should be perfectly in sync
    for name, limb_result in perf.items():
        assert limb_result["average_score"] >= 99.0, f"Limb {name} score {limb_result['average_score']} should be ~100 for identical input"
        assert limb_result["average_difference"] < 1.0, f"Limb {name} difference should be ~0 for identical input"
        assert limb_result["frames_analyzed"] == 1
