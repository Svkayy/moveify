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
    base = [float(i) for i in range(10)]
    shifted = [-1.0, -1.0, -1.0] + base   # series2 is 'base' delayed by 3 frames
    a1, a2 = apply_offset(base, shifted, 3)
    assert a1[0] == a2[0]
    assert len(a1) == len(a2)
    assert a1[:5] == a2[:5]


def test_apply_offset_negative_trims_series2():
    a1, a2 = apply_offset([0.,1.,2.,3.,4.], [9.,9.,0.,1.,2.], -2)
    assert len(a1) == len(a2)
    assert a1[:3] == a2[:3]


def test_apply_offset_zero_truncates_to_min():
    a1, a2 = apply_offset([1.,2.,3.], [1.,2.], 0)
    assert len(a1) == len(a2) == 2
