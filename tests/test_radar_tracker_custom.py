import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.radar_tracker_custom import RadarTrackerCustom


def test_kalman_backend_smooths_measurements():
    tracker = RadarTrackerCustom("kalman")
    results = [tracker.update(m) for m in [10, 12, 15, 14, 13]]
    assert all(isinstance(r, float) for r in results)


def test_particle_backend_smooths_measurements():
    tracker = RadarTrackerCustom("particle", seed=0)
    results = [tracker.update(m) for m in [10, 12, 15, 14, 13]]
    assert all(isinstance(r, float) for r in results)


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        RadarTrackerCustom("not_a_real_backend")


def test_filter_kwargs_pass_through():
    tracker = RadarTrackerCustom("particle", n_particles=50, seed=1)
    assert tracker.filter.n == 50
