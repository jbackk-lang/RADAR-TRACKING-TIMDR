import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
from core.kalman_filter_custom import KalmanFilterCustom


def test_no_deprecation_warning_on_update():
    """Regression test for the original bug: float(self.x[0]) on a
    (1,) array slice triggers numpy's DeprecationWarning today and
    will hard-error in a future numpy version. Fixed with self.x[0,0]."""
    kf = KalmanFilterCustom()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kf.update(10.0)  # must not raise


def test_converges_toward_constant_velocity_target():
    import numpy as np
    kf = KalmanFilterCustom()
    rng = np.random.default_rng(0)
    true_pos, true_vel = 0.0, 3.0
    last_est = None
    for _ in range(30):
        true_pos += true_vel
        z = true_pos + rng.normal(0, np.sqrt(5.0))
        last_est = kf.update(z)
    assert abs(last_est - true_pos) < 5.0
    assert abs(kf.velocity_estimate - true_vel) < 1.0


def test_returns_plain_float():
    kf = KalmanFilterCustom()
    result = kf.update(5.0)
    assert isinstance(result, float)
