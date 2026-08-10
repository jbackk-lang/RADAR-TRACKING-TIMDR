import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.j_regulator import JRegulator
from core.radar_tracker_custom import RadarTrackerCustom


def test_first_step_returns_zeros():
    """No previous echo yet -- nothing to compute a delta against."""
    r = JRegulator()
    j = r.step(10.0)
    assert j == {"twist": 0.0, "defect": 0.0, "resonance": 0.0}


def test_twist_is_signed_delta():
    r = JRegulator()
    r.step(10.0)
    j = r.step(16.0)
    assert j["twist"] == 6.0
    assert j["defect"] == 6.0


def test_twist_sign_flips_with_direction():
    r = JRegulator()
    r.step(10.0)
    j = r.step(4.0)
    assert j["twist"] == -6.0
    assert j["defect"] == 6.0  # defect is magnitude, always positive


def test_resonance_is_smoothed_not_instantaneous():
    """A single large jump shouldn't push resonance as high as defect
    itself -- it's an EMA, so it should lag behind a lone spike."""
    r = JRegulator(resonance_smoothing=0.15)
    r.step(0.0)
    j = r.step(100.0)
    assert j["defect"] == 100.0
    assert j["resonance"] < j["defect"]
    assert abs(j["resonance"] - 15.0) < 1e-9  # alpha * defect + 0


def test_resonance_climbs_under_sustained_defects():
    r = JRegulator(resonance_smoothing=0.15)
    pos = 0.0
    last = None
    for _ in range(30):
        pos += 50.0  # steady large jump every step
        last = r.step(pos)
    # after many sustained large steps, resonance (EMA of defect) should
    # have climbed close to the steady-state defect value of 50
    assert last["resonance"] > 40.0


def test_gate_rejects_large_defect():
    r = JRegulator(defect_threshold=5.0)
    r.step(0.0)
    assert r.gate(20.0) is None  # defect=20 > threshold=5


def test_gate_accepts_small_defect():
    r = JRegulator(defect_threshold=50.0)
    r.step(0.0)
    assert r.gate(5.0) == 5.0


def test_regulate_particle_params_boosts_on_big_twist():
    r = JRegulator(twist_threshold=2.0, process_std_boost=2.0)
    r.step(0.0)
    new_vel, new_pos, new_sigma, spread = r.regulate_particle_params(
        10.0, vel_process_std=1.0, pos_process_std=0.5, measurement_sigma=5.0
    )
    assert new_vel == 2.0
    assert new_pos == 1.0


def test_regulate_particle_params_boosts_sigma_on_defect():
    r = JRegulator(defect_threshold=2.0, measurement_sigma_boost=3.0)
    r.step(0.0)
    _, _, new_sigma, _ = r.regulate_particle_params(
        10.0, vel_process_std=1.0, pos_process_std=0.5, measurement_sigma=5.0
    )
    assert new_sigma == 15.0


def test_regulate_particle_params_no_change_below_thresholds():
    r = JRegulator(twist_threshold=1000, defect_threshold=1000, resonance_threshold=1000)
    r.step(0.0)
    new_vel, new_pos, new_sigma, spread = r.regulate_particle_params(
        1.0, vel_process_std=1.0, pos_process_std=0.5, measurement_sigma=5.0
    )
    assert (new_vel, new_pos, new_sigma, spread) == (1.0, 0.5, 5.0, False)


def test_tracker_with_regulator_kalman_gates_outlier_by_coasting():
    """A wild single outlier, with a tight defect threshold, should be
    gated -- the tracker should not jump toward it."""
    tracker = RadarTrackerCustom(
        "kalman", regulator=JRegulator(defect_threshold=3.0)
    )
    for m in [10, 10, 10, 10]:
        est = tracker.update(m)
    est_before_outlier = est
    est_after_outlier = tracker.update(500.0)  # huge, implausible jump
    # gated: estimate should stay close to where it was, not leap to ~500
    assert abs(est_after_outlier - est_before_outlier) < 20.0


def test_tracker_with_regulator_particle_runs_without_crashing():
    tracker = RadarTrackerCustom(
        "particle", regulator=JRegulator(), seed=0
    )
    results = [tracker.update(m) for m in [10, 12, 40, 41, 13]]
    assert all(isinstance(r, float) for r in results)


def test_tracker_without_regulator_unaffected():
    """No regulator passed -- behaviour must be identical to the plain
    backend (regression guard so wiring the regulator in doesn't change
    default behaviour)."""
    plain = RadarTrackerCustom("kalman")
    regulated_but_none = RadarTrackerCustom("kalman", regulator=None)
    seq = [10, 12, 15, 14, 13]
    r1 = [plain.update(m) for m in seq]
    r2 = [regulated_but_none.update(m) for m in seq]
    assert r1 == r2
