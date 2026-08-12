import sys, os, csv, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.curvature_detector import CurvatureDetector, CurvatureResult


def test_first_two_updates_are_gated_no_history():
    det = CurvatureDetector()
    r1 = det.update(0.0, 0.0)
    assert r1.gated is True
    assert r1.curvature == 0.0
    r2 = det.update(10.0, 0.0)
    assert r2.gated is True
    assert r2.curvature == 0.0


def test_straight_line_gives_zero_curvature_axis_aligned():
    det = CurvatureDetector(min_step_m=1.0)
    pts = [(10.0 * i, 0.0) for i in range(6)]
    results = [det.update(x, y) for x, y in pts]
    for r in results[2:]:
        assert abs(r.curvature) < 1e-9
        assert r.gated is False
        assert abs(r.direction_stability - 1.0) < 1e-9  # perfectly aligned direction


def test_straight_line_gives_zero_curvature_diagonal():
    """Same straight-line motion but at an angle -- must still be exactly zero.
    (This is the case where the rejected THE-GEO-PRO-2D-to-3D 'tau' extension
    broke, reporting a false nonzero 'torsion' for diagonal straight motion.)"""
    det = CurvatureDetector(min_step_m=1.0)
    pts = [(3.0 * i, 4.0 * i) for i in range(6)]
    results = [det.update(x, y) for x, y in pts]
    for r in results[2:]:
        assert abs(r.curvature) < 1e-9


def test_rotation_invariance():
    """Curvature of a physical path must not depend on how the x/y axes
    happen to be oriented. Rotate the same turning path by an arbitrary
    angle and confirm the curvature sequence is unchanged (within float
    tolerance)."""
    path = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (20.0, 20.0)]

    def curvatures_for(points):
        det = CurvatureDetector(min_step_m=1.0)
        return [det.update(x, y).curvature for x, y in points]

    base = curvatures_for(path)

    theta = 37.0 * math.pi / 180.0
    rotated = [
        (x * math.cos(theta) - y * math.sin(theta), x * math.sin(theta) + y * math.cos(theta))
        for x, y in path
    ]
    rotated_curv = curvatures_for(rotated)

    for a, b in zip(base, rotated_curv):
        assert abs(a - b) < 1e-9


def test_sharp_turn_produces_clear_spike():
    det = CurvatureDetector(min_step_m=1.0)
    path = [(0, 0), (10, 0), (20, 0), (20, 10), (20, 20)]
    results = [det.update(x, y) for x, y in path]
    # the 90-degree turn happens going into point index 3 -- curvature should
    # spike there and be zero on the straight segments before/after
    assert results[2].curvature == 0.0   # still straight (0,0)->(10,0)->(20,0)
    assert results[3].curvature > 0.05   # the turn
    assert results[4].curvature == 0.0   # straight again (20,0)->(20,10)->(20,20)


def test_stationary_gps_noise_is_gated_not_amplified():
    """This is the bug found in the raw THE pseudocode: dividing by a tiny
    step length (GPS jitter while the target is parked) blows curvature up
    to huge false-positive values. With the speed gate it must be exactly 0
    and explicitly marked as gated (not 'confirmed straight')."""
    import random
    random.seed(0)
    det = CurvatureDetector(min_step_m=3.0)
    results = []
    for _ in range(30):
        x = 100.0 + random.uniform(-0.3, 0.3)
        y = 100.0 + random.uniform(-0.3, 0.3)
        results.append(det.update(x, y))
    for r in results[2:]:
        assert r.curvature == 0.0
        assert r.gated is True


def test_ungated_detector_is_vulnerable_to_gps_noise_regression_guard():
    """Regression guard proving *why* the gate exists: with min_step_m=0,
    the same noisy-stationary sequence used above produces a large spurious
    curvature spike. If this test ever starts failing (spike disappears),
    something about the underlying math changed and the gate's necessity
    should be re-checked."""
    import random
    random.seed(0)
    det = CurvatureDetector(min_step_m=0.0)
    results = []
    for _ in range(30):
        x = 100.0 + random.uniform(-0.3, 0.3)
        y = 100.0 + random.uniform(-0.3, 0.3)
        results.append(det.update(x, y))
    max_curv = max(r.curvature for r in results[2:])
    assert max_curv > 1.0  # noise-amplified spike, far above any real turn signal


def test_min_step_m_rejects_negative():
    try:
        CurvatureDetector(min_step_m=-1.0)
        assert False, "should have raised"
    except ValueError:
        pass


def test_real_trip_correlation_regression_guard():
    """Regression guard for the calibrated threshold: on real GPS trip data
    with labeled ground-truth heading change, curvature (gated at 3m) must
    correlate positively and meaningfully with |Heading_Change|. This is the
    core empirical claim behind recommending this detector -- if it silently
    degrades below sanity, tests should fail."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "real_trips_sample.csv")
    if not os.path.exists(data_path):
        return  # data file not bundled in this environment -- skip quietly

    trips = {}
    with open(data_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            trips.setdefault(row["TripId"], []).append(row)

    def latlon_to_xy(lat, lon, lat0):
        R = 6371000.0
        x = math.radians(lon) * R * math.cos(math.radians(lat0))
        y = math.radians(lat) * R
        return x, y

    correlations = []
    for trip_id, rows in trips.items():
        lat0 = float(rows[0]["Latitude"])
        det = CurvatureDetector(min_step_m=3.0)
        curvs, heading_changes = [], []
        for row in rows:
            x, y = latlon_to_xy(float(row["Latitude"]), float(row["Longitude"]), lat0)
            res = det.update(x, y)
            curvs.append(res.curvature)
            heading_changes.append(abs(float(row["Heading_Change(degrees)"])))

        n = len(curvs)
        mean_c = sum(curvs) / n
        mean_h = sum(heading_changes) / n
        cov = sum((curvs[i] - mean_c) * (heading_changes[i] - mean_h) for i in range(n))
        var_c = sum((c - mean_c) ** 2 for c in curvs)
        var_h = sum((h - mean_h) ** 2 for h in heading_changes)
        corr = cov / math.sqrt(var_c * var_h) if var_c > 0 and var_h > 0 else 0.0
        correlations.append(corr)

    assert all(c > 0.2 for c in correlations), f"korelacje spadły poniżej sanity: {correlations}"
    assert sum(correlations) / len(correlations) > 0.4
