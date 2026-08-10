import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import math
import statistics
from collections import defaultdict

from core.particle_filter_custom import ParticleFilterCustom

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "real_trips_sample.csv")


def test_no_divergence_on_fast_synthetic_target():
    """Regression test for the original bug: process_std=1.0 with an
    implicit measurement sigma of 1.0 caused permanent loss of lock on
    a target moving at v=3/step -- the estimate would freeze while the
    true position kept climbing."""
    import numpy as np
    pf = ParticleFilterCustom(seed=0)
    rng = np.random.default_rng(0)
    true_pos, true_vel = 0.0, 3.0
    lags = []
    for _ in range(30):
        true_pos += true_vel
        z = true_pos + rng.normal(0, np.sqrt(5.0))
        est = pf.update(z)
        lags.append(abs(true_pos - est))
    assert max(lags) < 20.0          # never diverges
    assert statistics.mean(lags[-15:]) < 5.0   # settles down, doesn't drift


def _load_real_trip(trip_id="T-29"):
    trips = defaultdict(list)
    with open(DATA_PATH) as f:
        for row in csv.DictReader(f):
            trips[row["TripId"]].append(row)
    rows = trips[trip_id]
    lat0, lon0 = float(rows[0]["Latitude"]), float(rows[0]["Longitude"])
    R = 6371000.0
    return [math.radians(float(r["Longitude"]) - lon0) * R * math.cos(math.radians(lat0)) for r in rows]


def test_no_divergence_on_real_highway_speed_data():
    """The first fix (raise process_std to 4.0, tuned against a v=3/step
    synthetic scenario) looked correct but diverged catastrophically on
    real driving data reaching 33 m/s (mean error ~2037m, peaks over
    5500m). This test locks in the structural fix (position+velocity
    particle state) against the same real 956-step GPS trip."""
    real_x = _load_real_trip("T-29")
    pf = ParticleFilterCustom(seed=0)
    diffs = [abs(pf.update(z) - z) for z in real_x]

    assert len(diffs) > 900
    assert statistics.mean(diffs) < 20.0        # was ~2037m before the structural fix
    assert statistics.mean(diffs[-20:]) < 10.0  # ends stable, not mid-divergence
