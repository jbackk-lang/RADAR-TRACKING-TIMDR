"""
Validates JRegulator (core/j_regulator.py) against the same real GPS
trips as data/validate_on_real_trips.py: with vs. without the regulator,
for both backends, plus an ablation on the particle-filter regulator
(which parts of it -- process_std boost, measurement_sigma boost,
roughening -- actually help).

Run: python3 data/validate_regulator.py
"""
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.radar_tracker_custom import RadarTrackerCustom
from core.j_regulator import JRegulator

DATA_PATH = os.path.join(os.path.dirname(__file__), "real_trips_sample.csv")
R_EARTH = 6371000.0


def load_trips():
    trips = defaultdict(list)
    with open(DATA_PATH) as f:
        for row in csv.DictReader(f):
            trips[row["TripId"]].append(row)
    out = {}
    for trip_id, rows in trips.items():
        lat0, lon0 = float(rows[0]["Latitude"]), float(rows[0]["Longitude"])
        out[trip_id] = [
            math.radians(float(r["Longitude"]) - lon0) * R_EARTH * math.cos(math.radians(lat0))
            for r in rows
        ]
    return out


def delta_stats(trips):
    print("Rozkład realnych skokow pozycji (|x[i]-x[i-1]|) per trasa:")
    for trip_id in sorted(trips):
        x = trips[trip_id]
        deltas = [abs(x[i] - x[i - 1]) for i in range(1, len(x))]
        deltas_sorted = sorted(deltas)
        p90 = deltas_sorted[int(0.9 * len(deltas_sorted))]
        print(f"  {trip_id:6} mean={statistics.mean(deltas):6.2f}m "
              f"p90={p90:6.2f}m max={max(deltas):6.2f}m")
    print()


def kalman_gate_threshold_sweep(trips):
    print("Kalman + gate(): wplyw progu defect_threshold (ile pomiarow "
          "bramkuje i jak to wplywa na blad):")
    for th in (15, 25, 35, 50):
        rows = []
        for trip_id in sorted(trips):
            x = trips[trip_id]
            reg = JRegulator(defect_threshold=th)
            tracker = RadarTrackerCustom("kalman", regulator=reg)
            diffs = [abs(tracker.update(z) - z) for z in x]
            n_gated = sum(1 for i in range(1, len(x)) if abs(x[i] - x[i - 1]) > th)
            rows.append((trip_id, statistics.mean(diffs), max(diffs), n_gated, len(x)))
        print(f"  defect_threshold={th}")
        for trip_id, mean_e, max_e, n_gated, n in rows:
            print(f"    {trip_id:6} mean={mean_e:8.2f}m max={max_e:9.2f}m gated={n_gated}/{n}")
    print()


def particle_ablation(trips):
    print("Particle + regulate_particle_params(): ablacja mechanizmow:")
    configs = {
        "plain (bez regulatora)": None,
        "pelny regulator (twist8/defect15/reson10)":
            dict(twist_threshold=8, defect_threshold=15, resonance_threshold=10),
        "bez roughening (resonance_threshold=1e9)":
            dict(twist_threshold=8, defect_threshold=15, resonance_threshold=1e9),
        "bez sigma-boost (defect_threshold=1e9)":
            dict(twist_threshold=8, defect_threshold=1e9, resonance_threshold=10),
        "bez process-boost (twist_threshold=1e9)":
            dict(twist_threshold=1e9, defect_threshold=15, resonance_threshold=10),
    }
    for label, kwargs in configs.items():
        means, maxes = [], []
        for trip_id in sorted(trips):
            x = trips[trip_id]
            reg = JRegulator(**kwargs) if kwargs else None
            tracker = RadarTrackerCustom("particle", regulator=reg, seed=0)
            diffs = [abs(tracker.update(z) - z) for z in x]
            means.append(statistics.mean(diffs))
            maxes.append(max(diffs))
        print(f"  {label:45} avg_mean={statistics.mean(means):6.2f}m "
              f"worst_max={max(maxes):8.2f}m")
    print()


if __name__ == "__main__":
    trips = load_trips()
    delta_stats(trips)
    kalman_gate_threshold_sweep(trips)
    particle_ablation(trips)
