"""
Validates KalmanFilterCustom and ParticleFilterCustom against real GPS
driving trips (data/real_trips_sample.csv -- same source as the
RADAR-TRACKING repo: sobhan-moosavi/Trajectory_Segmentation on GitHub,
real recorded car trips, not synthetic).

Each trip's lat/lon is projected to local x/y meters (equirectangular,
centered on the trip's first point) and the x-coordinate is fed to each
filter as a noisy scalar measurement stream, one point per timestep.

Run: python3 data/validate_on_real_trips.py
"""
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.radar_tracker_custom import RadarTrackerCustom

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


def run():
    trips = load_trips()
    print(f"{'trip':6} {'backend':9} {'n':>5} {'mean':>8} {'max':>9} {'last20':>8}")
    for trip_id in sorted(trips):
        x = trips[trip_id]
        for backend in ("kalman", "particle"):
            kwargs = {"seed": 0} if backend == "particle" else {}
            tracker = RadarTrackerCustom(backend, **kwargs)
            diffs = [abs(tracker.update(z) - z) for z in x]
            print(f"{trip_id:6} {backend:9} n={len(diffs):4} "
                  f"mean={statistics.mean(diffs):7.2f}m max={max(diffs):8.2f}m "
                  f"last20={statistics.mean(diffs[-20:]):7.2f}m")


if __name__ == "__main__":
    run()
