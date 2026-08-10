"""
RadarTrackerCustom -- picks a filter backend by name and exposes the
same .update(measurement) interface as the base RadarTracker.

    tracker = RadarTrackerCustom("kalman")
    tracker.update(10)

Deliberately thin: it does not reimplement RadarTracker's wrapping
logic, it just builds the right *_custom filter and delegates,
mirroring how core/radar_tracker.py wraps core/kalman_filter.py.
"""
from core.radar_tracker import RadarTracker
from core.kalman_filter_custom import KalmanFilterCustom
from core.particle_filter_custom import ParticleFilterCustom

_BACKENDS = {
    "kalman": KalmanFilterCustom,
    "particle": ParticleFilterCustom,
}


class RadarTrackerCustom(RadarTracker):
    def __init__(self, backend: str, **filter_kwargs):
        if backend not in _BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}, expected one of {sorted(_BACKENDS)}"
            )
        super().__init__(_BACKENDS[backend](**filter_kwargs))
        self.backend = backend
