"""
RadarTrackerCustom -- picks a filter backend by name and exposes the
same .update(measurement) interface as the base RadarTracker.

    tracker = RadarTrackerCustom("kalman")
    tracker.update(10)

Deliberately thin: it does not reimplement RadarTracker's wrapping
logic, it just builds the right *_custom filter and delegates,
mirroring how core/radar_tracker.py wraps core/kalman_filter.py.

ADDED LATER: optional `regulator=JRegulator()` -- wires in the
"topologiczny regulator filtra" (core/j_regulator.py), implementing
the two pseudocode versions Jacek gave:

    Wersja minimalna (gate a measurement outright if it looks like a
    defect) -> applied to the kalman backend, since Kalman doesn't
    have a natural "soften this reading" knob the way the particle
    filter does. A gated measurement causes a predict()-only step
    (coast on the motion model) instead of update(z).

    Wersja dla filtra cząsteczkowego (adapt process_std / measurement_
    sigma / trigger roughening from twist/defect/resonance) -> applied
    to the particle backend as given.

With no regulator passed, behaviour is identical to before -- this is
purely additive.

BUG FOUND AND FIXED during real-data validation: the first version of
this wiring read `self.filter.vel_process_std` (etc.) as the "current"
value to boost from -- but that attribute had already been boosted on
the previous step, so every step the regulator triggered on multiplied
an already-multiplied number. On real GPS data (where the regulator's
default thresholds turned out to trigger on nearly every step, see
core/j_regulator.py) this compounded into nonsense: process_std growing
by ~1.5x *every single step* over 956 steps, i.e. exponentially, driving
the particle filter to garbage estimates in the range of 1e20+ meters.
Fixed by boosting from the fixed *base* parameters captured at __init__
time, not from the filter's current (possibly already-boosted) state --
so a step that doesn't trigger the regulator resets back to baseline
instead of staying inflated forever.
"""
from core.radar_tracker import RadarTracker
from core.kalman_filter_custom import KalmanFilterCustom
from core.particle_filter_custom import ParticleFilterCustom
from core.j_regulator import JRegulator

_BACKENDS = {
    "kalman": KalmanFilterCustom,
    "particle": ParticleFilterCustom,
}


class RadarTrackerCustom(RadarTracker):
    def __init__(self, backend: str, regulator: JRegulator | None = None, **filter_kwargs):
        if backend not in _BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}, expected one of {sorted(_BACKENDS)}"
            )
        super().__init__(_BACKENDS[backend](**filter_kwargs))
        self.backend = backend
        self.regulator = regulator

        # Sfotografuj bazowe parametry PRZED jakąkolwiek regulacją, żeby
        # regulator zawsze skalował od stałego punktu odniesienia, a nie
        # od wartości, którą sam już zmienił krok wcześniej (patrz błąd
        # kompozycji opisany w docstringu modułu).
        if backend == "particle":
            self._base_vel_std = self.filter.vel_process_std
            self._base_pos_std = self.filter.pos_process_std
            self._base_sigma = self.filter.measurement_sigma

    def update(self, measurement):
        if self.regulator is None:
            return self.filter.update(measurement)

        if self.backend == "particle":
            new_vel_std, new_pos_std, new_sigma, should_spread = (
                self.regulator.regulate_particle_params(
                    measurement,
                    self._base_vel_std,
                    self._base_pos_std,
                    self._base_sigma,
                )
            )
            self.filter.vel_process_std = new_vel_std
            self.filter.pos_process_std = new_pos_std
            self.filter.measurement_sigma = new_sigma
            if should_spread:
                self.filter.roughen()
            return self.filter.update(measurement)

        # kalman: wersja minimalna -- gate albo update
        gated = self.regulator.gate(measurement)
        if gated is None:
            return self.filter.predict()
        return self.filter.update(gated)
