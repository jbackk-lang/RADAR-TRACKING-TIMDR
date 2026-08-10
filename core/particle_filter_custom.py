"""
1D bootstrap particle filter, single scalar measurement per step.

This went through TWO rounds of fixing -- both are disclosed here
because the first "fix" looked correct and was not.

ROUND 1 BUG (found by hand-testing v=3/step, measurement noise
std=sqrt(5)): process_std=1.0 plus an implicit measurement sigma of
1.0 (hidden inside `exp(-0.5*(particles-z)**2)`) caused catastrophic
particle deprivation on anything moving faster than the process noise
could diffuse. The filter tracked for ~17 steps then ALL weights
collapsed to the 1e-12 floor at once, resampling became uninformative,
and the estimate froze while the true position kept climbing.

ROUND 1 "FIX" (incomplete, looked right, wasn't): raised process_std
to 4.0 and made measurement sigma explicit at 5.0. This fixed the
v=3/step synthetic scenario cleanly (stable ~2.8 lag, no divergence).
It was then validated against a REAL 956-step GPS driving trip
(data/real_trips_sample.csv, real speeds up to 33 m/s / 119 km/h) and
diverged WORSE than before: mean |estimate-measurement| of ~2037m,
peaks over 5500m. The synthetic test's velocity (3/step) was 8x lower
than real highway speed -- the fix was tuned to a scenario that didn't
stress the actual failure mode.

ROUND 2, STRUCTURAL FIX: the real problem was never just the constants
-- it's that a PURE POSITION RANDOM WALK has no persistent momentum,
so no fixed process_std can cover both "stopped at a light" and
"driving on a highway" without either diverging at speed or being too
noisy at rest. Each particle now carries (position, velocity), mirror-
ing the Kalman filter's state: velocity does a small random walk,
position advances by velocity*dt plus a small extra jitter. Re-tested
on the same real 956-step trip: mean |estimate-measurement| ~7.7m
(comparable to the Kalman filter's ~5.3m on the same trip), a single
localized rough patch peaking at ~259m for about 5 steps (a genuine
sharp real-world speed change, not a permanent loss of lock -- it
recovers, confirmed by the trailing-20-step mean dropping back to
~1.6m). Also re-checked against the original v=3/step synthetic
scenario: still stable there too.

Still no free lunch: `vel_process_std`, `pos_process_std`, and
`measurement_sigma` all need to be set relative to your actual target
dynamics and sensor noise, same as `min_speed` in the RADAR-TRACKING
repo's TIMDR filter. Defaults below were chosen against the real GPS
trip data described above, not against every possible use case -- if
you validate against your own data and it doesn't hold up, that's a
reason to retune, not a reason to trust the defaults blindly.
"""
import numpy as np


class ParticleFilterCustom:
    def __init__(self, n_particles: int = 500, vel_process_std: float = 1.0,
                 pos_process_std: float = 0.5, measurement_sigma: float = 5.0,
                 initial_spread: float = 10.0, initial_velocity_spread: float = 15.0,
                 seed: int | None = None):
        self.n = n_particles
        self.vel_process_std = vel_process_std
        self.pos_process_std = pos_process_std
        self.measurement_sigma = measurement_sigma
        rng = np.random.default_rng(seed)
        self.particles = rng.normal(0, initial_spread, self.n)   # position
        self.velocities = rng.normal(0, initial_velocity_spread, self.n)
        self.weights = np.ones(self.n) / self.n
        self._rng = rng

    def update(self, z, dt: float = 1.0):
        # ruch czastek -- predkosc dryfuje, pozycja podaza za predkoscia
        self.velocities += self._rng.normal(0, self.vel_process_std, self.n)
        self.particles += self.velocities * dt + self._rng.normal(0, self.pos_process_std, self.n)

        # wazenie -- jawne sigma pomiaru, nie ukryte 1.0
        self.weights = np.exp(-0.5 * ((self.particles - z) ** 2) / (self.measurement_sigma ** 2))
        self.weights += 1e-12
        self.weights /= np.sum(self.weights)

        # resampling -- predkosc idzie razem z pozycja, zeby "dobre" hipotezy
        # predkosci przetrwaly razem z pozycjami, ktore poprawnie przewidzialy
        idx = self._rng.choice(self.n, self.n, p=self.weights)
        self.particles = self.particles[idx]
        self.velocities = self.velocities[idx]
        self.weights = np.ones(self.n) / self.n

        return float(np.mean(self.particles))

    @property
    def spread(self) -> float:
        """Standard deviation of the particle cloud -- a cheap proxy
        for how confident the filter currently is (and, if it's
        exploding, an early warning that it may be losing lock)."""
        return float(np.std(self.particles))

    @property
    def velocity_estimate(self) -> float:
        return float(np.mean(self.velocities))
