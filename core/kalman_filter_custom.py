"""
1D constant-velocity Kalman filter, single scalar measurement per step.

Matches the design Jacek pasted for review, with one real bug fixed:
`float(self.x[0])` operates on a (1,) array slice, not a scalar. NumPy
already emits DeprecationWarning: "Conversion of an array with ndim > 0
to a scalar is deprecated ... will error in future" for this -- tested
directly by turning warnings into errors, and it raises today.
Fixed with `self.x[0, 0]`, which indexes an actual scalar element.

Everything else about the math was verified numerically (constant-
velocity target, Gaussian measurement noise): position estimate tracks
the true trajectory and the velocity estimate converges to the true
velocity within a handful of steps.

Known simplification, disclosed rather than silently assumed: `F` and
`Q` are built for a fixed dt=1 per update() call. If your measurements
arrive at irregular intervals, rebuild F/Q with the real dt each step
(F = [[1, dt], [0, 1]]) instead of using this class as-is.
"""
import numpy as np


class KalmanFilterCustom:
    def __init__(self, process_var: float = 0.01, measurement_var: float = 5.0,
                 initial_position: float = 0.0, initial_uncertainty: float = 100.0):
        self.x = np.array([[initial_position], [0.0]])  # pozycja + predkosc
        self.P = np.eye(2) * initial_uncertainty
        self.F = np.array([[1, 1],
                            [0, 1]])
        self.H = np.array([[1, 0]])
        self.R = np.array([[measurement_var]])
        self.Q = np.eye(2) * process_var

    def update(self, z):
        # predykcja
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # korekcja
        y = z - (self.H @ self.x)[0]
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K * y
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return float(self.x[0, 0])

    @property
    def velocity_estimate(self) -> float:
        return float(self.x[1, 0])
