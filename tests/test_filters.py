def test_filters():
    from core.kalman_filter import KalmanFilter
    from core.particle_filter import ParticleFilter
    assert KalmanFilter().update(5) == 5
    assert ParticleFilter().update(7) == 7
