def test_tracker():
    from core.radar_tracker import RadarTracker
    from core.kalman_filter import KalmanFilter
    t = RadarTracker(KalmanFilter())
    assert t.update(10) == 10
