from core.radar_tracker import RadarTracker
from core.kalman_filter import KalmanFilter

if __name__ == "__main__":
    tracker = RadarTracker(KalmanFilter())
    print(tracker.update(42))
