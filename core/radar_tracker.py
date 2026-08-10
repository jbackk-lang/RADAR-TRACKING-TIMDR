class RadarTracker:
    def __init__(self, filter):
        self.filter = filter

    def update(self, measurement):
        return self.filter.update(measurement)
