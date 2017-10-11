import numpy as np
from collections import deque
class VehicleTracker():
    def __init__(self):
        self.threshold = 4
        self.heatmaps = deque([],maxlen=8)
