import pickle

import numpy as np

from models import BaseClassifierBackend


class PoseClassifierBackend(BaseClassifierBackend[int]):
    def __init__(self, capture_pkl: str, land_pkl: str):
        self.capture_clf = pickle.load(open(capture_pkl, 'rb'))
        self.land_clf = pickle.load(open(land_pkl, 'rb'))

    def classify(self, data: np.ndarray, **kwargs) -> int:  # 1 for capture, 2 for land and 0 for others
        if np.all(data == 0):
            return self.NEGATIVE
        if self.capture_clf.predict(data):
            return self.POSE_CAPTURE
        elif self.land_clf.predict(data):
            return self.POSE_LAND
        return self.NEGATIVE

    def init_model(self, **kwargs):
        pass
