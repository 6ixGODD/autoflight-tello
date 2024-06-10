import pickle

import numpy as np

from models import BaseClassifierBackend

class PoseClassifierBackend(BaseClassifierBackend[int]):
    def __init__(self, capture_pkl: str, land_pkl: str):
        self.capture_clf = pickle.load(open(capture_pkl, 'rb'))
        self.land_clf = pickle.load(open(land_pkl, 'rb'))

    def predict(self, data: np.ndarray, **kwargs) -> int:  # 1 for capture, 2 for land and 0 for others
        return 2 if self.capture_clf.predict(data) == 1 else 1 if self.land_clf.predict(data) == 1 else 0

    def init_model(self, **kwargs):
        pass
