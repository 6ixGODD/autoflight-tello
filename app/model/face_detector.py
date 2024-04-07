class Detector:
    def __init__(self, model):
        self.model = model

    def detect(self, image):
        # do some detection
        return self.model.predict(image)
