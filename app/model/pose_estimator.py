class PoseEstimator:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, image):
        return self.model.predict(image)