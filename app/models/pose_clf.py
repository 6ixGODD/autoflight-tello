import pickle


class PoseClassifier:
    def __init__(self):
        self.capture_clf = pickle.load(open("weights/capture_SVC.pkl", "rb"))
        self.land_clf = pickle.load(open("weights/land_SVC.pkl", "rb"))

    def predict(self, x: list) -> str:
        capture_pred = self.capture_clf.predict_proba([x])
        land_pred = self.land_clf.predict_proba([x])
        if capture_pred[0][1] > 0.5:
            return "capture"
        if land_pred[0][1] > 0.5:
            return "land"
        return "others"
