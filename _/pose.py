import numpy as np

from models.mobile_net import MobileNetPoseEstimatorBackend
from models.adaboost import PoseClassifierBackend
import cv2

pose_estimator = MobileNetPoseEstimatorBackend(weights_path='weights/mobilenet-pose.pth')
pose_estimator.init_model()
pose_classifier = PoseClassifierBackend(
    capture_pkl='weights/pose_clf/adaboost-capture-openpose.pkl',
    land_pkl='weights/pose_clf/adaboost-land-openpose.pkl'
)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    keypoint, frame = pose_estimator.estimate(frame)
    # keypoint_center = (keypoint[0][0], keypoint[0][1]) if len(keypoint) > 0 else (0, 0)
    # cv2.circle(frame, keypoint_center, 5, (0, 255, 0), -1)

    keypoints = keypoint.flatten() if len(keypoint) else np.zeros(36, )
    keypoints = keypoints.reshape(1, -1)
    print(pose_classifier.classify(keypoints))
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
