from models.pose import MobileNetPoseEstimatorBackend
import cv2

pose_estimator = MobileNetPoseEstimatorBackend(weights_path='weights/mobilenet-pose.pth')
pose_estimator.init_model()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    poses, frame = pose_estimator.predict(frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
