from controller.pid import PIDController
from controller.tello import TelloController
from models.mobile_net import MobileNetPoseEstimatorBackend
from models.adaboost import PoseClassifierBackend

if __name__ == '__main__':
    pose_estimator = MobileNetPoseEstimatorBackend(weights_path='weights/mobilenet-pose.pth')
    pose_classifier = PoseClassifierBackend(
        capture_pkl='weights/pose_clf/adaboost-capture.pkl',
        land_pkl='weights/pose_clf/adaboost-land.pkl'
    )
    pid_controller = PIDController(
        (0.25, 0, 0), (-100, 100), (0.3, 0, 0.01), (-40, 40), (0.5, 0.04, 0.05), (-50, 50)
    )

    tello = TelloController(
        error_controller=pid_controller,
        pose_classifier=pose_classifier,
        pose_estimator=pose_estimator,
        save_dir='output/tello'
    )
    tello.run()
