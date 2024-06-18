from controller.pid import PIDController
from controller.tello import TelloController
from models.adaboost import PoseClassifierBackend
from models.mobile_net import MobileNetPoseEstimatorBackend
from models.yolo import YoloV8Backend

if __name__ == '__main__':
    # pose_estimator = MobileNetPoseEstimatorBackend(weights_path='weights/mobilenet-pose.pth', track=False)
    pose_estimator = YoloV8Backend(
        weights_path='weights/yolov8n-coco-pose.pt',
        device='cuda', tracker='weights/bytetrack.yaml'
    )
    # detector = YoloV8Backend(weights_path='weights/yolov8n-coco-pose.pt', device='cuda')
    # pose_estimator.pose_classifier = PoseClassifierBackend(
    #     capture_pkl='weights/pose_clf/adaboost-capture-openpose.pkl',
    #     land_pkl='weights/pose_clf/adaboost-land-openpose.pkl'
    # )
    # mobilenet_pid_controller = PIDController(
    #     (0.25, 0, 0), (-100, 100),
    #     (0.3, 0, 0.01), (-40, 40),
    #     (0.5, 0.04, 0.05), (-50, 50)
    # )
    yolo_pose_pid_controller = PIDController(
        (0.25, 0, 0), (-100, 100),
        (0.3, 0, 0), (-40, 40),
        (0.5, 0.04, 0.1), (-50, 50)
    )

    tello = TelloController(
        error_controller=yolo_pose_pid_controller,
        pose_estimator=pose_estimator,
        record_video=True,
        save_dir='output/tello',
    )

    tello.run()
