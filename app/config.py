class Config:
    # Tello configurations
    TELLO_ADDRESS = '192.168.10.2'
    CONNECTION_TIMEOUT = 60.0
    FRAME_SKIP = 300

    # PID configurations
    ERROR_X_THRESHOLD = 60
    ERROR_Y_THRESHOLD = 90
    ERROR_SIZE_THRESHOLD = 0.3

    PID_CC_KP = 0.35
    PID_CC_KI = 0.2
    PID_CC_KD = 0.2

    PID_UD_KP = 0.3
    PID_UD_KI = 0.3
    PID_UD_KD = 0.3

    PID_FB_KP = 0.3
    PID_FB_KI = 0.1
    PID_FB_KD = 0.4

    PID_CC_OUTPUT_LIMITS = (-50, 50)
    PID_UD_OUTPUT_LIMITS = (-40, 40)
    PID_FB_OUTPUT_LIMITS = (-10, 10)

    # Model configurations
    IMG_SIZE = (960, 720)
    DEVICE = 'cuda:0'
    MODEL_WEIGHTS_DIR = './weights'
    YOLO_WEIGHT_PATH = MODEL_WEIGHTS_DIR + '/yolov8n-pose.pt'
    CONFIDENT_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.4
    IOU_THRESHOLD = 0.5
    TRACKER_CONFIG = './botsort.yaml'

    # Logger configurations
    LOGS_DIR = './logs'


