import cv2
import numpy as np
import torch

from utils import non_max_suppression, scale_boxes

# torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference

BUFFER_SIZE = 65507

DEVICE = 'cuda:0'

DET_TORCHSCRIPT = 'yolov5s.torchscript'

print("Loading model")
model = torch.jit.load(DET_TORCHSCRIPT, map_location=torch.device(DEVICE))
model.float()  # FP32
model.to(DEVICE).eval().forward(torch.zeros(1, 3, 640, 640, device=DEVICE))  # warm-up

print("Waiting for UDP stream")
cap = cv2.VideoCapture('udp://0.0.0.0:11111', cv2.CAP_FFMPEG)

while not cap.isOpened():
    print("Error: Could not open video stream")
    cap.open("udp://0.0.0.0:11111")

print(cap.isOpened())
print('Waiting for video frame...')

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    # Preprocess image
    frame = cv2.resize(frame, (640, 640))

    img = np.transpose(frame, (2, 0, 1))[::-1]  # HWC -> CHW & BGR -> RGB
    img = np.ascontiguousarray(img)  # contiguous memory

    image_tensor = torch.from_numpy(img).to(DEVICE).float()  # FP32
    image_tensor /= 255.0  # normalize
    image_tensor = image_tensor[None]  # add batch dimension

    # Perform inference
    result = model(image_tensor)
    bbs = non_max_suppression(result)[0]  # NMS

    if len(bbs):
        bbs[:, :4] = scale_boxes(image_tensor.shape[2:], bbs[:, :4], (640, 640, 3)).round()
        # print(f'Found {len(bbs)} boxes')
        for i, (*xyxy, conf, cls) in enumerate(bbs):
            # x1, y1, x2, y2 = xyxy
            x1, y1, x2, y2 = map(int, xyxy)
            # print(f'Box {i}: ({x1}, {y1}, {x2}, {y2}) with confidence {conf.item()}')
            if cls == 0:  # person
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    cv2.namedWindow('Tello Video Stream', cv2.WINDOW_NORMAL)
    cv2.imshow('Tello Video Stream', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cv2.destroyAllWindows()
server_socket.close()
