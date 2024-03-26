import socket
import time

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

from utils import non_max_suppression, scale_boxes

torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference

BUFFER_SIZE = 65507

DEVICE = 'cuda:0'

LOCAL_ADDRESS = ('0.0.0.0', 11111)

DET_TORCHSCRIPT = 'yolov5l.torchscript'

CLASS_NUM = 80

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(LOCAL_ADDRESS)

det_model = torch.jit.load(DET_TORCHSCRIPT, map_location=torch.device(DEVICE))
det_model.float()  # FP32
det_model.to(DEVICE).eval().forward(torch.zeros(1, 3, 640, 640, device=DEVICE))  # warm-up

tracker = DeepSort(max_age=5)

print('Waiting for video frame...')
while True:
    # Receive video frame
    data, _ = server_socket.recvfrom(BUFFER_SIZE)

    if not data:
        continue

    image = np.frombuffer(data, dtype=np.uint8)

    # Preprocess image
    frame = cv2.imdecode(image, 1)  # 1 means load color image
    frame = cv2.resize(frame, (640, 640))

    img = np.transpose(frame, (2, 0, 1))[::-1]  # HWC -> CHW & BGR -> RGB
    img = np.ascontiguousarray(img)  # contiguous memory

    image_tensor = torch.from_numpy(img).to(DEVICE).float()  # FP32
    image_tensor /= 255.0  # normalize
    image_tensor = image_tensor[None]  # add batch dimension

    # Perform inference
    result = det_model(image_tensor)
    bbs = non_max_suppression(result)[0]  # NMS

    if len(bbs):
        bbs[:, :4] = scale_boxes(image_tensor.shape[2:], bbs[:, :4], (640, 640, 3)).round()
        bx = bbs.cpu().numpy()
        # bx = [([sub[0], sub[1], sub[2] - sub[0], sub[3] - sub[1]], sub[4], sub[5] == 0) for sub in bx]
        bx = [([sub[0], sub[1], sub[2] - sub[0], sub[3] - sub[1]], sub[4], sub[5]) for sub in bx]
        print(f'Found {len(bbs)} persons')
        start = time.time_ns()
        track = tracker.update_tracks(bx, frame=frame)
        print(f'Tracking time: {(time.time_ns() - start) / 1e6:.2f} ms')
        for t in track:
            if not t.is_confirmed():
                continue
            t_id = t.track_id
            bbox = t.to_tlbr().astype(int)
            # bbox[2:] -= bbox[:2]
            bbox[1], bbox[3] = bbox[3], bbox[1]
            color = (255, 0, 0)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            cv2.putText(
                frame,
                str(t_id),
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

    cv2.namedWindow('Tello Video Stream', cv2.WINDOW_NORMAL)
    cv2.imshow('Tello Video Stream', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cv2.destroyAllWindows()
server_socket.close()
