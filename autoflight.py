import socket
import time

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

from utils import non_max_suppression, scale_boxes

torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference

TELLO_IP = '192.168.10.1'

TELLO_PORT = 8889

BUFFER_SIZE = 65507

DEVICE = 'cuda:0'

DET_TORCHSCRIPT = 'yolov5s.torchscript'

CLASS_NUM = 80


client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.sendto('command'.encode('utf-8'), (TELLO_IP, TELLO_PORT))  # send `command` to Tello to open SDK mode
time.sleep(1)
client_socket.sendto('streamon'.encode('utf-8'), (TELLO_IP, TELLO_PORT))
time.sleep(1)
client_socket.sendto('takeoff'.encode('utf-8'), (TELLO_IP, TELLO_PORT))
time.sleep(1)
client_socket.sendto('up 80'.encode('utf-8'), (TELLO_IP, TELLO_PORT))

det_model = torch.jit.load(DET_TORCHSCRIPT, map_location=torch.device(DEVICE))
det_model.float()  # FP32
det_model.to(DEVICE).eval().forward(torch.zeros(1, 3, 640, 640, device=DEVICE))  # warm-up

tracker = DeepSort(max_age=5)

cap = cv2.VideoCapture('udp://0.0.0.0:11111', cv2.CAP_FFMPEG)

while not cap.isOpened():
    print("Error: Could not open video stream")
    cap.open("udp://0.0.0.0:11111")

print(cap.isOpened())

print('Waiting for video frame...')
while True:
    # Receive video frame
    ret, frame = cap.read()

    time.sleep(5)

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
        min_t_id = min([t.track_id for t in track if t.is_confirmed()])
        for t in track:
            if not t.is_confirmed():
                continue
            t_id = t.track_id
            bbox = t.to_tlbr().astype(int)
            if t_id == min_t_id:
                if not (bbox[0] < 320 < bbox[2]):
                    if bbox[2] < 320:
                        client_socket.sendto('cw 100'.encode('utf-8'), (TELLO_IP, TELLO_PORT))
                    else:
                        client_socket.sendto('ccw 100'.encode('utf-8'), (TELLO_IP, TELLO_PORT))
                if not (bbox[1] < 320 < bbox[3]):
                    if bbox[3] < 320:
                        client_socket.sendto('back 30'.encode('utf-8'), (TELLO_IP, TELLO_PORT))
                    else:
                        client_socket.sendto('forward 30'.encode('utf-8'), (TELLO_IP, TELLO_PORT))
                if (bbox[3] - bbox[1]) / 640 > 0.3 or (bbox[2] - bbox[0]) / 640 > 0.6:
                    client_socket.sendto('back 30'.encode('utf-8'), (TELLO_IP, TELLO_PORT))
                elif (bbox[3] - bbox[1]) / 640 < 0.1 or (bbox[2] - bbox[0]) / 640 < 0.2:
                    client_socket.sendto('forward 30'.encode('utf-8'), (TELLO_IP, TELLO_PORT))
                cv2.rectangle(frame, (bbox[0], bbox[3]), (bbox[2], bbox[1]), (255, 0, 0), 3)
                break

    cv2.namedWindow('Tello Video Stream', cv2.WINDOW_NORMAL)
    cv2.imshow('Tello Video Stream', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cv2.destroyAllWindows()
server_socket.close()
