import socket

import cv2

VIDEO_SOURCE = "./videos/street.mp4"

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 204800)

server_address = ('127.0.0.1', 11111)

cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])[1].tobytes()
    print(f'Sending {len(encoded_frame)} bytes')
    client_socket.sendto(encoded_frame, server_address)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
client_socket.close()
