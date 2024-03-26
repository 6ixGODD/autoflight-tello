import cv2

cap = cv2.VideoCapture('udp://0.0.0.0:11111', cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Could not open video stream")
    cap.open("udp://0.0.0.0:11111")

print(cap.isOpened())

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    cv2.namedWindow('Tello Video Stream', cv2.WINDOW_NORMAL)
    cv2.imshow('Tello Video Stream', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
server_socket.close()
