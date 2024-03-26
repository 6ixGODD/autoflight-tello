import socket
import threading

TELLO_IP = '192.168.10.1'

TELLO_PORT = 8889

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

client_socket.sendto('command'.encode('utf-8'), (TELLO_IP, TELLO_PORT))


def receive_thread():
    while True:
        try:
            data, _ = client_socket.recvfrom(1518)
            print("Received:", data.decode('utf-8'))
        except KeyboardInterrupt:
            break


threading.Thread(target=receive_thread, daemon=True).start()

while True:
    try:
        message = input('Enter command: ')
        if not message:
            break
        client_socket.sendto(message.encode('utf-8'), (TELLO_IP, TELLO_PORT))

    except KeyboardInterrupt:
        break

client_socket.close()
