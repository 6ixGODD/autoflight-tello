import socket

TELLO_IP = '192.168.10.1'
# TELLO_IP = '127.0.0.1'

TELLO_PORT = 8889
# TELLO_PORT = 8890

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

client_socket.sendto('command'.encode('utf-8'), (TELLO_IP, TELLO_PORT))

while True:
    try:
        message = input('Enter command: ')
        if not message:
            break
        client_socket.sendto(message.encode('utf-8'), (TELLO_IP, TELLO_PORT))

    except KeyboardInterrupt:
        break

client_socket.close()
