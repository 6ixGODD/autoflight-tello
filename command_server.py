import socket

LOCAL_ADDRESS = ('0.0.0.0', 8889)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_socket.bind(LOCAL_ADDRESS)

while True:
    try:
        data, _ = server_socket.recvfrom(1518)
        print(data.decode('utf-8'))
    except KeyboardInterrupt:
        break

server_socket.close()
