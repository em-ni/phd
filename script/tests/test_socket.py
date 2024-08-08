# to diable firewall: sudo ufw disable
# to get ip address: ip addr

import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 65432))
server_socket.listen()

print("Server listening on port 65432...")

while True:
    conn, addr = server_socket.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            print(f"Received data: {data.decode('utf-8')}")
