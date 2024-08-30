import socket
import re


def start_server(host="127.0.0.1", port=5000):
    # Create a socket object using the AF_INET address family and SOCK_STREAM socket type
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the address and port number
    server_socket.bind((host, port))

    # Start listening for incoming connections with a backlog of 5
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")

    try:
        while True:
            # Accept a connection
            client_socket, addr = server_socket.accept()
            print(f"Connected by {addr}")

            try:
                while True:
                    # Receive data from the client
                    data = client_socket.recv(1024)
                    if not data:
                        break  # Break the loop if data is not received
                    data_decoded = data.decode()
                    # print(f"Received: {data_decoded}")

                    # Extract pressure values using regex
                    matches = re.finditer(
                        r"Channel (\d) voltage=[\d.]+V pressure=([-\d.]+)psi",
                        data_decoded,
                    )
                    for match in matches:
                        channel, pressure = int(match.group(1)), float(match.group(2))
                        # setattr(self, f"p{channel}", pressure)  # D
                        if channel == 1:
                            print(f"Channel {channel} pressure: {pressure} psi")

            finally:
                client_socket.close()  # Ensure the client socket is closed after processing
                print("Connection closed.")

    except KeyboardInterrupt:
        print("Server is shutting down.")

    finally:
        server_socket.close()  # Ensure the server socket is closed when done
        print("Server closed.")


if __name__ == "__main__":
    start_server()
