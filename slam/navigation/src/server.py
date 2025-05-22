import socket
import time
import numpy as np


def sim_server(app, host="127.0.0.1", port=12345):
    time.sleep(1)  # Give the server time to start
    print("Starting sim_server...")
    s = None  # Initialize s to None
    try:
        # Create a socket and connect to the server
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        print(f"Sim_server connected to {host}:{port}")

        while True:
            # Placeholder for sending simulation data if needed in the future
            # For now, it just keeps the connection alive or waits for a signal
            # to send something.
            # Example: if app.should_send_data:
            # data_to_send = "some_sim_data"
            # s.sendall(data_to_send.encode())
            time.sleep(0.1)  # Add small delay between potential sends or checks

    except ConnectionRefusedError:
        print(
            f"Sim_server could not connect to server at {host}:{port}. Is it running?"
        )
    except Exception as e:
        print(f"Error in sim_server: {e}")
    finally:
        if s:
            s.close()
            print("Sim_server connection closed.")


def start_server(app, host="127.0.0.1", port=12345):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            s.listen()
            print(f"Server listening on {host}:{port}")

            while True:
                conn, addr = s.accept()
                with conn:
                    # print(f"Connected by {addr}") # Original code commented this out
                    while True:
                        try:
                            data = conn.recv(1024)
                            if not data:
                                break
                            # Parse the received data and update the transformation matrix
                            w_T_c_string = data.decode()
                            lines = w_T_c_string.splitlines()

                            # Basic validation for matrix format
                            if not lines or not all(
                                len(line.split()) > 0 for line in lines
                            ):
                                print(f"Received malformed data: {w_T_c_string}")
                                continue

                            matrix = [list(map(float, line.split())) for line in lines]

                            # Ensure it's a 4x4 matrix before assigning
                            if len(matrix) == 4 and all(
                                len(row) == 4 for row in matrix
                            ):
                                app.w_T_c = np.array(matrix)
                                app.connected = True
                                # print(f"Received: {app.w_T_c}") # Original code commented this out
                            else:
                                print(f"Received data is not a 4x4 matrix: {matrix}")

                            # The original time.sleep(1) here would make the server very slow
                            # to process frequent updates. Reducing or removing it.
                            # If a delay is needed, it should be much smaller, e.g., for yielding.
                            time.sleep(1)

                        except ValueError as ve:
                            print(
                                f"ValueError processing received data: {ve}. Data: '{data.decode() if data else ''}'"
                            )
                            # Optionally break or continue based on error severity
                            break
                        except Exception as e:
                            print(f"Error during connection with {addr}: {e}")
                            break  # Break inner loop on other errors
                    print(f"Connection from {addr} closed")
        except Exception as e:
            print(f"Server error: {e}")
