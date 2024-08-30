"""
motor_1 = Couple12.M1
motor_2 = Couple12.M2
motor_3 = Couple34.M1
motor_4 = Couple34.M2

up = motor_1
down = motor_3
right = motor_2
left = motor_4

down-up range
motor_3      -   motor_1
[-3000000,0) U [0,3000000]

left-right range
motor_4      -   motor_2
[-4000000,0) U [0,4000000]
4000000      - 0 - 4000000
"""

import collections
import json
import socket
import threading
import time
from roboclaw_python.roboclaw_3 import Roboclaw
import serial
import re
import subprocess
import websocket


class RobotControl:
    def __init__(
        self,
        couple12_port,
        couple34_port,
        arduino_port,
        baud_rate=115200,
        plot=False,
        load=False,
    ):

        # Initialize motors
        self.couple12 = Roboclaw(couple12_port, baud_rate)
        self.couple34 = Roboclaw(couple34_port, baud_rate)
        self.couple12.Open()
        self.couple34.Open()
        self.address = 0x80
        print("Motors initialized")

        # Set up the serial connection
        self.arduino = serial.Serial(arduino_port, 9600)
        print("Arduino serial connection initialized")

        # Initialize parameters
        self.speed = 80
        self.pressure_switch_threshold = 20
        self.max_pressure_1 = 60
        self.max_pressure_2 = 60
        self.max_pressure_3 = 60
        self.max_pressure_4 = 60
        self.min_pressure = 1
        self.running = True
        print("Parameters initialized")

        # Pressure sensors [psi]
        self.p1 = 0
        self.p2 = 0
        self.p3 = 0
        self.p4 = 0

        # Moving averages initialization
        self.moving_avg_size = 100
        self.p1_buffer = collections.deque(maxlen=self.moving_avg_size)
        self.p2_buffer = collections.deque(maxlen=self.moving_avg_size)
        self.p3_buffer = collections.deque(maxlen=self.moving_avg_size)
        self.p4_buffer = collections.deque(maxlen=self.moving_avg_size)
        print("Pressure sensors initialized")

        # Variable set by the parent keyboard controller
        self.moving_up = False
        self.moving_down = False
        self.moving_right = False
        self.moving_left = False
        self.moving_in = False
        self.moving_out = False
        self.move_up_prev = False
        self.move_down_prev = False
        self.move_right_prev = False
        self.move_left_prev = False
        self.moving_in_prev = False
        self.moving_out_prev = False
        print("Movement variables initialized")

        # Socket variables
        self.host = "127.0.0.1"
        self.port = 5000

        # Set intial position to zero
        self.go_to_zero()
        print("Motors at zero position")

        # Go to loaded state
        if load:
            self.go_to_loaded_state()
            print("Motors at loaded state")

        # Main thread
        self.main_thread = threading.Thread(target=self.run, daemon=True)
        self.main_thread.start()

        # Read sensors thread
        self.sensors_thread = threading.Thread(target=self.read_sensors, daemon=True)
        self.sensors_thread.start()

        # After the server thread is started, the client can connect to the server
        # Use subprocess.Popen for asynchronous execution
        self.sensor_process = subprocess.Popen(
            ["daq/cpp/build/read_sensors"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Non-blocking way to get output from the subprocess
        self.read_output_thread = threading.Thread(target=self.read_process_output)
        self.read_output_thread.start()

        if plot:
            # Send data thread
            send_thread = threading.Thread(target=self.send_data, daemon=True)
            send_thread.start()

        print("Robot control initialized")

    def go_to_loaded_state(self):
        # Read from file if it exists
        try:
            with open("loaded_state.txt", "r") as f:
                enc1 = int(f.readline())
                enc2 = int(f.readline())
                enc3 = int(f.readline())
                enc4 = int(f.readline())
                print("Encoder values loaded from loaded_state.txt")
                print(f"Motor 1: {enc1}")
                print(f"Motor 2: {enc2}")
                print(f"Motor 3: {enc3}")
                print(f"Motor 4: {enc4}")
        except FileNotFoundError:
            print("No saved state found")
            return

        # Move to loaded state using move_to_position
        self.move_to_position("1", enc1)
        self.move_to_position("2", enc2)
        self.move_to_position("3", enc3)
        self.move_to_position("4", enc4)
        time.sleep(1)

    def go_to_zero(self):
        print("Moving to zero position")
        # Position all motors to zero position
        self.move_to_position("1", 0)
        self.move_to_position("2", 0)
        self.move_to_position("3", 0)
        self.move_to_position("4", 0)
        time.sleep(1)
        print("Motors at zero position")

    def load_pressure(self):
        # Bring every motor to the switch pressure threshold
        print("\nLoading pressure and saving new state")
        # Load motor 1
        while self.p1 < self.pressure_switch_threshold:
            self.couple12.ForwardM1(self.address, self.speed // 10)
            print(
                f"\rPressure 1: {self.p1:.3f} / {self.pressure_switch_threshold}",
                end="",
                flush=True,
            )
            time.sleep(0.2)
        print("\nStopping motor 1")
        counter = 0
        while counter < 10:
            self.couple12.ForwardM1(self.address, 0)
            time.sleep(0.2)
            counter += 1
        print("Motor 1 reached pressure threshold\n")
        time.sleep(0.2)

        # Load motor 2
        while self.p2 < self.pressure_switch_threshold:
            self.couple12.ForwardM2(self.address, self.speed // 10)
            print(
                f"\rPressure 2: {self.p2:.3f} / {self.pressure_switch_threshold}",
                end="",
                flush=True,
            )
            time.sleep(0.2)
        print("\nStopping motor 2")
        counter = 0
        while counter < 10:
            self.couple12.ForwardM2(self.address, 0)
            time.sleep(0.2)
            counter += 1
        print("Motor 2 reached pressure threshold\n")
        time.sleep(0.2)

        # Load motor 3
        while self.p3 < self.pressure_switch_threshold:
            self.couple34.ForwardM1(self.address, self.speed // 10)
            print(
                f"\rPressure 3: {self.p3:.3f} / {self.pressure_switch_threshold}",
                end="",
                flush=True,
            )
            time.sleep(0.2)
        print("\nStopping motor 3")
        counter = 0
        while counter < 10:
            self.couple34.ForwardM1(self.address, 0)
            time.sleep(0.2)
            counter += 1
        print("Motor 3 reached pressure threshold\n")
        time.sleep(0.2)

        # Load motor 4
        while self.p4 < self.pressure_switch_threshold:
            self.couple34.ForwardM2(self.address, self.speed // 10)
            print(
                f"\rPressure 4: {self.p4:.3f} / {self.pressure_switch_threshold}",
                end="",
                flush=True,
            )
            time.sleep(0.2)
        print("\nStopping motor 4")
        counter = 0
        while counter < 10:
            self.couple34.ForwardM2(self.address, 0)
            time.sleep(0.2)
            counter += 1
        print("Motor 4 reached pressure threshold\n")
        time.sleep(0.2)

        print("Loading pressure completed\n")

        # Read encoder values of each motor
        print("Encoder values:")
        enc1 = self.couple12.ReadEncM1(self.address)[1]
        time.sleep(1)
        enc2 = self.couple12.ReadEncM2(self.address)[1]
        time.sleep(1)
        enc3 = self.couple34.ReadEncM1(self.address)[1]
        time.sleep(1)
        enc4 = self.couple34.ReadEncM2(self.address)[1]
        time.sleep(1)
        print(f"Motor 1: {enc1}")
        print(f"Motor 2: {enc2}")
        print(f"Motor 3: {enc3}")
        print(f"Motor 4: {enc4}")

        # Save encoder values to file
        with open("loaded_state.txt", "w") as f:
            f.write(f"{enc1}\n{enc2}\n{enc3}\n{enc4}")
            print("Encoder values saved to loaded_state.txt")

    # Move functions ----------------------------
    def move_down(self):
        print("RobotControl: Moving down")
        if self.p1 < self.pressure_switch_threshold:
            if self.p3 < self.max_pressure_3:
                self.couple34.ForwardM1(self.address, self.speed)
            else:
                print("Pressure max limit reached for motor 3")
        else:
            if self.p1 > self.min_pressure:
                self.couple12.BackwardM1(self.address, self.speed)
            else:
                print("Pressure min limit reached for motor 1")

        time.sleep(0.1)

    def move_left(self):
        print("RobotControl: Moving left")
        if self.p2 < self.pressure_switch_threshold:
            if self.p4 < self.max_pressure_4:
                self.couple34.ForwardM2(self.address, self.speed)
            else:
                print("Pressure max limit reached for motor 4")
        else:
            if self.p2 > self.min_pressure:
                self.couple12.BackwardM2(self.address, self.speed)
            else:
                print("Pressure min limit reached for motor 2")

        time.sleep(0.1)

    def move_right(self):
        print("RobotControl: Moving right")
        if self.p4 < self.pressure_switch_threshold:
            if self.p2 < self.max_pressure_2:
                self.couple12.ForwardM2(self.address, self.speed)
            else:
                print("Pressure max limit reached for motor 2")
        else:
            if self.p2 > self.min_pressure:
                self.couple34.BackwardM2(self.address, self.speed)
            else:
                print("Pressure min limit reached for motor 4")

        time.sleep(0.1)

    def move_up(self):
        print("RobotControl: Moving up")
        if self.p3 < self.pressure_switch_threshold:
            if self.p1 < self.max_pressure_1:
                self.couple12.ForwardM1(self.address, self.speed)
            else:
                print("Pressure max limit reached for motor 1")
        else:
            if self.p3 > self.min_pressure:
                self.couple34.BackwardM1(self.address, self.speed)
            else:
                print("Pressure min limit reached for motor 3")

        time.sleep(0.1)

    def move_to_position(self, motor, position):
        if motor == "1":
            self.couple12.SpeedAccelDeccelPositionM1(
                self.address, 320000, 500000, 320000, position, 0
            )
        if motor == "2":
            self.couple12.SpeedAccelDeccelPositionM2(
                self.address, 320000, 500000, 320000, position, 0
            )
        if motor == "3":
            self.couple34.SpeedAccelDeccelPositionM1(
                self.address, 320000, 500000, 320000, position, 0
            )
        if motor == "4":
            self.couple34.SpeedAccelDeccelPositionM2(
                self.address, 320000, 500000, 320000, position, 0
            )
        time.sleep(0.1)

    # ----------------------------

    def on_close(self):
        # Stop run thread and close motors
        self.stop()

        print("Closing motors")
        self.go_to_zero()

        print("Closing serial connection")
        self.send_arduino("s")
        self.arduino.close()

        print("Closing threads")

        # Stop sensors thread
        self.sensors_thread.join()
        print("Sensors thread closed")

        # Stop subprocess
        self.sensor_process.terminate()
        self.read_output_thread.join()
        print("Subprocess closed")

        # Try to start and stop again before closing to avoid the double start problem
        self.sensor_process = subprocess.Popen(
            ["daq/cpp/build/read_sensors"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Non-blocking way to get output from the subprocess
        self.read_output_thread = threading.Thread(target=self.read_process_output)
        self.read_output_thread.start()
        self.sensor_process.terminate()
        self.read_output_thread.join()
        print("Subprocess closed")

        self.main_thread.join()
        print("Main thread closed")

    def read_sensors(self):
        # Create a socket object using the AF_INET address family and SOCK_STREAM socket type
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the address and port number
        server_socket.bind((self.host, self.port))

        # Start listening for incoming connections with a backlog of 5
        server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}")

        try:
            while self.running:
                # Accept a connection
                client_socket, addr = server_socket.accept()
                print(f"Connected by {addr}")

                try:
                    while self.running:
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
                            channel, pressure = int(match.group(1)), float(
                                match.group(2)
                            )

                            self.update_pressure(channel, pressure)

                finally:
                    client_socket.close()  # Ensure the client socket is closed after processing
                    print("Connection closed.")

        except KeyboardInterrupt:
            print("Server is shutting down.")

        finally:
            server_socket.close()  # Ensure the server socket is closed when done
            print("Server closed.")

    def read_process_output(self):
        # Print stdout and stderr without blocking
        stdout, stderr = self.sensor_process.communicate()
        # print("Subprocess read_sensors stdout:", stdout)
        print("Subprocess read_sensors stderr:", stderr)

        if self.sensor_process.returncode != 0:
            print(f"Process exited with code {self.sensor_process.returncode}")

    # main loop
    def run(self):
        while self.running:
            # print("run")
            # print(self.moving_up, self.moving_down, self.moving_right, self.moving_left)
            if self.moving_up and not self.move_up_prev:
                self.move_up()
                self.move_up_prev = True
            if not self.moving_up and self.move_up_prev:
                self.stop_motors()
                self.move_up_prev = False
            if self.moving_down and not self.move_down_prev:
                self.move_down()
                self.move_down_prev = True
            if not self.moving_down and self.move_down_prev:
                self.stop_motors()
                self.move_down_prev = False
            if self.moving_right and not self.move_right_prev:
                self.move_right()
                self.move_right_prev = True
            if not self.moving_right and self.move_right_prev:
                self.stop_motors()
                self.move_right_prev = False
            if self.moving_left and not self.move_left_prev:
                self.move_left()
                self.move_left_prev = True
            if not self.moving_left and self.move_left_prev:
                self.stop_motors()
                self.move_left_prev = False
            if self.moving_in and not self.moving_in_prev:
                if self.moving_out:
                    self.moving_out = False
                    continue
                self.send_arduino("f")
                self.moving_in_prev = True
            if not self.moving_in and self.moving_in_prev:
                self.send_arduino("s")
                self.moving_in_prev = False
            if self.moving_out and not self.moving_out_prev:
                if self.moving_in:
                    self.moving_in = False
                    continue
                self.send_arduino("b")
                self.moving_out_prev = True
            if not self.moving_out and self.moving_out_prev:
                self.send_arduino("s")
                self.moving_out_prev = False
            # time.sleep(0.01)

    def send_arduino(self, command):
        self.arduino.write(command.encode())
        print(f"Sent '{command}' to Arduino")
        time.sleep(0.1)

    def send_data(self):

        while self.running:
            try:
                ws = websocket.create_connection("ws://127.0.0.1:65432")
                try:
                    while self.running:
                        data = json.dumps(
                            {
                                "p1": self.p1,
                                "p2": self.p2,
                                "p3": self.p3,
                                "p4": self.p4,
                            }
                        )
                        ws.send(data)
                        time.sleep(0.1)
                finally:
                    ws.close()
            except ConnectionRefusedError:
                print("Connection refused. Retrying in 5 seconds.")
                time.sleep(5)

    # Set functions
    def set_moving_up(self, value):
        self.moving_up = value

    def set_moving_down(self, value):
        self.moving_down = value

    def set_moving_right(self, value):
        self.moving_right = value

    def set_moving_left(self, value):
        self.moving_left = value

    def set_moving_in(self, value):
        self.moving_in = value

    def set_moving_out(self, value):
        self.moving_out = value

    def stop(self):
        self.running = False

    def set_encoders_zero(self):
        print("TODO")

    # ----------------------------

    def stop_motors(self):
        print("RobotControl: Stopping motors")
        self.couple12.ForwardM1(self.address, 0)
        self.couple12.ForwardM2(self.address, 0)
        self.couple34.ForwardM1(self.address, 0)
        self.couple34.ForwardM2(self.address, 0)
        time.sleep(0.1)

    def test(self):
        print("Test")
        self.move_to_position("1", 1000000)

    def update_pressure(self, channel, pressure):
        # Get the appropriate buffer based on channel number
        buffer = getattr(self, f"p{channel}_buffer")
        buffer.append(pressure)  # Append new pressure reading

        # Calculate moving average
        avg_pressure = sum(buffer) / len(buffer)
        if avg_pressure < 0:
            avg_pressure = 0

        setattr(self, f"p{channel}", avg_pressure)  # Update the pressure attribute

        # print(f"Filtered pressure for Channel {channel}: {avg_pressure} psi")
