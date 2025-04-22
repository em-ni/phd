import re
import serial # type: ignore
import time

# Setup serial connection (adjust the port name to match your system)
ser = serial.Serial("/dev/ttyACM0", 9600)  # Example for Linux
# ser = serial.Serial("COM3", 9600)  # Example for Windows


def send_command(command):
    """Send a single character command to the Arduino."""
    ser.write(command.encode())  # Encode the string to bytes
    print(f"Sent '{command}' to Arduino")
    time.sleep(0.1)  # Give some time for Arduino to process the command


def read_arduino():

    try:
        while True:
            # Read a line from the serial port
            line = ser.readline().decode("utf-8").strip()
            # print(line)

            # Extract the pressure value using regex
            match = re.search(r"Voltage: (\d+\.\d+)", line)
            if match:
                voltage_value = float(match.group(1))
                print(f"Voltage: {voltage_value} V")

    except KeyboardInterrupt:
        # Close the serial connection when you terminate the script
        ser.close()
        print("Serial connection closed.")


def main():
    try:
        while True:
            # # User inputs the command
            # command = input(
            #     "Enter command ('f' for forward, 'b' for backward, 's' for stop): "
            # )
            # if command in ["f", "b", "s"]:
            #     send_command(command)
            # else:
            #     print("Invalid command. Please enter only 'f', 'b', or 's'.")

            read_arduino()
    except KeyboardInterrupt:
        print("Program exited by user.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
