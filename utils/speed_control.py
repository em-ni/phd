import tty
import sys
import termios
import time
from roboclaw_python.roboclaw_3 import Roboclaw

# Save the original terminal settings for later restoration
orig_settings = termios.tcgetattr(sys.stdin)

# Initial speed
current_speed = 0


def control_speed(couple, motor, address):
    print("Speed: {}".format(current_speed))
    if motor in ["1", "3"]:
        couple.SpeedM1(address, current_speed)
        time.sleep(0.01)
    elif motor in ["2", "4"]:
        couple.SpeedM2(address, current_speed)
        time.sleep(0.01)


while True:
    motor = input("Which motor to control? (1,2,3,4): ")
    if motor == "1" or motor == "2":
        couple = Roboclaw("/dev/ttyACM0", 115200)
        couple.Open()
    elif motor in ["3", "4"]:
        couple = Roboclaw("/dev/ttyACM1", 115200)
        couple.Open()
    elif motor == "q":
        break
    else:
        print("Invalid motor number")
        break

    address = 0x80
    print("Use the arrow keys to increase or decrease speed. Press 'q' to quit.")
    print("Speed: {}".format(current_speed))

    tty.setcbreak(sys.stdin)
    try:
        while True:
            x = sys.stdin.read(1)
            if x == "\x1b":  # Begin of escape sequence for arrow keys
                # Read next two characters
                next1, next2 = sys.stdin.read(2)
                if next1 == "[":
                    if next2 == "A":  # Up arrow
                        current_speed += 10000
                        control_speed(couple, motor, address)
                    elif next2 == "B":  # Down arrow
                        current_speed -= 10000
                        control_speed(couple, motor, address)
            elif x == "q":  # Quit
                print("Exiting program.")
                # Stop motor
                current_speed = 0
                control_speed(couple, motor, address)

                # Restore the original terminal settings before taking standard input
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)

                # Ask if user want to set this position to zero
                set_zero = input("Set current position to zero? (y/n): ")
                if set_zero == "y":
                    if motor in ["1", "3"]:
                        couple.ResetEncoders(address)
                    elif motor in ["2", "4"]:
                        couple.ResetEncoders(address)

                # Reapply cbreak settings if continuing
                tty.setcbreak(sys.stdin)

                break

            time.sleep(0.1)  # Delay to prevent rapid changes

    except KeyboardInterrupt:
        # Stop motor
        current_speed = 0
        control_speed(couple, motor, address)
        print("Program interrupted.")
    finally:
        # Stop motor
        current_speed = 0
        control_speed(couple, motor, address)

        # Restore the original terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)
        print("Terminal settings restored.")

    if x == "q":  # Quit on 'q'
        # Stop motor
        current_speed = 0
        control_speed(couple, motor, address)
        print("Program interrupted.")
        break
