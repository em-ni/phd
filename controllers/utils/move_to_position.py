import time
from roboclaw_python.roboclaw_3 import Roboclaw


while True:
    # Ask user which motor to control
    motor = input("Which motor to control? (1,2,3,4): ")

    if motor == "1" or motor == "2":
        couple = Roboclaw("/dev/ttyACM0", 115200)
        couple.Open(),
    elif motor == "3" or motor == "4":
        couple = Roboclaw("/dev/ttyACM1", 115200)
        couple.Open()
    else:
        print("Invalid motor number")
        exit()

    address = 0x80

    # Ask user for position
    position = int(input("Enter position: "))
    if motor == "1" or motor == "3":
        limit_range = [0, 3000000]
    else:
        limit_range = [0, 4000000]

    # Check if position is within range
    if position < limit_range[0] or position > limit_range[1]:
        print("Position out of range")
        break

    # Move to position
    if motor == "1" or motor == "3":
        couple.SpeedAccelDeccelPositionM1(address, 320000, 500000, 320000, position, 0)
        time.sleep(0.01)

        # Read encoder
        for i in range(0, 80):
            enc = couple.ReadEncM1(address)
            print("Encoder"),
            if enc[0] == 1:
                print(enc[1])
                # print(format(enc[2], "02x"))
            else:
                print("failed ")
            time.sleep(0.1)
    else:
        couple.SpeedAccelDeccelPositionM2(address, 320000, 500000, 320000, position, 0)
        time.sleep(0.01)

        # Read encoder
        for i in range(0, 80):
            enc = couple.ReadEncM1(address)
            print("Encoder:")
            if enc[0] == 1:
                print(enc[1])
                # print(format(enc[2], "02x"))
            else:
                print("failed ")
            time.sleep(0.1)
