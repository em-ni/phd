from src.KeyboardController import KeyboardController

couple12_port = "/dev/ttyACM0"
couple34_port = "/dev/ttyACM1"
arduino_port = "/dev/ttyACM2"

if __name__ == "__main__":

    keyboard_controller = KeyboardController(
        couple12_port=couple12_port,
        couple34_port=couple34_port,
        arduino_port=arduino_port,
        plot=True,
        load=True,
    )
    keyboard_controller.listener.join()
