from pynput import keyboard
from RobotControl import RobotControl


class KeyboardController:
    def __init__(
        self, couple12_port, couple34_port, arduino_port, plot=False, load=False
    ):
        self.robot_control = RobotControl(
            couple12_port=couple12_port,
            couple34_port=couple34_port,
            arduino_port=arduino_port,
            plot=plot,
            load=load,
        )
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char == "q":
                print("Exiting program.")
                self.robot_control.on_close()
                return False  # Stop listener
            elif key.char == "w":
                self.robot_control.set_moving_up(True)
            elif key.char == "s":
                self.robot_control.set_moving_down(True)
            elif key.char == "d":
                self.robot_control.set_moving_right(True)
            elif key.char == "a":
                self.robot_control.set_moving_left(True)
            elif key.char == "i":
                self.robot_control.set_moving_in(True)
            elif key.char == "o":
                self.robot_control.set_moving_out(True)
            elif key.char == "l":
                self.robot_control.load_pressure()
            elif key.char == "t":
                self.robot_control.test()
            elif key.char == "z":
                self.robot_control.set_encoders_zero()

        except AttributeError:
            pass  # Special keys (like arrow keys) don't have a char attribute

    def on_release(self, key):
        try:
            if key.char == "w":
                self.robot_control.set_moving_up(False)
            elif key.char == "s":
                self.robot_control.set_moving_down(False)
            elif key.char == "d":
                self.robot_control.set_moving_right(False)
            elif key.char == "a":
                self.robot_control.set_moving_left(False)
            elif key.char == "i":
                self.robot_control.set_moving_in(False)
            elif key.char == "o":
                self.robot_control.set_moving_out(False)
        except AttributeError:
            pass
