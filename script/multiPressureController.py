#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Sofa
import Sofa.Core
from Sofa.constants import *
import threading


def moveRestPos(rest_pos, dx, dy, dz):
    out = []
    for i in range(0,len(rest_pos)) :
        out += [[rest_pos[i][0]+dx, rest_pos[i][1]+dy, rest_pos[i][2]+dz]]
    return out

class MultiPressureController(Sofa.Core.Controller):

    def __init__(self, *args, **kwargs):
        # Initialize the base Controller class
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        # Retrieve necessary parameters from kwargs
        self.root_node = kwargs['node']
        self.device_name = kwargs['device_name']  # Name of the 'catheter' node
        self.real_time = kwargs['real_time']
        self.cavity_names = kwargs['cavity_names']  # List of cavity names ['cavity_0', 'cavity_1', ...]

        # Initialize lists to store DOFs and constraints for each cavity
        self.constraints = []
        self.dofs = []

        # Retrieve the 'catheter' node
        catheter_node = self.root_node.getChild(self.device_name)

        # Iterate over each cavity name and retrieve the corresponding node
        for cavity_name in self.cavity_names:
            # Get the cavity node
            cavity_node = catheter_node.getChild(cavity_name)

            # Retrieve the 'SurfacePressureConstraint' from the cavity node
            if cavity_node.hasObject('SurfacePressureConstraint'):
                self.constraints.append(cavity_node.getObject('SurfacePressureConstraint'))
            else:
                raise AttributeError(f"Cavity '{cavity_name}' does not have a 'SurfacePressureConstraint'")

        # Initialize real-time threading if applicable
        if self.real_time:
            thread = threading.Thread(target=self.readArduino)
            thread.start()

    def adjustPressure(self, channel_index, pressure_increment):
        print("Adjusting pressure of channel " + str(channel_index) + " by " + str(pressure_increment) + " MPa")
        pressureValue = self.constraints[channel_index].value.value[0] + pressure_increment
        if pressureValue < 0.0:
            pressureValue = 0.0
        self.constraints[channel_index].value = [pressureValue]
        

    # Default BaseObject functions********************************
    def init(self):
        pass

    def bwdInit():
        pass
    
    def reinit():
        pass

    # Default Events *********************************************
    def onAnimateBeginEvent(self, event): # called at each begin of animation step
        #print("onAnimateBeginEvent") # working
        pass

    def onAnimateEndEvent(self, event): # called at each end of animation step
        #print("onAnimateEndEvent") 
        pass

    def onKeypressedEvent(self, event):
        #print("onKeypressedEvent") 
        key = event['key']
        print("key pressed: " + str(key))
        increment = 0.01           
        
        # Use 'u' 'i' 'o' 'p' to increase pressure of the respective cavity
        if key == 'U':  # u
            print("You pressed the u key")
            self.adjustPressure(0, increment)

        if key == 'I':  # i
            self.adjustPressure(1, increment)

        if key == 'O':  # o
            self.adjustPressure(2, increment)

        if key == 'P':  # p
            self.adjustPressure(3, increment)

        # Use 'h' 'j' 'k' 'l' to decrease pressure of the respective cavity
        if key == 'H':  # h
            self.adjustPressure(0, -increment)

        if key == 'J':  # j
            self.adjustPressure(1, -increment)

        if key == 'K':  # k
            self.adjustPressure(2, -increment)

        if key == 'L':  # l
            self.adjustPressure(3, -increment)

    def onKeyreleasedEvent(self, event):
        #print("onKeyreleasedEvent")
        key = event['key']
        if ord(key) == 19:  # up
            print("You released the Up key")

        if ord(key) == 21:  # down
            print("You released the Down key")

        if ord(key) == 18:  # left
            print("You released the Left key")

        if ord(key) == 20:  # right
            print("You released the Right key")

    def onMouseEvent(self, event):
        #print("onMouseEvent")
        if (event['State']== 0): # mouse moving
            print("Mouse is moving (x,y) = "+str(event['mouseX'])+" , "+str(event['mouseY']))

        if (event['State']==1): # left mouse clicked
            print("Left mouse clicked")

        if (event['State']==2): # left mouse released
            print("Left mouse released")

        if (event['State']==3): # right mouse released
            print("Right mouse clicked")

        if (event['State']==4): # right mouse released
            print("Right mouse released")

        if (event['State']==5): # wheel clicked
            print("Mouse wheel clicked")

        if (event['State']==6): # wheel released
            print("Mouse wheel released")

    def onScriptEvent(self, event):
        pass

    def onEvent(self, event):
        #print("onEvent")
        pass
    
    def onKeyPressedEvent(self,e):
        #print("key pressed")
        """ 
        # Move with arrows            
        if e["key"] == Sofa.constants.Key.uparrow:
            print("up arrow pressed")
            results = moveRestPos(self.dofs.rest_position.value, 3.0, 0.0, 0.0)
            self.dofs.rest_position.value = results

        elif e["key"] == Sofa.constants.Key.downarrow:
            print("down arrow pressed")
            results = moveRestPos(self.dofs.rest_position.value, -3.0, 0.0, 0.0)
            self.dofs.rest_position.value = results """
        
    def readArduino(self):
        import serial
        import re

        # Set up the serial connection
        arduino = serial.Serial('/dev/ttyACM0', 9600) 

        try:
            while True:
                # Read a line from the serial port
                line = arduino.readline().decode('utf-8').strip()
                print(line)
                
                # Extract the pressure value using regex
                match = re.search(r'Pressure from A0: (\d+\.\d+) MPa', line)
                if match:
                    pressure_value = float(match.group(1))
                    #print(pressure_value)

                    if pressure_value < 0.0:
                        pressure_value = 0.0

                    if pressure_value > 2:
                        pressure_value = 2
                    
                    # Set the pressure value
                    self.constraints[0].value = [pressure_value]
                    #print("Pressure value: " + str(self.constraints[0].value.value[0]))

        except KeyboardInterrupt:
            # Close the serial connection when you terminate the script
            arduino.close()
            print("Serial connection closed.")







""" def createScene(root):
    root.dt = 0.01
    root.addObject('DefaultVisualManagerLoop')
    root.addObject('DefaultAnimationLoop')

    # Add our python controller in the scene
    root.addObject( PressureController(name="PressureController") )


def main():
    import SofaRuntime
    import Sofa.Gui
    SofaRuntime.importPlugin("SofaOpenglVisual")
    root=Sofa.Core.Node("root")
    createScene(root)

    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()

    print("End of simulation.")


if __name__ == '__main__':
    main()
 """