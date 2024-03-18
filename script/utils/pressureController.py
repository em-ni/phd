#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Sofa
import Sofa.Core
from Sofa.constants import *
import threading
import socket
import struct
import json
import time

def moveRestPos(rest_pos, dx, dy, dz):
    out = []
    for i in range(0,len(rest_pos)) :
        out += [[rest_pos[i][0]+dx, rest_pos[i][1]+dy, rest_pos[i][2]+dz]]
    return out

class PressureController(Sofa.Core.Controller):

    def __init__(self, *a, **kw): # arga, kwargs
        
        Sofa.Core.Controller.__init__(self, *a, **kw)
        self.root_node = kw['node']
        self.device_name = kw['device_name']
        self.real_time = kw['real_time']
        self.communication = kw['communication']
        self.debug = kw['debug']
        self.plot = kw['plot']
        self.constraints = []
        self.dofs = []
        
        self.dofs.append(self.root_node.getChild(self.device_name).tetras)
        self.constraints.append(self.root_node.getChild(self.device_name).cavity.SurfacePressureConstraint)

        if self.real_time:
            if self.communication == 'UDP':
                thread = threading.Thread(target=self.readUDP)
                thread.start()
            elif self.communication == 'Arduino':
                thread = threading.Thread(target=self.readArduino)
                thread.start()

        if self.plot:
            thread = threading.Thread(target=self.sendData)
            thread.start()
                
        
    # Default BaseObject functions********************************
    def init(self):
        pass

    def bwdInit():
        pass
    
    def reinit():
        pass

    # Default Events *********************************************
    def onAnimateBeginEvent(self, event): # called at each begin of animation step
        #if self.debug : print("onAnimateBeginEvent") # working
        pass

    def onAnimateEndEvent(self, event): # called at each end of animation step
        #if self.debug : print("onAnimateEndEvent") 
        pass

    def onKeypressedEvent(self, event):
        #if self.debug : print("onKeypressedEvent") 
        key = event['key']
        increment = 0.01           
        
        if ord(key) == 19:  # up
            if self.debug : print("You pressed the Up key")
            pressureValue = self.constraints[0].value.value[0] + increment
            if pressureValue > 1.5:
                pressureValue = 1.5
            self.constraints[0].value = [pressureValue]

            if self.debug : print("Pressure value: " + str(self.constraints[0].value.value[0]))

        if ord(key) == 21:  # down
            if self.debug : print("You pressed the Down key")
            pressureValue = self.constraints[0].value.value[0] - increment
            if pressureValue < 0.0:
                pressureValue = 0.0
            self.constraints[0].value = [pressureValue]

            if self.debug : print("Pressure value: " + str(self.constraints[0].value.value[0]))

        if ord(key) == 18:  # left
            if self.debug : print("You pressed the Left key")

        if ord(key) == 20:  # right
            if self.debug : print("You pressed the Right key")

    def onKeyreleasedEvent(self, event):
        #if self.debug : print("onKeyreleasedEvent")
        key = event['key']
        if ord(key) == 19:  # up
            if self.debug : print("You released the Up key")

        if ord(key) == 21:  # down
            if self.debug : print("You released the Down key")

        if ord(key) == 18:  # left
            if self.debug : print("You released the Left key")

        if ord(key) == 20:  # right
            if self.debug : print("You released the Right key")

    def onMouseEvent(self, event):
        #if self.debug : print("onMouseEvent")
        if (event['State']== 0): # mouse moving
            if self.debug : print("Mouse is moving (x,y) = "+str(event['mouseX'])+" , "+str(event['mouseY']))

        if (event['State']==1): # left mouse clicked
            if self.debug : print("Left mouse clicked")

        if (event['State']==2): # left mouse released
            if self.debug : print("Left mouse released")

        if (event['State']==3): # right mouse released
            if self.debug : print("Right mouse clicked")

        if (event['State']==4): # right mouse released
            if self.debug : print("Right mouse released")

        if (event['State']==5): # wheel clicked
            if self.debug : print("Mouse wheel clicked")

        if (event['State']==6): # wheel released
            if self.debug : print("Mouse wheel released")

    def onScriptEvent(self, event):
        pass

    def onEvent(self, event):
        #if self.debug : print("onEvent")
        pass
    
    def onKeyPressedEvent(self,e):
        #if self.debug : print("key pressed")
        """ 
        # Move with arrows            
        if e["key"] == Sofa.constants.Key.uparrow:
            if self.debug : print("up arrow pressed")
            results = moveRestPos(self.dofs.rest_position.value, 3.0, 0.0, 0.0)
            self.dofs.rest_position.value = results

        elif e["key"] == Sofa.constants.Key.downarrow:
            if self.debug : print("down arrow pressed")
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
                if self.debug : print(line)
                
                # Extract the pressure value using regex
                match = re.search(r'Pressure from A0: (\d+\.\d+) MPa', line)
                if match:
                    pressure_value = float(match.group(1))
                    #if self.debug : print(pressure_value)

                    if pressure_value < 0.0:
                        pressure_value = 0.0

                    if pressure_value > 2:
                        pressure_value = 2
                    
                    # Set the pressure value
                    self.constraints[0].value = [pressure_value]
                    #if self.debug : print("Pressure value: " + str(self.constraints[0].value.value[0]))

        except KeyboardInterrupt:
            # Close the serial connection when you terminate the script
            arduino.close()
            if self.debug : print("Serial connection closed.")
              
    def readUDP(self):
        
        # Configurationn
        UDP_IP = "192.168.130.148"
        UDP_PORT = 25000

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))

        print(f"Listening on {UDP_IP}:{UDP_PORT}")

        try:
            while True:
                data, addr = sock.recvfrom(1024)  # Adjust buffer size if necessary
                
                if len(data) == 4:  # Likely a single precision float or 32-bit integer
                    try:
                        # Attempt to decode as a single precision float
                        value_float = struct.unpack('f', data)
                        if self.debug : print(f"Received float: {value_float[0]} from {addr}")
                    except:
                        pass

                    try:
                        # Attempt to decode as a 32-bit integer
                        value_int = struct.unpack('i', data)
                        if self.debug : print(f"Received int: {value_int[0]} from {addr}")
                    except:
                        pass

                elif len(data) == 8:  # Likely a double precision float or 64-bit integer
                    try:
                        # Attempt to decode as a double precision float
                        value_double = struct.unpack('d', data)
                        #print(f"Received double: {value_double[0]} from {addr}")
                        
                        # Set received pressure value
                        self.constraints[0].value = [value_double[0]]
                    except:
                        pass

                # Example of decoding a fixed-length string (adjust length as needed)
                elif len(data) > 0:  # Assuming there's no fixed size, trying to interpret as a string
                    try:
                        # Decode as UTF-8 string, replace errors to avoid exceptions
                        value_str = data.decode('utf-8', errors='replace')
                        if self.debug : print(f"Received string: {value_str} from {addr}")
                    except:
                        if self.debug : print(f"Received unhandled data type: {data} from {addr}")

        except KeyboardInterrupt:
            print("Stopping receiver.")
            sock.close()
            
    def sendData(self):
        print("Sending data")
        host = '127.0.0.1'  # The server's hostname or IP address
        port = 65432        # The port used by the server

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            try:
                while True:  # Continuously send data
                    pressure = self.constraints[0].value.value[0]
                    data = {"p": pressure}  
                    s.sendall(json.dumps(data).encode('utf-8'))
                    if self.debug: print(f"Sent data: {data}")
                    time.sleep(0.1)  
            except Exception as e:
                print(f"Error sending data: {e}")
            
            
            
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

    if self.debug : print("End of simulation.")


if __name__ == '__main__':
    main()
 """