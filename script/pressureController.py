#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Sofa
import Sofa.Core
from Sofa.constants import *

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
        self.constraints = []
        self.dofs = []
        
        self.dofs.append(self.root_node.getChild(self.device_name).tetras)
        self.constraints.append(self.root_node.getChild(self.device_name).cavity.SurfacePressureConstraint)
        
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
        increment = 0.01           

        
        if ord(key) == 19:  # up
            print("You pressed the Up key")
            #esults = moveRestPos(self.root_node.getChild('finger1').tetras.position.value , 3.0, 0.0, 0.0)
            #print(results)
            #self.dofs.rest_position.value = results
            
            """             cube_translation = self.root_node.getChild('cube').getChild('cubeVisu').cube_loader.translation.value
            print("Before increment", cube_translation)
            # list all attributes of the object
            print(dir(self.root_node.getChild('cube').getChild('cubeVisu').cube_loader))
            # Move the cube up of an increment
            increment = 100

            # Update the position of the cube (READ ONLY :( )
            self.root_node.getChild('cube').getChild('cubeVisu').cube_loader.translation.value = cube_translation + [increment, 0, 0]

            cube_translation = self.root_node.getChild('cube').getChild('cubeVisu').cube_loader.translation.value
            print("After increment", cube_translation) """

            for i in range(1):
                pressureValue = self.constraints[i].value.value[0] + increment
                if pressureValue > 1.5:
                    pressureValue = 1.5
                self.constraints[i].value = [pressureValue]



            print("done")

        if ord(key) == 21:  # down
            print("You pressed the Down key")
            results = moveRestPos(self.dofs.rest_position.value, -3.0, 0.0, 0.0)
            self.dofs.rest_position.value = results

        if ord(key) == 18:  # left
            print("You pressed the Left key")

        if ord(key) == 20:  # right
            print("You pressed the Right key")

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

def createScene(root):
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
