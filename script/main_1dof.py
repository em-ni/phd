# runSofa -l /home/emanuele/Desktop/github/sim/sofa/build/v22.12/lib/libSofaPython3.so ./main_1dof.py 

import Sofa
from utils.pressureController import PressureController
import math
from utils.functions import add_cube, add_floor, add_spring, add_catheter

real_time = False

# Catheter parameters
youngModulusCatheters = 500
youngModulusStiffLayerCatheters = 1500
translationCatheter = [-120, 25, 0]
anglesCathter = [0, 90, 0]

# Spring parameters
youngModulusSpring = 500
youngModulusStiffLayerSpring = 1500
translationSpring = [-120, 25, 0]
anglesSpring = [0, 90, 0]

def createScene(rootNode):

    ## SETUP
    rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels hideBoundingCollisionModels hideForceFields showInteractionForceFields hideWireframe')
    rootNode.addObject('RequiredPlugin', pluginName='SoftRobots SofaPython3')
    rootNode.gravity.value = [-9810, 0, 0]
    rootNode.addObject('AttachBodyButtonSetting', stiffness=10)
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('GenericConstraintSolver', tolerance=1e-12, maxIterations=10000)

    # Add scene objects
    rootNode.addObject('DefaultPipeline')
    rootNode.addObject('BruteForceBroadPhase')
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('DefaultContactManager', response='FrictionContactConstraint', responseParams='mu=0.6')
    rootNode.addObject('LocalMinDistance', name='Proximity', alarmDistance=5, contactDistance=1, angleCone=0.0)

    # Background 
    rootNode.addObject('BackgroundSetting', color=[0, 0.168627, 0.211765, 1.])
    rootNode.addObject('OglSceneFrame', style='Arrows', alignment='TopRight')
    
    # Objects
    #add_floor(rootNode, [-130, 0, 0], [0, 0, 270])
    #add_cube(rootNode, [-100, 25, 40, 0, 0, 0, 1])

    ## Catheter
    add_catheter(rootNode, translationCatheter, anglesCathter, youngModulusCatheters, youngModulusStiffLayerCatheters)

    # Spring
    #add_spring(rootNode, translationSpring, anglesSpring, youngModulusSpring, youngModulusStiffLayerSpring)

    # Pressure controller    
    rootNode.addObject( PressureController(name="PressureController", node=rootNode, device_name="catheter", real_time=real_time) )
    print("\nadded PressureController\n")
