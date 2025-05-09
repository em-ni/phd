# runSofa -l /home/emanuele/Desktop/github/sim/sofa/build/v23.12/lib/libSofaPython3.so ./main_1dof.py

import Sofa
from utils.pressureController import PressureController
from utils.contactListener import ContactListener
from utils.functions import (
    add_cube,
    add_floor,
    add_spring,
    add_catheter,
    add_wall,
    add_organ,
)

# Simulation parameters
real_time = False
gpu = False
object = "wall"
# object = "organ"
# object = None

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
    rootNode.addObject(
        "VisualStyle",
        displayFlags="showVisualModels hideBehaviorModels hideCollisionModels hideBoundingCollisionModels showForceFields showInteractionForceFields hideWireframe",
    )
    plugin_names = "SoftRobots SofaPython3 MultiThreading Sofa.Component.MechanicalLoad Sofa.Component.Mapping.NonLinear Sofa.Component.Constraint.Projective Sofa.Component.AnimationLoop Sofa.Component.Collision.Detection.Algorithm Sofa.Component.Collision.Detection.Intersection Sofa.Component.Collision.Geometry Sofa.Component.Collision.Response.Contact Sofa.Component.Constraint.Lagrangian.Correction Sofa.Component.Constraint.Lagrangian.Solver Sofa.Component.Engine.Select Sofa.Component.IO.Mesh Sofa.Component.LinearSolver.Direct Sofa.Component.Mapping.Linear Sofa.Component.Mass Sofa.Component.ODESolver.Backward Sofa.Component.Setting Sofa.Component.SolidMechanics.FEM.Elastic Sofa.Component.SolidMechanics.Spring Sofa.Component.StateContainer Sofa.Component.Topology.Container.Constant Sofa.Component.Topology.Container.Dynamic Sofa.Component.Visual Sofa.GL.Component.Rendering3D Sofa.GUI.Component"
    if gpu:
        plugin_names = plugin_names + " SofaCUDA"

    rootNode.addObject("RequiredPlugin", pluginName=plugin_names)
    # rootNode.gravity.value = [-9810, 0, 0]
    rootNode.gravity.value = [0, 0, 0]
    rootNode.addObject("AttachBodyButtonSetting", stiffness=10)
    rootNode.addObject("FreeMotionAnimationLoop")
    rootNode.addObject(
        "GenericConstraintSolver",
        tolerance=1e-12,
        maxIterations=10000,
        computeConstraintForces=True,
    )
    # rootNode.addObject('EulerImplicitSolver', name='odesolver', firstOrder=False, rayleighMass=0.1, rayleighStiffness=0.1)

    # Add scene objects
    rootNode.addObject("DefaultPipeline", depth=15, verbose=0, draw=0)

    if gpu:
        rootNode.addObject("ParallelBruteForceBroadPhase")
        rootNode.addObject("ParallelBVHNarrowPhase")
    else:
        rootNode.addObject("BruteForceBroadPhase")
        rootNode.addObject("BVHNarrowPhase")

    rootNode.addObject(
        "DefaultContactManager",
        response="FrictionContactConstraint",
        responseParams="mu=0.6",
    )
    rootNode.addObject(
        "LocalMinDistance",
        name="Proximity",
        alarmDistance=5,
        contactDistance=1,
        angleCone=0.0,
    )

    # Background
    rootNode.addObject("BackgroundSetting", color=[0, 0.168627, 0.211765, 1.0])
    rootNode.addObject("OglSceneFrame", style="Arrows", alignment="TopRight")

    # Objects
    # add_floor(rootNode, [-130, 0, 0], [0, 0, 270])
    # add_floor(rootNode, [-130, 0, 30], [90, 0, 180])
    # add_cube(rootNode, [-100, 25, 40, 0, 0, 0, 1])

    if object == "wall":
        add_wall(rootNode, [150, 190, 37], [180, 90, 0])
        rootNode.addObject(
            ContactListener(
                name="wallContactListener",
                node=rootNode,
                object_name="wall",
                collision_name="wallCollis",
                debug=True,
                plot=True,
            )
        )
    elif object == "organ":
        add_organ(
            rootNode,
            "data/mesh/vascularmodel/0167_0001_sim/print.obj",
            [0, -275, -520],
            [0, 270, 0],
            gpu,
        )
        rootNode.addObject(
            ContactListener(
                name="organContactListener",
                node=rootNode,
                object_name="organ",
                collision_name="organCollis",
                debug=False,
                plot=True,
            )
        )
    elif object == None:
        pass

    ## Catheter
    add_catheter(
        rootNode,
        translationCatheter,
        anglesCathter,
        youngModulusCatheters,
        youngModulusStiffLayerCatheters,
        gpu,
    )

    # Spring (it slows down the sim too much)
    # add_spring(rootNode, translationSpring, anglesSpring, youngModulusSpring, youngModulusStiffLayerSpring, gpu)

    # Pressure controller
    rootNode.addObject(
        PressureController(
            name="PressureController",
            node=rootNode,
            device_name="catheter",
            real_time=real_time,
            communication="UDP",
            debug=False,
            plot=True,
        )
    )
    print("\nScene loaded\n")
