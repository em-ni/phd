# runSofa -l /home/emanuele/Desktop/github/sim/sofa/build/v22.12/lib/libSofaPython3.so ./main_1dof.py 

import Sofa
from utils.pressureController import PressureController
import math
from utils.functions import add_cube, add_floor

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
    
    ## OBJECTS
    #add_floor(rootNode, [-130, 0, 0], [0, 0, 270])
    #add_cube(rootNode, [-100, 25, 40, 0, 0, 0, 1])

    ## CATHETER
    # Catheter Model
    catheter = rootNode.addChild('catheter')
    catheter.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness=0.1, rayleighMass=0.1)
    catheter.addObject('SparseLDLSolver', name='preconditioner')
    catheter.addObject('MeshVTKLoader', name='loader', filename='data/mesh/1dof/cylinder_with_cavity.vtk', scale=20, translation=translationCatheter, rotation=anglesCathter)
    catheter.addObject('MeshTopology', src='@loader', name='container')
    catheter.addObject('MechanicalObject', name='tetras', template='Vec3', showIndices=False, showIndicesScale=4e-5)
    catheter.addObject('UniformMass', totalMass=0.04)
    catheter.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large', poissonRatio=0.3, youngModulus=youngModulusCatheters)
    catheter.addObject('BoxROI', name='boxROI', box=[20, 15, -10, -18, 35, 10], doUpdate=False, drawBoxes=True)
    catheter.addObject('BoxROI', name='boxROISubTopo', box=[-118, 22.5, 0, -18, 28, 8], strict=False, drawBoxes=True)
    catheter.addObject('RestShapeSpringsForceField', points='@boxROI.indices', stiffness=1e12, angularStiffness=1e12)
    catheter.addObject('LinearSolverConstraintCorrection')

    # Sub topology
    modelSubTopo = catheter.addChild('modelSubTopo')
    modelSubTopo.addObject('TetrahedronSetTopologyContainer', position='@loader.position', tetrahedra='@boxROISubTopo.tetrahedraInROI', name='container')
    modelSubTopo.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large', poissonRatio=0.3, youngModulus=youngModulusStiffLayerCatheters - youngModulusCatheters)

    # Constraint
    cavity = catheter.addChild('cavity')
    cavity.addObject('MeshSTLLoader', name='loader', filename='data/mesh/1dof/cavity.stl', scale=20, translation=translationCatheter, rotation=anglesCathter)
    cavity.addObject('MeshTopology', src='@loader', name='topo')
    cavity.addObject('MechanicalObject', name='cavity')
    cavity.addObject('SurfacePressureConstraint', name='SurfacePressureConstraint', template='Vec3', value=0.0001, triangles='@topo.triangles', valueType='pressure')
    cavity.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    # Collision	
    collisionCatheter = catheter.addChild('collisionCatheter')
    collisionCatheter.addObject('MeshSTLLoader', name='loader', filename='data/mesh/1dof/cylinder_with_cavity.stl', scale=20, translation=translationCatheter, rotation=anglesCathter)
    collisionCatheter.addObject('MeshTopology', src='@loader', name='topo')
    collisionCatheter.addObject('MechanicalObject', name='collisMech')
    collisionCatheter.addObject('TriangleCollisionModel', selfCollision=False)
    collisionCatheter.addObject('LineCollisionModel', selfCollision=False)
    collisionCatheter.addObject('PointCollisionModel', selfCollision=False)
    collisionCatheter.addObject('BarycentricMapping')

    # Visualization	
    modelVisu = catheter.addChild('visu')
    modelVisu.addObject('MeshSTLLoader', name='loader', filename='data/mesh/1dof/cylinder_with_cavity.stl', scale = 20, translation=translationCatheter, rotation=anglesCathter)
    modelVisu.addObject('OglModel', src='@loader', color=[0.7, 0.7, 0.7, 0.6])
    modelVisu.addObject('BarycentricMapping')

    ## SPRING
    # Spring model
    spring = rootNode.addChild('spring')
    spring.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness=0.1, rayleighMass=0.1)
    spring.addObject('SparseLDLSolver', name='preconditioner')
    spring.addObject('MeshVTKLoader', name='loader', filename='data/mesh/1dof/spring.vtk', scale=10, translation=translationSpring, rotation=anglesSpring)
    spring.addObject('MeshTopology', src='@loader', name='container')
    spring.addObject('MechanicalObject', name='tetras', template='Vec3', showIndices=False, showIndicesScale=4e-5)
    spring.addObject('UniformMass', totalMass=0.04)
    spring.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large', poissonRatio=0.3, youngModulus=youngModulusSpring)
    spring.addObject('BoxROI', name='boxROI', box=[20, 15, -10, -18, 35, 10], doUpdate=False, drawBoxes=True)
    spring.addObject('BoxROI', name='boxROISubTopo', box=[-118, 22.5, 0, -18, 28, 8], strict=False, drawBoxes=True)
    spring.addObject('RestShapeSpringsForceField', points='@boxROI.indices', stiffness=1e12, angularStiffness=1e12)
    spring.addObject('LinearSolverConstraintCorrection')

    # Sub topology
    #modelSubTopoSpring = spring.addChild('modelSubTopo')
    #modelSubTopoSpring.addObject('TetrahedronSetTopologyContainer', position='@loader.position', tetrahedra='@boxROISubTopo.tetrahedraInROI', name='container')
    #modelSubTopoSpring.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large', poissonRatio=0.3, youngModulus=youngModulusStiffLayerSpring - youngModulusSpring)

    # Collision
    #collisionSpring = spring.addChild('collisionSpring')
    #collisionSpring.addObject('MeshSTLLoader', name='loader', filename='data/mesh/1dof/spring.stl', scale=10, translation=translationSpring, rotation=anglesSpring)
    #collisionSpring.addObject('MeshTopology', src='@loader', name='topo')
    #collisionSpring.addObject('MechanicalObject', name='collisMech')
    #collisionSpring.addObject('TriangleCollisionModel', selfCollision=False)
    #collisionSpring.addObject('LineCollisionModel', selfCollision=False)
    #collisionSpring.addObject('PointCollisionModel', selfCollision=False)
    #collisionSpring.addObject('BarycentricMapping')

    # Visualization
    springVisu = spring.addChild('visu')
    springVisu.addObject('MeshSTLLoader', name='loader', filename='data/mesh/1dof/spring.stl', scale=10, translation=translationSpring, rotation=anglesSpring)
    springVisu.addObject('OglModel', src='@loader', color=[0.7, 0.7, 0.7, 0.6])
    #springVisu.addObject('BarycentricMapping') # it slows down the loading by a lot, understand if it is necessary

    
    
    rootNode.addObject( PressureController(name="PressureController", node=rootNode, device_name="catheter", real_time=real_time) )
    print("\nadded PressureController\n")
