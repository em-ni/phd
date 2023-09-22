# runSofa -l /home/emanuele/Desktop/github/sim/sofa/build/v22.12/lib/libSofaPython3.so ./main.py 

import Sofa
from pressureController import PressureController
import math


youngModulusCatheters = 500
youngModulusStiffLayerCatheters = 1500

translationCatheter = [-120, 25, 0]
anglesCathter = [0, 90, 0]

def createScene(rootNode):
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

    # Add floor
    planeNode = rootNode.addChild('Plane')
    planeNode.addObject('MeshOBJLoader', name='plane_loader', filename='data/mesh/floorFlat.obj', triangulate=True, rotation=[0, 0, 270], scale=10, translation=[-130, 0, 0])
    planeNode.addObject('MeshTopology', src='@plane_loader')
    planeNode.addObject('MechanicalObject', src='@plane_loader')
    planeNode.addObject('TriangleCollisionModel', simulated=False, moving=False)
    planeNode.addObject('LineCollisionModel', simulated=False, moving=False)
    planeNode.addObject('PointCollisionModel', simulated=False, moving=False)
    planeNode.addObject('OglModel', name='Visual_plane', src='@plane_loader', color=[1, 0, 0, 1])

    # Add cube    
    cube = rootNode.addChild('cube')
    cube.addObject('EulerImplicitSolver', name='odesolver')
    cube.addObject('SparseLDLSolver', name='linearSolver')
    cube.addObject('MechanicalObject', template='Rigid3', position=[-100, 25, 40, 0, 0, 0, 1])
    cube.addObject('UniformMass', totalMass=0.001)
    cube.addObject('UncoupledConstraintCorrection')

    # Cube collision
    cubeScale = 5
    cubeCollis = cube.addChild('cubeCollis')
    cubeCollis.addObject('MeshOBJLoader', name='cube_loader', filename='data/mesh/smCube27.obj', triangulate=True, scale=cubeScale)
    cubeCollis.addObject('MeshTopology', src='@cube_loader')
    cubeCollis.addObject('MechanicalObject')
    cubeCollis.addObject('TriangleCollisionModel')
    cubeCollis.addObject('LineCollisionModel')
    cubeCollis.addObject('PointCollisionModel')
    cubeCollis.addObject('RigidMapping')

    # Cube visualization
    cubeVisu = cube.addChild('cubeVisu')
    cubeVisu.addObject('MeshOBJLoader', name='cube_loader', filename='data/mesh/smCube27.obj')
    cubeVisu.addObject('OglModel', name='Visual', src='@cube_loader', color=[0.0, 0.1, 0.5], scale=cubeScale)
    cubeVisu.addObject('RigidMapping')

    # Catheter Model
    catheter = rootNode.addChild('catheter')
    catheter.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness=0.1, rayleighMass=0.1)
    catheter.addObject('SparseLDLSolver', name='preconditioner')
    catheter.addObject('MeshVTKLoader', name='loader', filename='data/mesh/cylinder_with_cavity.vtk', scale=20, translation=translationCatheter, rotation=anglesCathter)
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
    cavity.addObject('MeshSTLLoader', name='loader', filename='data/mesh/cavity.stl', scale=20, translation=translationCatheter, rotation=anglesCathter)
    cavity.addObject('MeshTopology', src='@loader', name='topo')
    cavity.addObject('MechanicalObject', name='cavity')
    cavity.addObject('SurfacePressureConstraint', name='SurfacePressureConstraint', template='Vec3', value=0.0001, triangles='@topo.triangles', valueType='pressure')
    cavity.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)


    """ # Constraint
    cavity = catheter.addChild('cavity')
    translateFinger = [0, 25, -30]
    cavity.addObject('MeshSTLLoader', name='loader', filename='data/mesh/finger/pneunetCavityCut.stl', translation=translateFinger, rotation=[90, 0, 0])
    cavity.addObject('MeshTopology', src='@loader', name='topo')
    cavity.addObject('MechanicalObject', name='cavity')
    cavity.addObject('SurfacePressureConstraint', name='SurfacePressureConstraint', template='Vec3', value=0.0001, triangles='@topo.triangles', valueType='pressure')
    cavity.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False) """

    # Collision	
    collisionCatheter = catheter.addChild('collisionCatheter')
    collisionCatheter.addObject('MeshSTLLoader', name='loader', filename='data/mesh/cylinder_with_cavity.stl', scale=20, translation=translationCatheter, rotation=anglesCathter)
    collisionCatheter.addObject('MeshTopology', src='@loader', name='topo')
    collisionCatheter.addObject('MechanicalObject', name='collisMech')
    collisionCatheter.addObject('TriangleCollisionModel', selfCollision=False)
    collisionCatheter.addObject('LineCollisionModel', selfCollision=False)
    collisionCatheter.addObject('PointCollisionModel', selfCollision=False)
    collisionCatheter.addObject('BarycentricMapping')

    # Visualization	
    modelVisu = catheter.addChild('visu')
    modelVisu.addObject('MeshSTLLoader', name='loader', filename='data/mesh/cylinder_with_cavity.stl', scale = 20, translation=translationCatheter, rotation=anglesCathter)
    modelVisu.addObject('OglModel', src='@loader', color=[0.7, 0.7, 0.7, 0.6])
    modelVisu.addObject('BarycentricMapping')
    
    rootNode.addObject( PressureController(name="PressureController", node=rootNode, device_name="catheter", real_time=True) )
    print("\nadded PressureController\n")
