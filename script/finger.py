# runSofa -l /home/emanuele/Desktop/github/sim/sofa/build/v22.12/lib/libSofaPython3.so ./finger.py 

import Sofa
from pressureController import PressureController
import math


youngModulusFingers = 500
youngModulusStiffLayerFingers = 1500
translateFinger = [0, 0, 0]


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
    planeNode.addObject('MeshOBJLoader', name='plane_loader', filename='data/mesh/floorFlat.obj', triangulate=True, rotation=[0, 0, 270], scale=10, translation=[-122, 0, 0])
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
    cube.addObject('MechanicalObject', template='Rigid3', position=[-100, 70, 0, 0, 0, 0, 1])
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

    # Finger Model	
    finger = rootNode.addChild('finger')
    finger.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness=0.1, rayleighMass=0.1)
    finger.addObject('SparseLDLSolver', name='preconditioner')
    finger.addObject('MeshVTKLoader', name='loader', filename='data/mesh/finger/pneunetCutCoarse.vtk', rotation=[360, 0, 0], translation=translateFinger)
    finger.addObject('MeshTopology', src='@loader', name='container')
    finger.addObject('MechanicalObject', name='tetras', template='Vec3', showIndices=False, showIndicesScale=4e-5)
    finger.addObject('UniformMass', totalMass=0.04)
    finger.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large', poissonRatio=0.3, youngModulus=youngModulusFingers)
    finger.addObject('BoxROI', name='boxROI', box=[-10, 0, -20, 0, 30, 20], drawBoxes=True)
    finger.addObject('BoxROI', name='boxROISubTopo', box=[-100, 22.5, -8, -19, 28, 8], strict=False, drawBoxes=True)
    finger.addObject('RestShapeSpringsForceField', points='@boxROI.indices', stiffness=1e12, angularStiffness=1e12)
    finger.addObject('LinearSolverConstraintCorrection')

    # Sub topology	
    modelSubTopo = finger.addChild('modelSubTopo')
    modelSubTopo.addObject('TetrahedronSetTopologyContainer', position='@loader.position', tetrahedra='@boxROISubTopo.tetrahedraInROI', name='container')
    modelSubTopo.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large', poissonRatio=0.3, youngModulus=youngModulusStiffLayerFingers - youngModulusFingers)

    # Constraint
    cavity = finger.addChild('cavity')
    cavity.addObject('MeshSTLLoader', name='loader', filename='data/mesh/pneunetCavityCut.stl', translation=translateFinger, rotation=[360, 0, 0])
    cavity.addObject('MeshTopology', src='@loader', name='topo')
    cavity.addObject('MechanicalObject', name='cavity')
    cavity.addObject('SurfacePressureConstraint', name='SurfacePressureConstraint', template='Vec3', value=0.0001, triangles='@topo.triangles', valueType='pressure')
    cavity.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    # Collision
    collisionFinger = finger.addChild('collisionFinger')
    collisionFinger.addObject('MeshSTLLoader', name='loader', filename='data/mesh/finger/pneunetCut.stl', translation=translateFinger, rotation=[360, 0, 0])
    collisionFinger.addObject('MeshTopology', src='@loader', name='topo')
    collisionFinger.addObject('MechanicalObject', name='collisMech')
    collisionFinger.addObject('TriangleCollisionModel', selfCollision=False)
    collisionFinger.addObject('LineCollisionModel', selfCollision=False)
    collisionFinger.addObject('PointCollisionModel', selfCollision=False)
    collisionFinger.addObject('BarycentricMapping')

    # Visualization	
    modelVisu = finger.addChild('visu')
    modelVisu.addObject('MeshSTLLoader', name='loader', filename='data/mesh/finger/pneunetCut.stl')
    modelVisu.addObject('OglModel', src='@loader', color=[0.7, 0.7, 0.7, 0.6], translation=translateFinger, rotation=[360, 0, 0])
    modelVisu.addObject('BarycentricMapping')

    # Add controller
    rootNode.addObject( PressureController(name="PressureController", node=rootNode, device_name="finger") )
    print("\nadded PressureController\n")
