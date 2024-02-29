def add_cube(rootNode, position):
        cube = rootNode.addChild('cube')
        cube.addObject('EulerImplicitSolver', name='odesolver')
        cube.addObject('SparseLDLSolver', name='linearSolver')
        cube.addObject('MechanicalObject', template='Rigid3', position=position)
        cube.addObject('UniformMass', totalMass=0.001)
        cube.addObject('UncoupledConstraintCorrection')

        cubeScale = 5
        cubeCollis = cube.addChild('cubeCollis')
        cubeCollis.addObject('MeshOBJLoader', name='cube_loader', filename='data/mesh/smCube27.obj', triangulate=True, scale=cubeScale)
        cubeCollis.addObject('MeshTopology', src='@cube_loader')
        cubeCollis.addObject('MechanicalObject')
        cubeCollis.addObject('TriangleCollisionModel')
        cubeCollis.addObject('LineCollisionModel')
        cubeCollis.addObject('PointCollisionModel')
        cubeCollis.addObject('RigidMapping')

        cubeVisu = cube.addChild('cubeVisu')
        cubeVisu.addObject('MeshOBJLoader', name='cube_loader', filename='data/mesh/smCube27.obj')
        cubeVisu.addObject('OglModel', name='Visual', src='@cube_loader', color=[0.0, 0.1, 0.5], scale=cubeScale)
        cubeVisu.addObject('RigidMapping')


def add_floor(rootNode, translation, rotation):
    planeNode = rootNode.addChild('Plane')
    planeNode.addObject('MeshOBJLoader', name='plane_loader', filename='data/mesh/floorFlat.obj', triangulate=True, rotation=rotation, scale=10, translation=translation)
    planeNode.addObject('MeshTopology', src='@plane_loader')
    planeNode.addObject('MechanicalObject', src='@plane_loader')
    planeNode.addObject('TriangleCollisionModel', simulated=False, moving=False)
    planeNode.addObject('LineCollisionModel', simulated=False, moving=False)
    planeNode.addObject('PointCollisionModel', simulated=False, moving=False)
    planeNode.addObject('OglModel', name='Visual_plane', src='@plane_loader', color=[1, 0, 0, 1])
