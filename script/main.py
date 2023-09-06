# runSofa -l /home/emanuele/Desktop/github/sim/sofa/build/v22.12/lib/libSofaPython3.so ./main.py 

import Sofa
from emptyController import EmptyController


def createScene(rootNode):
    rootNode.addObject('VisualStyle', displayFlags='showForceFields showBehaviorModels')
    rootNode.addObject('RequiredPlugin', pluginName='SoftRobots SofaPython3')
    rootNode.gravity.value = [-9810, 0, 0]
    rootNode.addObject('AttachBodyButtonSetting', stiffness=10)
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('GenericConstraintSolver', tolerance=1e-12, maxIterations=10000)

    catheter = rootNode.addChild('catheter')
    catheter.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness=0.1, rayleighMass=0.1)
    catheter.addObject('SparseLDLSolver', name='directSolver')
    catheter.addObject('MeshVTKLoader', name='loader', filename='data/mesh/1dof_catheter.vtk')
    catheter.addObject('MeshTopology', src='@loader', name='container')
    catheter.addObject('MechanicalObject', name='tetras', template='Vec3', showObject=True, showObjectScale=1)
    #catheter.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large', poissonRatio=0.3, youngModulus=500)
    catheter.addObject('UniformMass', totalMass=0.0008)
    catheter.addObject('BoxROI', name='boxROISubTopo', box=[-100, 22.5, -8, -19, 28, 8], strict=False)
    catheter.addObject('BoxROI', name='boxROI', box=[-10, 0, -20, 0, 30, 20], drawBoxes=True)
    catheter.addObject('RestShapeSpringsForceField', points='@boxROI.indices', stiffness=1e12, angularStiffness=1e12)
    catheter.addObject('LinearSolverConstraintCorrection')

    modelSubTopo = catheter.addChild('modelSubTopo')
    modelSubTopo.addObject('MeshTopology', position='@loader.position', tetrahedra='@boxROISubTopo.tetrahedraInROI', name='container')
    #modelSubTopo.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large', poissonRatio=0.3, youngModulus=1500)

    cavity = catheter.addChild('cavity')
    cavity.addObject('MeshSTLLoader', name='cavityLoader', filename='data/mesh/1dof_catheter.STL')
    cavity.addObject('MeshTopology', src='@cavityLoader', name='cavityMesh')
    cavity.addObject('MechanicalObject', name='cavity')
    cavity.addObject('SurfacePressureConstraint', name='SurfacePressureConstraint', template='Vec3', value=1, triangles='@cavityMesh.triangles', valueType='pressure')
    cavity.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    rootNode.addObject(EmptyController(rootNode))
    print("\nadded EmptyController\n")

"""     # Importing sofagym and other necessary libraries
    import sofagym
    #import numpy as np

    # Creating an environment
    env = sofagym.make("Pendulum-v0")

    # Define a simple policy function
    def policy(observation, t):
        angle = observation[0]
        return [np.cos(t / 10.), np.sin(t / 10.)]

    # Reset the environment
    observation = env.reset()

    # Start the main loop
    for t in range(1000):
        env.render()
        action = policy(observation, t) # get action from the policy
        observation, reward, done, info = env.step(action) # apply the action

        # If the episode is done, reset the environment
        if done:
            observation = env.reset()

    env.close()

    # Explanation:
    # We first import the necessary libraries and create the environment.
    # We then define a simple policy function that determines the action based on the current observation and time step.
    # We reset the environment to get the initial observation.
    # In the main loop, we render the environment, get the action from the policy, and apply the action using the step function.
    # The step function returns the new observation, the reward, whether the episode is done, and additional info.
    # If the episode is done (e.g., the pendulum has fallen), we reset the environment.
    # Finally, we close the environment. """
    
