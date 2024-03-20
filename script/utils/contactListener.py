#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Sofa
import Sofa.Core
from Sofa.constants import *
import numpy as np
import re
from collections import defaultdict
import socket
import json
import time
import threading

class ContactListener(Sofa.Core.Controller):
    def __init__(self, *a, **kw): # arga, kwargs
        Sofa.Core.Controller.__init__(self, *a, **kw)
        self.root_node = kw['node']
        self.object_name = kw['object_name']
        self.collision_name = kw['collision_name']
        self.debug = kw['debug']
        self.plot = kw['plot']
        self.fx = 0
        self.fy = 0
        self.fz = 0
        
        # Add a drawNode to visualize the forces applied by the constraints.
        self.add_drawNode()
        
        if self.plot:
            thread = threading.Thread(target=self.sendData)
            thread.start()
        
    def add_drawNode(self):
        drawNode = self.root_node.addChild('drawNode')
        MOdraw = drawNode.addObject('MechanicalObject', name="drawPositions", position="@/root_node/{self.object_name}/{self.collision_name}/MechanicalObject.position", size=8) 
        drawNode.addObject('ConstantForceField', name="drawForceFF", indices=[0], forces=[0, 0, 0], showArrowSize=1)

    def onAnimateEndEvent(self, event):
        # This function is called at each end of animation step.
        if self.debug : print("\nContactListener: onAnimateEndEvent")
        # Retrieve the constraint value from the MechanicalObject of the collision model.
        # This contains information about the constraints applied during the simulation step.
        constraint = self.root_node.getChild(self.object_name).getChild(self.collision_name).MechanicalObject.constraint.value
        # Parse constraint data (needed to pass from v22.12 to v23.12)
        constraint = parse_constraint_data(constraint)
        if self.debug : print(f"ContactListener: Constraint: {constraint}")
        if self.debug : print(f"ContactListener: type(constraint): {type(constraint)}, len(constraint): {len(constraint)}")
        # Get the simulation time step value for force computation.
        dt = self.root_node.dt.value

        # Convert the constraint string into a numpy array for easier manipulation.
        constraintMatrixInline = np.fromstring(constraint, sep=' ')
        if self.debug : print(f"ContactListener: constraintMatrixInline: {constraintMatrixInline}")


        # Initialize variables to store point IDs and constraint directions.
        pointId = []
        constraintId = []
        constraintDirections = []
        index = 0
        i = 0

        # Retrieve the norm of forces applied by the constraints from the solver.
        forcesNorm = self.root_node.GenericConstraintSolver.constraintForces.value
        contactforce_x = 0
        contactforce_y = 0
        contactforce_z = 0

        # Prepare an array to store constraint directions, initialized to zeros.
        #if len(constraintMatrixInline) > 0:
            #constraintDirections = np.zeros((int(len(constraintMatrixInline)/6), 3))

        # Loop through the constraint matrix to extract constraint information.
        while index < len(constraintMatrixInline):
            if self.debug : print(f"ContactListener: index: {index}")
            nbConstraint   = int(constraintMatrixInline[index+1])
            currConstraintID = int(constraintMatrixInline[index])
            if self.debug : print(f"ContactListener: nbConstraint: {nbConstraint}, currConstraintID: {currConstraintID}")
            for pts in range(nbConstraint):
                currIDX = index+2+pts*4
                pointId = np.append(pointId, constraintMatrixInline[currIDX])
                constraintId.append(currConstraintID)
                constraintDirections.append([constraintMatrixInline[currIDX+1],constraintMatrixInline[currIDX+2],constraintMatrixInline[currIDX+3]])
                if self.debug : print(f"ContactListener: pointId: {pointId}, constraintId: {constraintId}, constraintDirections: {constraintDirections}")
            index = index + 2 + nbConstraint*4

        # Debugging print to check the extracted constraint directions.
        if self.debug : print(f"\nContactListener: Constraint Directions: {constraintDirections}")

        # Determine the number of degrees of freedom (DOFs) for the object.
        nbDofs = len(self.root_node.getChild(self.object_name).getChild(self.collision_name).MechanicalObject.position.value)
        # Initialize a matrix to store the computed forces for each DOF.
        forces = np.zeros((nbDofs, 3))

        # After retrieving forcesNorm
        if self.debug: print(f"ContactListener: forcesNorm size: {len(forcesNorm)}")

        for i in range(len(pointId)):
            indice = int(pointId[i])
            
            # Additional debug to ensure indices are within bounds
            if i < len(constraintDirections) and constraintId[i] < len(forcesNorm):
                if indice < nbDofs:
                    forces[indice][0] += constraintDirections[i][0] * forcesNorm[constraintId[i]] / dt
                    forces[indice][1] += constraintDirections[i][1] * forcesNorm[constraintId[i]] / dt
                    forces[indice][2] += constraintDirections[i][2] * forcesNorm[constraintId[i]] / dt
                else:
                    if self.debug: print(f"ContactListener: Warning: indice {indice} out of bounds for forces array with size {nbDofs}")
            else:
                if self.debug: print(f"ContactListener: Warning: Access out of bounds. i: {i}, constraintId[i]: {constraintId[i]}, len(constraintDirections): {len(constraintDirections)}, len(forcesNorm): {len(forcesNorm)}")

        for i in range (nbDofs):
            contactforce_x += forces[i][0]
            contactforce_y += forces[i][1]
            contactforce_z += forces[i][2]
            
        # Assign force values
        self.fx = contactforce_x
        self.fy = contactforce_y
        self.fz = contactforce_z
      
        if self.debug : print('\nContactListener: nbDof: ', nbDofs)
        # TODO something's wrong: [ERROR]   [ConstantForceField(drawForceFF)] Size mismatch: indices > system size
        # Visualize the computed forces in the simulation.
        if len(constraintMatrixInline) > 0:
            if self.debug : print(f"ContactListener: indices: {list(range(0,nbDofs,1))}")
            if self.debug : print(f"ContactListener: forces: {forces}")
            if self.debug : print(f"ContactListener: position: {self.root_node.getChild(self.object_name).getChild(self.collision_name).MechanicalObject.position.value}")
            #self.root_node.drawNode.drawForceFF.indices.value = list(range(0,nbDofs,1)) 
            #self.root_node.drawNode.drawForceFF.forces.value = forces
            #self.root_node.drawNode.drawPositions.position.value = self.root_node.getChild(self.object_name).getChild(self.collision_name).MechanicalObject.position.value

        if self.debug : print('\nContactListener: contactforce: ', contactforce_x, contactforce_y, contactforce_z)
        
    
    def sendData(self):
        print("Trying to connect")
        host = '127.0.0.1'  # The server's hostname or IP address
        port = 65432        # The port used by the server

        while True:  # Outer loop for reconnection attempts
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((host, port))
                    print("Connection established")
                    try:
                        while True:  # Inner loop for continuously sending data
                            #data = {"fx": self.fx, "fy": self.fy, "fz": self.fz}
                            data = {"fz": self.fz}
                            s.sendall(json.dumps(data).encode('utf-8'))
                            if self.debug: print(f"Sent data: {data}")
                            time.sleep(0.1)
                    except Exception as e:
                        print(f"Error during data transmission: {e}")
                        print("Attempting to reconnect...")
                        time.sleep(1)  # Wait a bit before trying to reconnect
            except Exception as e:
                print(f"Connection failed: {e}")
                print("Retrying to connect...")
                time.sleep(1) 


        
def parse_constraint_data(constraint_str):
    # Regular expression to extract constraint data
    pattern = re.compile(r"Constraint ID : (\d+) +dof ID : (\d+) +value : ([\-\d\.e]+) ([\-\d\.e]+) ([\-\d\.e]+)")
    matches = pattern.findall(constraint_str)

    # Group data by Constraint ID
    constraints = defaultdict(list)
    for match in matches:
        constraint_id, dof_id, *values = match
        constraints[int(constraint_id)].append((int(dof_id), [float(val) for val in values]))

    # Sort and format the data
    formatted_constraints = []
    for constraint_id, dofs in sorted(constraints.items()):
        dofs.sort(key=lambda x: x[0])  # Sort by dof ID, if necessary
        formatted_constraint = f"{constraint_id} {len(dofs)}"
        for dof_id, values in dofs:
            formatted_constraint += f" {dof_id} {' '.join(map(str, values))}"
        formatted_constraints.append(formatted_constraint)

    return "\n".join(formatted_constraints)







