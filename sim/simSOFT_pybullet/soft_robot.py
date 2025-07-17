# Save this as soft_robot_that_actually_works.py
import pybullet as p
import pybullet_data
import time
import numpy as np
from scipy.spatial.transform import Rotation

# ===================================================================
# utils.py section (This part has been correct all along)
# ===================================================================
TOL = 1e-6

def vec_to_so3(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def so3_to_vec(so3):
    return np.array([so3[2, 1], so3[0, 2], so3[1, 0]])

def exp_so3(w):
    theta = np.linalg.norm(w)
    if theta < TOL: return np.identity(3)
    w_hat = vec_to_so3(w)
    a = np.sin(theta) / theta
    b = (1 - np.cos(theta)) / (theta**2)
    return np.identity(3) + a * w_hat + b * (w_hat @ w_hat)

def log_so3(R):
    cos_theta = np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if np.abs(theta) < TOL: return so3_to_vec(0.5 * (R - R.T))
    if np.abs(theta - np.pi) < TOL:
        eigvals, eigvecs = np.linalg.eigh(R)
        axis_idx = np.argmin(np.abs(eigvals - 1.0))
        return eigvecs[:, axis_idx] * np.pi
    w_hat = (theta / (2 * np.sin(theta))) * (R - R.T)
    return so3_to_vec(w_hat)

def exp_se3(twist):
    v, w = twist[:3], twist[3:]
    theta = np.linalg.norm(w)
    R = exp_so3(w)
    if theta < TOL: p = v
    else:
        w_hat = vec_to_so3(w)
        a = (1 - np.cos(theta)) / (theta**2)
        b = (theta - np.sin(theta)) / (theta**3)
        T = np.identity(3) + a * w_hat + b * (w_hat @ w_hat)
        p = T @ v
    H = np.identity(4)
    H[:3, :3] = R
    H[:3, 3] = p
    return H

def log_se3(H):
    R, p = H[:3, :3], H[:3, 3]
    w = log_so3(R)
    theta = np.linalg.norm(w)
    if theta < TOL: v = p
    else:
        w_hat = vec_to_so3(w)
        T_inv = np.identity(3) - 0.5 * w_hat + \
            (1/theta**2 - (1 + np.cos(theta))/(2 * theta * np.sin(theta))) * (w_hat @ w_hat)
        v = T_inv @ p
    return np.concatenate([v, w])

def get_H(pos, quat):
    H = np.identity(4)
    H[:3, :3] = Rotation.from_quat(quat).as_matrix()
    H[:3, 3] = pos
    return H

def adjoint_se3(H):
    R = H[:3, :3]
    p = H[:3, 3]
    p_hat = vec_to_so3(p)
    Adj = np.zeros((6, 6))
    Adj[:3, :3] = R
    Adj[3:, 3:] = R
    Adj[:3, 3:] = p_hat @ R
    return Adj

# ===================================================================
# The Main Arm Class and Simulation
# ===================================================================

class SoftContinuumArm:
    def __init__(self, base_pos, base_orn, num_segments, segment_length, segment_radius, K_stiffness, D_damping):
        self.num_segments = num_segments
        self.segment_length = segment_length
        self.robot_id = self._create_arm_body(base_pos, base_orn, num_segments, segment_length, segment_radius)
        self.K = K_stiffness
        self.D = D_damping
        self.actuation_wrenches = [np.zeros(6) for _ in range(num_segments)]

    def _create_arm_body(self, base_pos, base_orn, num_segments, segment_length, segment_radius):
        segment_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=segment_radius, height=segment_length)
        segment_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=segment_radius, length=segment_length, rgbaColor=[0.8, 0.2, 0.2, 1])
        link_Masses = [0.1] * num_segments
        
        # This defines the position of the CHILD joint frame relative to the PARENT joint frame.
        # For a simple serial chain, this is a fixed translation along an axis.
        linkPositions = [[0, 0, segment_length]] * num_segments
        linkOrientations = [[0, 0, 0, 1]] * num_segments
        linkInertialFramePositions = [[0, 0, 0]] * num_segments # CoM is at the joint frame origin
        
        return p.createMultiBody(
            baseMass=0, basePosition=base_pos, baseOrientation=base_orn,
            linkMasses=link_Masses,
            linkCollisionShapeIndices=[segment_shape] * num_segments,
            linkVisualShapeIndices=[segment_visual] * num_segments,
            linkPositions=linkPositions,
            linkOrientations=linkOrientations,
            linkInertialFramePositions=linkInertialFramePositions,
            linkInertialFrameOrientations=linkOrientations,
            linkParentIndices=list(range(num_segments)),
            linkJointTypes=[p.JOINT_FIXED] * num_segments,
            linkJointAxis=[[0, 0, 1]] * num_segments
        )

    def set_actuation(self, segment_index, wrench):
        if 0 <= segment_index < self.num_segments:
            self.actuation_wrenches[segment_index] = wrench

    def step(self):
        for i in range(self.num_segments):
            child_link_idx = i
            parent_link_idx = i - 1
            
            if parent_link_idx == -1:
                H_parent = get_H(*p.getBasePositionAndOrientation(self.robot_id))
                lin_vel_parent, ang_vel_parent = p.getBaseVelocity(self.robot_id)
            else:
                state_parent = p.getLinkState(self.robot_id, parent_link_idx, computeLinkVelocity=1)
                H_parent = get_H(state_parent[0], state_parent[1]) # CORRECT: Use CoM world pos/orn for H
                lin_vel_parent, ang_vel_parent = state_parent[6], state_parent[7]
            
            state_child = p.getLinkState(self.robot_id, child_link_idx, computeLinkVelocity=1)
            H_child = get_H(state_child[0], state_child[1]) # CORRECT: Use CoM world pos/orn for H
            lin_vel_child, ang_vel_child = state_child[6], state_child[7]

            H_rest = np.identity(4)
            H_rest[2, 3] = self.segment_length
            H_rel_current = np.linalg.inv(H_parent) @ H_child
            H_deformation = np.linalg.inv(H_rest) @ H_rel_current
            d_j = log_se3(H_deformation)
            
            Adj_H_parent_inv = np.linalg.inv(adjoint_se3(H_parent))
            twist_parent_body = Adj_H_parent_inv @ np.concatenate([lin_vel_parent, ang_vel_parent])
            twist_child_body = Adj_H_parent_inv @ np.concatenate([lin_vel_child, ang_vel_child])
            d_dot_j = twist_child_body - twist_parent_body

            internal_wrench = -self.K @ d_j - self.D @ d_dot_j
            total_wrench_on_child_local = internal_wrench + self.actuation_wrenches[i]
            
            R_parent_to_world = H_parent[:3, :3]
            force_child_world = R_parent_to_world @ total_wrench_on_child_local[:3]
            torque_child_world = R_parent_to_world @ total_wrench_on_child_local[3:]
            p.applyExternalForce(self.robot_id, child_link_idx, force_child_world, state_child[0], p.WORLD_FRAME)
            p.applyExternalTorque(self.robot_id, child_link_idx, torque_child_world, p.WORLD_FRAME)

            reaction_wrench_on_parent_local = -internal_wrench
            if parent_link_idx != -1:
                force_parent_world = R_parent_to_world @ reaction_wrench_on_parent_local[:3]
                torque_parent_world = R_parent_to_world @ reaction_wrench_on_parent_local[3:]
                p.applyExternalForce(self.robot_id, parent_link_idx, force_parent_world, state_parent[0], p.WORLD_FRAME)
                p.applyExternalTorque(self.robot_id, parent_link_idx, torque_parent_world, p.WORLD_FRAME)

if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    planeId = p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=25, cameraPitch=-35, cameraTargetPosition=[0,0,1.0])

    NUM_SEGMENTS = 10
    SEGMENT_LENGTH = 0.1
    
    k_bend_slider = p.addUserDebugParameter("Bending Stiffness", 0, 50, 5)
    d_ratio_slider = p.addUserDebugParameter("Damping Ratio", 0, 0.2, 0.05)
    torque_slider = p.addUserDebugParameter("Tip Torque (Y-axis)", -1.0, 1.0, 0.0)

    K_base = np.diag([5000, 5000, 5000, 5.0, 5.0, 5.0])
    
    base_orientation_quat = p.getQuaternionFromEuler([0, -np.pi / 2, 0])
    base_position = [0, 0, 1.5]

    arm = SoftContinuumArm(
        base_pos=base_position,
        base_orn=base_orientation_quat,
        num_segments=NUM_SEGMENTS,
        segment_length=SEGMENT_LENGTH,
        segment_radius=0.02,
        K_stiffness=K_base,
        D_damping=K_base * 0.05
    )

    while p.isConnected():
        try:
            k_val = p.readUserDebugParameter(k_bend_slider)
            d_ratio = p.readUserDebugParameter(d_ratio_slider)
            torque_val = p.readUserDebugParameter(torque_slider)

            arm.K[3,3] = arm.K[4,4] = arm.K[5,5] = k_val
            arm.D = arm.K * d_ratio
            
            for i in range(NUM_SEGMENTS):
                arm.set_actuation(i, np.zeros(6))

            actuation_wrench = np.array([0, 0, 0, 0, torque_val, 0])
            arm.set_actuation(NUM_SEGMENTS - 1, actuation_wrench)
            
            arm.step()
            p.stepSimulation()
            time.sleep(1./240.)
        except p.error:
            break
            
    p.disconnect()