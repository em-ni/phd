import numpy as np
from scipy.spatial.transform import Rotation

# Tolerance for floating point comparisons
TOL = 1e-6

def vec_to_so3(w):
    """
    Converts a 3D angular velocity vector to a 3x3 skew-symmetric matrix.
    This is the 'hat' operator.
    Args:
        w (np.ndarray): 3D vector.
    Returns:
        np.ndarray: 3x3 skew-symmetric matrix.
    """
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def so3_to_vec(so3):
    """
    Converts a 3x3 skew-symmetric matrix to a 3D angular velocity vector.
    This is the 'vee' operator.
    Args:
        so3 (np.ndarray): 3x3 skew-symmetric matrix.
    Returns:
        np.ndarray: 3D vector.
    """
    return np.array([so3[2, 1], so3[0, 2], so3[1, 0]])

def exp_so3(w):
    """
    Computes the exponential map from so(3) to SO(3) using Rodrigues' formula.
    Args:
        w (np.ndarray): 3D angular velocity vector.
    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    theta = np.linalg.norm(w)
    if theta < TOL:
        return np.identity(3)
    
    w_hat = vec_to_so3(w)
    w_hat_sq = w_hat @ w_hat
    
    # Rodrigues' formula from thesis Eq. 1.19
    a = np.sin(theta) / theta
    b = (1 - np.cos(theta)) / (theta**2)
    
    return np.identity(3) + a * w_hat + b * w_hat_sq

def log_so3(R):
    """
    Computes the logarithmic map from SO(3) to so(3).
    Args:
        R (np.ndarray): 3x3 rotation matrix.
    Returns:
        np.ndarray: 3D angular velocity vector.
    """
    if not isinstance(R, np.ndarray) or R.shape != (3, 3):
        raise ValueError("Input must be a 3x3 numpy array")

    # Thesis Eq. 1.25
    cos_theta = (np.trace(R) - 1) / 2.0
    
    # Clip to handle numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    if np.abs(theta) < TOL:
        # For small angles, use the approximation
        return so3_to_vec(0.5 * (R - R.T))
    
    if np.abs(theta - np.pi) < TOL:
        # Handle the 180-degree rotation case
        # Find the axis of rotation
        eigvals, eigvecs = np.linalg.eigh(R)
        axis_idx = np.argmin(np.abs(eigvals - 1.0))
        w = eigvecs[:, axis_idx]
        return w * np.pi

    # Thesis Eq. 1.24
    w_hat = (theta / (2 * np.sin(theta))) * (R - R.T)
    return so3_to_vec(w_hat)


def exp_se3(twist):
    """
    Computes the exponential map from se(3) to SE(3).
    Args:
        twist (np.ndarray): 6D twist vector [v, w].
    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix.
    """
    v = twist[:3]
    w = twist[3:]
    theta = np.linalg.norm(w)
    
    R = exp_so3(w)
    
    if theta < TOL:
        p = v
    else:
        w_hat = vec_to_so3(w)
        w_hat_sq = w_hat @ w_hat
        
        # Tangent operator T from thesis appendix B
        a = (1 - np.cos(theta)) / (theta**2)
        b = (theta - np.sin(theta)) / (theta**3)
        
        T_inv = np.identity(3) + 0.5 * w_hat + ((1/(theta**2)) - ((1+np.cos(theta))/(2*theta*np.sin(theta)))) * w_hat_sq
        p = (np.identity(3) + a * w_hat + b * w_hat_sq) @ v

    H = np.identity(4)
    H[:3, :3] = R
    H[:3, 3] = p
    return H

def log_se3(H):
    """
    Computes the logarithmic map from SE(3) to se(3).
    Args:
        H (np.ndarray): 4x4 homogeneous transformation matrix.
    Returns:
        np.ndarray: 6D twist vector [v, w].
    """
    R = H[:3, :3]
    p = H[:3, 3]
    
    w = log_so3(R)
    theta = np.linalg.norm(w)
    
    if theta < TOL:
        v = p
    else:
        w_hat = vec_to_so3(w)
        w_hat_sq = w_hat @ w_hat
        
        # Inverse tangent operator T^-1 from thesis appendix B
        # T^-1 = I - 1/2 * w_hat + (1/theta^2 - (1+cos(theta))/(2*theta*sin(theta))) * w_hat^2
        # A simpler form is often used: T = I + (1-cos)/th^2 * w_hat + (th-sin)/th^3 * w_hat^2
        # And v = T_inv @ p
        
        # Using a stable implementation of the inverse tangent map
        T_inv = np.identity(3) - 0.5 * w_hat + (1 / theta**2 - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))) * w_hat_sq
        v = T_inv @ p
        
    return np.concatenate([v, w])

def get_H(pos, quat):
    """Converts PyBullet position and quaternion to a 4x4 SE(3) matrix."""
    H = np.identity(4)
    H[:3, :3] = Rotation.from_quat(quat).as_matrix()
    H[:3, 3] = pos
    return H

def get_pos_quat(H):
    """Converts a 4x4 SE(3) matrix to PyBullet position and quaternion."""
    pos = H[:3, 3]
    quat = Rotation.from_matrix(H[:3, :3]).as_quat()
    return pos, quat
    
def adjoint_se3(H):
    """Computes the Adjoint representation of an SE(3) matrix."""
    R = H[:3, :3]
    p = H[:3, 3]
    p_hat = vec_to_so3(p)
    
    Adj = np.zeros((6, 6))
    Adj[:3, :3] = R
    Adj[3:, 3:] = R
    Adj[:3, 3:] = p_hat @ R
    return Adj