# simsoft_python/lie_group_utils.py

import numpy as np
from scipy.linalg import expm, logm

def skew(v: np.ndarray) -> np.ndarray:
    """
    Converts a 3D vector to a 3x3 skew-symmetric matrix.
    Thesis Equation: 1.9
    """
    v = np.asarray(v).flatten()
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def unskew(S: np.ndarray) -> np.ndarray:
    """Converts a 3x3 skew-symmetric matrix back to a 3D vector."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

def exp_so3(w: np.ndarray) -> np.ndarray:
    """
    Exponential map for SO(3) using Rodrigues' formula.
    Thesis Equation: 1.19
    """
    w = np.asarray(w).flatten()
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.identity(3)
    
    w_skew = skew(w)
    a = np.sin(theta) / theta
    b = (1 - np.cos(theta)) / (theta**2)
    
    return np.identity(3) + a * w_skew + b * (w_skew @ w_skew)

def exp_se3(xi: np.ndarray) -> np.ndarray:
    """
    Exponential map for SE(3).
    Thesis Equation: B.18
    xi is a 6x1 twist vector [v, w]
    """
    xi = np.asarray(xi).flatten()
    v = xi[:3]
    w = xi[3:]
    
    w_skew = skew(w)
    theta = np.linalg.norm(w)
    
    if theta < 1e-12:
        R = np.identity(3)
        T_mat = np.identity(3)
    else:
        R = exp_so3(w)
        a = (1 - np.cos(theta)) / (theta**2)
        b = (theta - np.sin(theta)) / (theta**3)
        T_mat = np.identity(3) + a * w_skew + b * (w_skew @ w_skew)

    u = T_mat @ v.reshape(3, 1)
    
    H = np.identity(4)
    H[:3, :3] = R
    H[:3, 3] = u.flatten()
    return H

def adjoint(H: np.ndarray) -> np.ndarray:
    """
    Adjoint representation of a homogeneous transformation matrix H.
    Thesis Equation: A.25 (in matrix form)
    """
    R = H[:3, :3]
    p = H[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[:3, 3:] = skew(p) @ R
    return Ad

def ad_se3(xi: np.ndarray) -> np.ndarray:
    """
    The 6x6 matrix representation of the Lie bracket operator 'ad' for se(3).
    For twists eta_1, eta_2, the bracket [eta_1, eta_2] is ad_se3(eta_1) @ eta_2.
    Appears as eta_hat in Thesis Equation: 4.8
    """
    xi = np.asarray(xi).flatten()
    v = xi[:3]
    w = xi[3:]
    w_skew = skew(w)
    v_skew = skew(v)
    
    ad_mat = np.zeros((6, 6))
    ad_mat[:3, :3] = w_skew
    ad_mat[:3, 3:] = v_skew
    ad_mat[3:, 3:] = w_skew
    return ad_mat