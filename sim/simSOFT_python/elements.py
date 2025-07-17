# simsoft_python/elements.py

import numpy as np
from typing import List
from scipy.integrate import quad_vec
from lie_group_utils import exp_se3, adjoint, ad_se3

class PiecewiseConstantDeformationArm:
    """
    Implements the Piecewise Constant Deformation (PWCD) Soft Arm model.
    This class handles 'n' segments, each with its own properties.
    The state is y = [epsilon_1, ..., epsilon_n, epsilon_dot_1, ..., epsilon_dot_n],
    where each epsilon is a 6x1 strain vector. Total state size is 12n.

    Core Thesis Concepts:
    - Kinematics: Chapter 2 (Eq. 2.15)
    - Geometric Jacobian: Chapter 3 (Table 3.1)
    - Dynamic Model: Chapter 4 (Eq. 4.33)
    """

    def __init__(self,
                 lengths: List[float],
                 M_sections: List[np.ndarray],
                 K_sections: List[np.ndarray],
                 D_sections: List[np.ndarray], # Damping matrix for dynamic relaxation
                 H_base: np.ndarray = np.identity(4)):
        
        self.n_segments = len(lengths)
        assert self.n_segments == len(M_sections) == len(K_sections) == len(D_sections), \
            "Input lists for segments must have the same length."

        self.lengths = np.array(lengths)
        self.M_sections = M_sections
        self.K_sections = K_sections
        self.D_sections = D_sections # Viscous damping term
        self.H_base = H_base

        self.total_length = np.sum(self.lengths)
        self.L_cumulative = np.hstack(([0], np.cumsum(self.lengths)))

    def get_strains(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extracts epsilon and epsilon_dot from the state vector y."""
        epsilons = y[:6 * self.n_segments].reshape((self.n_segments, 6, 1))
        epsilon_dots = y[6 * self.n_segments:].reshape((self.n_segments, 6, 1))
        return epsilons, epsilon_dots

    def forward_kinematics(self, epsilons: np.ndarray, alpha: float) -> np.ndarray:
        """
        Computes the pose H(alpha) using the Product of Exponentials formula.
        Thesis Equation: 2.15
        - epsilons: (n, 6, 1) array of strain vectors.
        - alpha: position along the arm's total length (0 to total_length).
        """
        H = self.H_base.copy()
        
        # Find which segment alpha lies in
        segment_idx = np.searchsorted(self.L_cumulative, alpha, side='right') - 1
        segment_idx = max(0, segment_idx)

        # Apply transformations for all preceding segments
        for i in range(segment_idx):
            d_i = epsilons[i] * self.lengths[i]
            H = H @ exp_se3(d_i)

        # Apply partial transformation for the current segment
        if alpha > self.L_cumulative[segment_idx]:
            local_alpha = alpha - self.L_cumulative[segment_idx]
            d_segment = epsilons[segment_idx] * self.lengths[segment_idx]
            H = H @ exp_se3(d_segment * (local_alpha / self.lengths[segment_idx]))
            
        return H

    def _compute_jacobian_at_alpha(self, epsilons: np.ndarray, alpha: float) -> np.ndarray:
        """
        Computes the full 6n x 6n geometric Jacobian J(alpha) that maps the stacked
        strain rates [e_dot_1, ..., e_dot_n] to the velocity twist at point alpha.
        eta(alpha) = J(alpha) @ [e_dot_1, ..., e_dot_n]^T
        Follows the structure of Table 3.1 in the thesis.
        """
        J_alpha = np.zeros((6, 6 * self.n_segments))
        
        segment_idx = np.searchsorted(self.L_cumulative, alpha, side='right') - 1
        segment_idx = np.clip(segment_idx, 0, self.n_segments - 1)

        # Loop through strain rates epsilon_dot_j
        for j in range(self.n_segments):
            d_j = epsilons[j] * self.lengths[j]
            
            # --- Contribution of e_dot_j to eta(alpha) ---
            if j > segment_idx:
                # e_dot_j has no effect on eta(alpha) if j is after the current segment
                continue

            # This is the J_j block from thesis Eq 3.58.
            # J_j(alpha_j, d_j)
            local_alpha_j = self.lengths[j] if j < segment_idx else alpha - self.L_cumulative[j]
            d_se3_j = ad_se3(d_j / self.lengths[j])
            
            # Simplified but structurally correct J_block from single element theory
            exp_d_lj = exp_se3(d_j * (local_alpha_j / self.lengths[j]))
            J_block = self.lengths[j] * np.linalg.inv(adjoint(exp_d_lj))
            
            # Propagation Adjoint term: Ad( H_alpha_to_H_j_end )
            H_prop = np.identity(4)
            for k in range(j + 1, segment_idx + 1):
                d_k = epsilons[k] * self.lengths[k]
                H_prop = H_prop @ exp_se3(d_k)
            
            # Adjust for partial segment at the end
            if segment_idx == j:
                Ad_term = np.identity(6)
            else:
                d_seg = epsilons[segment_idx] * self.lengths[segment_idx]
                local_alpha_seg = alpha - self.L_cumulative[segment_idx]
                H_partial = exp_se3(d_seg * (local_alpha_seg / self.lengths[segment_idx]))
                Ad_term = adjoint(np.linalg.inv(H_partial)) @ adjoint(np.linalg.inv(H_prop))

            J_alpha[:, 6*j:6*(j+1)] = Ad_term @ J_block
            
        return J_alpha

    def compute_system_matrices(self, epsilons: np.ndarray, epsilon_dots: np.ndarray):
        """
        Computes the full system matrices M_sys, C_sys, K_sys, D_sys.
        Thesis Equation: 4.33 (structure)
        """
        epsilons_flat = epsilons.flatten()
        epsilon_dots_flat = epsilon_dots.flatten()
        
        integrand = lambda alpha: self.integrand_for_matrices(epsilons, epsilon_dots_flat, alpha)
        
        # quad_vec integrates a vector-returning function. We flatten the matrices.
        result, _ = quad_vec(integrand, 0, self.total_length)
        
        size = 6 * self.n_segments
        M_sys = result[0 : size**2].reshape((size, size))
        K_sys = result[size**2 : 2*size**2].reshape((size, size))
        D_sys = result[2*size**2 : 3*size**2].reshape((size, size))
        C_sys = result[3*size**2 : 4*size**2].reshape((size, size))

        return M_sys, C_sys, K_sys, D_sys

    def integrand_for_matrices(self, epsilons: np.ndarray, epsilon_dots_flat: np.ndarray, alpha: float) -> np.ndarray:
        """Helper function to compute all matrix integrands at a point alpha."""
        J_alpha = self._compute_jacobian_at_alpha(epsilons, alpha)
        eta_alpha = J_alpha @ epsilon_dots_flat
        
        segment_idx = np.searchsorted(self.L_cumulative, alpha, side='right') - 1
        segment_idx = max(0, segment_idx)
        
        M_sec = self.M_sections[segment_idx]
        K_sec = self.K_sections[segment_idx]
        D_sec = self.D_sections[segment_idx]

        m_integrand = J_alpha.T @ M_sec @ J_alpha
        k_integrand = J_alpha.T @ K_sec @ J_alpha
        d_integrand = J_alpha.T @ D_sec @ J_alpha
        # Coriolis term: J^T * ad(eta)^T * M * J
        c_integrand = J_alpha.T @ ad_se3(eta_alpha).T @ M_sec @ J_alpha
        
        return np.hstack([m.flatten() for m in [m_integrand, k_integrand, d_integrand, c_integrand]])
        
    def compute_generalized_forces(self, epsilons: np.ndarray, F_ext_list) -> np.ndarray:
        """
        Computes the generalized force vector from a list of external wrenches.
        F_gen = Sum( J(alpha_i)^T @ F_ext_i )
        F_ext_list is a list of tuples: (alpha, F_wrench)
        """
        F_gen = np.zeros(6 * self.n_segments)
        for alpha, F_wrench in F_ext_list:
            J_alpha = self._compute_jacobian_at_alpha(epsilons, alpha)
            F_gen += J_alpha.T @ F_wrench
        return F_gen

    def state_derivative(self, t: float, y: np.ndarray, F_ext_func) -> np.ndarray:
        """
        The function f(t, y) for the ODE solver, where y = [epsilons, epsilon_dots].
        It computes dy/dt = [epsilon_dots, epsilon_ddots].
        """
        epsilons, epsilon_dots = self.get_strains(y)
        
        M_sys, C_sys, K_sys, D_sys = self.compute_system_matrices(epsilons, epsilon_dots)
        
        F_ext_list = F_ext_func(t)
        F_gen = self.compute_generalized_forces(epsilons, F_ext_list)
        
        # Dynamic Equation: M*e_ddot + D*e_dot + C*e_dot + K*e = F_gen
        # C term is gyroscopic, D is viscous damping.
        rhs = F_gen - (C_sys @ epsilon_dots.flatten()) - (D_sys @ epsilon_dots.flatten()) - (K_sys @ epsilons.flatten())
        
        try:
            epsilon_ddots_flat = np.linalg.solve(M_sys, rhs)
        except np.linalg.LinAlgError:
            epsilon_ddots_flat = np.linalg.pinv(M_sys) @ rhs
            
        return np.hstack((epsilon_dots.flatten(), epsilon_ddots_flat))