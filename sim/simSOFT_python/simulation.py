# simsoft_python/simulation.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from elements import PiecewiseConstantDeformationArm
from lie_group_utils import skew

def setup_multisegment_benchmark():
    """
    Sets up the parameters for the Multisegment Rod-Driven Manipulator.
    Thesis Section: 6.2, Figure 6.11, Tables 6.8, 6.9
    """
    # Two segments, each 4 disks * 30mm = 120mm long
    lengths = [0.120, 0.120] 
    
    # Mass matrices for each segment from Table 6.9
    m1 = np.diag([2.869e-3]*3 + [3.531e-7, 7.690e-6, 7.690e-6])
    m2 = np.diag([2.269e-3]*3 + [2.593e-7, 3.894e-6, 3.894e-6])
    M_sections = [m1, m2]

    # Stiffness matrices for each segment from Table 6.8
    # K = diag(EA, GA_y, GA_z, GJ, EI_y, EI_z)
    # The thesis combines these values. We'll use the structure from the table.
    k_diag = [1.7839e5, 6.8808e4, 6.8808e4, 0.093, 0.0121, 0.0121]
    K_sections = [np.diag(k_diag), np.diag(k_diag)]

    # Damping matrix for dynamic relaxation (proportional damping)
    alpha_damp, beta_damp = 0.5, 5e-5
    D_sections = [alpha_damp * m + beta_damp * k for m, k in zip(M_sections, K_sections)]
    
    arm = PiecewiseConstantDeformationArm(lengths, M_sections, K_sections, D_sections)
    return arm

def run_static_equilibrium_simulation(load_case_N: float):
    """
    Runs a dynamic relaxation to find static equilibrium for a given load.
    """
    arm = setup_multisegment_benchmark()
    
    # Rod 1-1 tension is constant at 25N.
    # Rod 2-2 tension is the variable load case.
    # The rods are offset from the central backbone, creating a wrench.
    # Let's assume a small offset 'r_offset' in the y-direction.
    r_offset = 0.02 # 2cm offset
    
    # Tension in rod 1-1 (at end of segment 1)
    F1 = np.array([25, 0, 0]) # Force along x-axis
    m1 = skew([0, r_offset, 0]) @ F1 # Moment
    wrench1 = np.hstack((F1, m1))

    # Tension in rod 2-2 (at end of segment 2)
    F2 = np.array([load_case_N, 0, 0])
    m2 = skew([0, r_offset, 0]) @ F2
    wrench2 = np.hstack((F2, m2))

    # The external force is constant over time
    def F_ext_func(t):
        return [
            (arm.L_cumulative[1], wrench1),
            (arm.L_cumulative[2], wrench2)
        ]

    # Initial condition: at rest, no strain
    y0 = np.zeros(12 * arm.n_segments)
    
    t_span = [0, 5] # Simulate for 5 seconds to ensure equilibrium
    t_eval = np.linspace(t_span[0], t_span[1], 100)

    print(f"\n--- Running simulation for load case: {load_case_N} N ---")
    result = solve_ivp(
        lambda t, y: arm.state_derivative(t, y, F_ext_func),
        t_span,
        y0,
        method='BDF', # BDF is good for stiff problems like this
        t_eval=t_eval
    )
    print("Simulation finished.")
    
    # Extract final shape
    final_epsilons, _ = arm.get_strains(result.y[:, -1])
    backbone_points = []
    alphas = np.linspace(0, arm.total_length, 100)
    for alpha in alphas:
        H = arm.forward_kinematics(final_epsilons, alpha)
        backbone_points.append(H[:3, 3])
        
    return np.array(backbone_points)

if __name__ == '__main__':
    load_cases = [5.0, 10.0, 15.0]
    results = {}
    for load in load_cases:
        results[load] = run_static_equilibrium_simulation(load)
        
    # Plotting to replicate Figure 6.12
    plt.figure(figsize=(8, 6))
    colors = {5.0: 'blue', 10.0: 'red', 15.0: 'green'}
    
    for load, points in results.items():
        plt.plot(points[:, 0], points[:, 2], '-o', markersize=4,
                 color=colors[load], label=f'{int(load)} N')

    # Initial straight configuration
    plt.plot([0, 0.24], [0, 0], 'k--', label='Initial Shape')

    plt.title("Static Equilibrium of Multisegment Manipulator (Thesis Fig. 6.12)")
    plt.xlabel("x-displacement [m]")
    plt.ylabel("z-displacement [m]")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()