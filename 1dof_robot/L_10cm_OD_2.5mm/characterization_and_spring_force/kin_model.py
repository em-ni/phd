import numpy as np

def tip_cartesian(k_coeff, p, L):
    if p == 0:
        x = L
        y = 0
    else:
        x =  L * np.sin(k_coeff * p)/(k_coeff * p)
        y =  L * (1 - np.cos(k_coeff * p))/(k_coeff * p)

    orientation = k_coeff * p
    return x, y, orientation
