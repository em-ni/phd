import cv2
import numpy as np

# Define the circle equation and the error function
def circle_eqn(x_c, y_c, x, y):
    return np.sqrt((x-x_c)**2 + (y-y_c)**2)

# Define the error function
def error_fct(params, x, y):
    x_c, y_c, r = params
    return np.sum((circle_eqn(x_c, y_c, x, y) - r)**2)

# Get average of coorinates of point in red tip
def get_red_tip_avg(red_tip):
    red_tip_gray = cv2.cvtColor(red_tip, cv2.COLOR_BGR2GRAY)
    _, red_tip_thresh = cv2.threshold(red_tip_gray, 127, 255, cv2.THRESH_BINARY)
    red_tip_edges = cv2.Canny(red_tip_thresh, 50, 150)
    red_tip_points = np.where((red_tip_edges > 0))
    red_tip_points = np.column_stack((red_tip_points[1], red_tip_points[0]))        

    # Get average of coorinates of point in red tip
    red_tip_x = red_tip_points[:, 0]
    red_tip_y = red_tip_points[:, 1]
    x_tip = np.mean(red_tip_x)
    y_tip = np.mean(red_tip_y)

    return x_tip, y_tip

# Compute arc length
def compute_arc_length(x_c, y_c, radius, x_base, y_base, x_tip, y_tip):
    theta_base = np.arctan2(y_base - y_c, x_base - x_c)
    theta_tip = np.arctan2(y_tip - y_c, x_tip - x_c)
    dtheta = np.abs(theta_tip - theta_base)
    dtheta = np.min([dtheta, 2*np.pi - dtheta])  # Ensure the angle is at most pi
    arc_length = radius * dtheta
    
    return arc_length