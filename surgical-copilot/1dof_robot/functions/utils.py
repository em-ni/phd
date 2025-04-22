import cv2
import numpy as np
import os

# Define the circle equation and the error function
def circle_eqn(x_c, y_c, x, y):
    return np.sqrt((x-x_c)**2 + (y-y_c)**2)

# Define the error function
def error_fct(params, x, y):
    x_c, y_c, r = params
    return np.sum((circle_eqn(x_c, y_c, x, y) - r)**2)

# Get average of coorinates of point in blue base
def get_blue_base_avg(mask_blue):
    blue_base_points = np.where(mask_blue > 0)
    blue_base_points = np.column_stack((blue_base_points[1], blue_base_points[0]))

    if blue_base_points.size == 0:
        return np.nan, np.nan  # Return NaN if no blue base is detected

    x_base = np.mean(blue_base_points[:, 0])
    y_base = np.mean(blue_base_points[:, 1])

    return x_base, y_base


# Get average of coorinates of point in red tip
def get_red_tip_avg(red_tip):
    red_tip_points = np.where(red_tip > 0)
    red_tip_points = np.column_stack((red_tip_points[1], red_tip_points[0]))    

    if red_tip_points.size == 0:
        return np.nan, np.nan    

    # Get average of coorinates of point in red tip
    red_tip_x = red_tip_points[:, 0]
    red_tip_y = red_tip_points[:, 1]
    x_tip = np.mean(red_tip_x)
    y_tip = np.mean(red_tip_y)

    return x_tip, y_tip

# Get average coordinates of yellow base
def get_yellow_base_avg(yellow_base):
    yellow_base_gray = cv2.cvtColor(yellow_base, cv2.COLOR_BGR2GRAY)
    _, yellow_base_thresh = cv2.threshold(yellow_base_gray, 127, 255, cv2.THRESH_BINARY)
    yellow_base_edges = cv2.Canny(yellow_base_thresh, 50, 150)
    yellow_base_points = np.where((yellow_base_edges > 0))
    yellow_base_points = np.column_stack((yellow_base_points[1], yellow_base_points[0]))

    # Get average coordinates of yellow base
    yellow_base_x = yellow_base_points[:, 0]
    yellow_base_y = yellow_base_points[:, 1]
    x_base = np.mean(yellow_base_x)
    y_base = np.mean(yellow_base_y)

    return x_base, y_base

# Compute arc length
def compute_arc_length(x_c, y_c, radius, x_base, y_base, x_tip, y_tip):
    theta_base = np.arctan2(y_base - y_c, x_base - x_c)
    theta_tip = np.arctan2(y_tip - y_c, x_tip - x_c)
    dtheta = np.abs(theta_tip - theta_base)
    dtheta = np.min([dtheta, 2*np.pi - dtheta])  # Ensure the angle is at most pi
    arc_length = radius * dtheta
    
    return arc_length

def parse_filename(filename):
    basename = os.path.basename(filename)  # Get the basename, e.g., "100_50.jpg"
    name, _ = os.path.splitext(basename)  # Remove the file extension, e.g., "100_50"
    parts = name.split('_')  # Split by underscore
    pressure = parts[0]
    force = parts[1] if len(parts) > 1 else 0  # Use 0 for force if it's not in the filename
    return pressure, force


# Function to process the frame for edge detection using green and yellow background
def process_frame_for_edges(frame):
    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for the green background
    lower_green = np.array([30, 40, 40])  # Lower bound for green
    upper_green = np.array([90, 255, 255])  # Upper bound for green

    # Define the color range for the yellow background
    lower_yellow = np.array([10, 100, 100])  # Lower bound for yellow
    upper_yellow = np.array([50, 255, 255])  # Upper bound for yellow
    
    # Create masks to filter out the green and yellow backgrounds
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine the masks to filter out both green and yellow backgrounds
    mask_background = cv2.bitwise_or(mask_green, mask_yellow)
    
    # Invert the mask to get the objects as white on a black background
    mask_objects = cv2.bitwise_not(mask_background)
    
    # Apply a median blur to reduce noise in the mask
    mask_objects_blurred = cv2.medianBlur(mask_objects, 5)
    
    # Detect edges using Canny on the masked objects
    edges = cv2.Canny(mask_objects_blurred, 100, 200)

    return edges