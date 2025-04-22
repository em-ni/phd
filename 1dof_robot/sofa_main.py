import cv2
import numpy as np
import glob
from sklearn.cluster import KMeans
from scipy import optimize
from utils import *  
import os

experiment_name = 'sofa_free_motion'
# experiment_name = 'sofa_force_small_bending'

# Path to the directory containing images
if experiment_name == 'sofa_free_motion':
    image_files = glob.glob('L_10cm_OD_1.5mm/characterization_and_scale_force/sofa_free_motion/crop/*')
elif experiment_name == 'sofa_force_small_bending':
    image_files = glob.glob('L_10cm_OD_1.5mm/characterization_and_scale_force/sofa_force_small_bending/crop/*')

# Flag variable to track the first iteration
first_iteration = True  

# Iterate over each image file
for image_file in image_files:
    frame = cv2.imread(image_file)
    
    if frame is not None:
        # Your image processing code goes here
        # For example, convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if experiment_name == 'sofa_free_motion':
            # Define the color range for red (tip)
            lower_red1 = np.array([0, 100, 100])  # Increase saturation and value
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])  # Increase saturation and value
            upper_red2 = np.array([180, 255, 255])

            # Define a color range for yellow
            lower_yellow = np.array([25, 150, 150])  # Wider range for yellow detection
            upper_yellow = np.array([35, 255, 250])

        elif experiment_name == 'sofa_force_small_bending':
            # Define the color ranges for red
            lower_red1 = np.array([0, 140, 90])  # Slightly higher saturation and value
            upper_red1 = np.array([6, 255, 255])
            lower_red2 = np.array([174, 140, 90])  # Slightly higher saturation and value
            upper_red2 = np.array([179, 255, 255])
            
            # Define a color range for yellow
            lower_yellow = np.array([25, 150, 150])  # Wider range for yellow detection
            upper_yellow = np.array([35, 255, 250])

        # Get average of coorinates of point in red tip
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        red_tip = cv2.bitwise_and(frame, frame, mask=mask_red)
        x_tip, y_tip = get_red_tip_avg(red_tip)
        
        # Detect yellow objects
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_base = cv2.bitwise_and(frame, frame, mask=mask_yellow)

        # Get average coordinates of yellow base
        x_base, y_base = get_blue_base_avg(yellow_base)

        # Draw points on the frame (for debugging)
        if not np.isnan(x_tip) and not np.isnan(y_tip):
            cv2.circle(frame, (int(x_tip), int(y_tip)), 10, (0, 255, 0), 5)       
        if not np.isnan(x_base) and not np.isnan(y_base):
            cv2.circle(frame, (int(x_base), int(y_base)), 10, (0, 255, 0), 5)    

        # Apply the new edge detection process
        edges = process_frame_for_edges(frame)

        points = np.where(edges > 0)
        points = np.column_stack((points[1], points[0]))

        offset = 35  # Offset from the base to start looking for points

        # Get points between x_base and x_tip
        points = points[(points[:, 0] > x_base) & (points[:, 0] < x_tip - offset)]
        
        if len(points) > 0:
            reduced_points = []
            cols, rows = edges.shape[:2]
            cols_per_step = 2  # Reduce the number of points by a factor of cols_per_step

            for col in range(0, cols, cols_per_step):
                if col > x_base and col < x_tip - offset:  # Ensure col is strictly between x_base and x_tip
                    row_indices = np.where(edges[:, col] > 0)[0]
                    if len(row_indices) > 0:
                        row = int(np.mean(row_indices))
                        reduced_points.append([col, row])

            # Convert reduced_points to a NumPy array for further operations
            reduced_points = np.array(reduced_points)            

            # Draw all the reduced points on the frame (for debugging)
            for point in reduced_points:
                cv2.circle(frame, tuple(point), 1, (204, 255, 204), 1)

            # Get x n and y coordinates of the points as np.array
            x = reduced_points[:, 0]
            y = reduced_points[:, 1]

            # Initial guess for the parameters (x_c, y_c, radius)
            guess = (np.mean(x), np.mean(y), np.std(x))

            # Use scipy.optimize.minimize to solve the problem
            result = optimize.minimize(error_fct, guess, args=(x, y))

            # Extract the result
            x_c, y_c, radius = result.x
            center = (int(x_c), int(y_c))
            radius = int(radius)
            curvature = 1 / radius

            # Compute arc length
            arc_length = compute_arc_length(x_c, y_c, radius, x_base, y_base, x_tip, y_tip)

            # Define the path to the CSV file
            csv_path = f'L_10cm_OD_1.5mm/characterization_and_scale_force/{experiment_name}/cv_output.csv'  

            if first_iteration:
                # Delete previously written CSV file if it exists
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                first_iteration = False  # Set the flag to False after the first iteration

            with open(csv_path, 'a') as f:
                # Include pressure and force in the CSV entry
                f.write(f'{radius:.2f},{curvature:.2f},{arc_length:.2f},{x_tip:.2f},{y_tip:.2f},{x_base:.2f},{y_base:.2f}\n')

            # Check radius thresold
            if radius < 1000:               
                # Draw the fitted circle on the frame (for debugging)
                cv2.circle(frame, center, radius, (255, 204, 204), 2)

            # Display the curvature (for debugging)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'Radius: {:.2f} px'.format(radius), (10, 50), font, 0.5, (204, 229, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Arc Length: {:.2f} px'.format(arc_length), (10, 90), font, 0.5, (204, 229, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'X Tip: {:.2f}'.format(x_tip), (10, 210), font, 0.5, (204, 229, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Y Tip: {:.2f}'.format(y_tip), (10, 250), font, 0.5, (204, 229, 255), 1, cv2.LINE_AA)
        
        # Display the frame (for debugging)
        cv2.imshow('Frame', frame)
        cv2.imshow('Edges', edges)

        # Save the processed image
        processed_image_path = image_file.replace('crop', 'processed')  # Modify path as needed
        cv2.imwrite(processed_image_path, frame)  # Save frame with drawings

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cv2.destroyAllWindows()

