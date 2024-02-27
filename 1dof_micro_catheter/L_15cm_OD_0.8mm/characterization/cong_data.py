import cv2
import numpy as np
from scipy import optimize
from utils import *

# Read the cropped video
cap = cv2.VideoCapture('data/video/red.mp4')

# Get the video's width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out_circle = cv2.VideoWriter('data/video/cong_circle.mp4', fourcc, 20.0, (width, height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # Filter the red, gray, and black colors (you might need to adjust the color ranges)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Filter red color
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        # Filter gray color
        lower_gray = np.array([0, 0, 100])
        upper_gray = np.array([180, 50, 180])
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)

        # Filter black color
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])
        black_mask = cv2.inRange(hsv, lower_black, upper_black)

        # Combine the masks
        mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, gray_mask), black_mask)
        points_mask = cv2.bitwise_and(frame, frame, mask=mask)

        # Find contours of the red, gray, and black regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract the coordinates of the reduced points
        reduced_points = []
        for contour in contours:
            for point in contour:
                x, y = point[0]
                reduced_points.append([x, y])

                # Draw a circle on the frame at the reduced point
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # Draw all the reduced points on the frame (for debugging)
        for point in reduced_points:
            cv2.circle(frame, tuple(point), 1, (204, 255, 204), 1)

        # Get x and y coordinates of the reduced points as np.array
        reduced_points = np.array(reduced_points)
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

        # Store radius, curvature and arc length in a csv file
        with open('data/cong_cv_output.csv', 'a') as f:
            f.write('{:.2f},{:.2f}\n'.format(radius, curvature))

        # Check radius thresold
        if radius < 600:               
            # Draw the fitted circle on the frame (for debugging)
            cv2.circle(frame, center, radius, (255, 204, 204), 2)

        # Display the curvature (for debugging)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Radius: {:.2f} px'.format(radius) , (10, 50), font, 0.5, (204, 229, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Curvature: {:.2f}'.format(curvature), (10, 90), font, 0.5, (204, 229, 255), 1, cv2.LINE_AA)

        # Display the frame (for debugging)
        cv2.imshow('Frame', frame)

        # Save video of frame and edges
        out_circle.write(frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
