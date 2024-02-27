from scipy import optimize
from utils import *
from sklearn.cluster import KMeans

# Read the cropped video
cap = cv2.VideoCapture('data/video/diagonal_force.mp4')

# Get the video's width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or use 'XVID'
out_circle = cv2.VideoWriter('data/video/circle.mp4', fourcc, 20.0, (width, height))
out_edges = cv2.VideoWriter('data/video/edges.mp4', fourcc, 20.0, (width, height))

# Initialize slope history for filtering
m1_history = []

# Max history length
history_length = 5

# Process the first frame only
ret, frame = cap.read()

if ret:
    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define a very narrow color range for yellow (base)
    lower_yellow = np.array([25, 170, 170])  # Very narrow range for yellow detection
    upper_yellow = np.array([32, 255, 240])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_base = cv2.bitwise_and(frame, frame, mask=mask_yellow)

    # Get average coordinates of yellow base
    x_base, y_base = get_yellow_base_avg(yellow_base)

print('Base coordinates:', x_base, y_base)  
#quit()

# Initialize lists to store the history of x and y coordinates of the red tip
x_tip_history = []
y_tip_history = []

# Define the length of the history for moving average calculation
tip_history_length = 5  # Adjust based on your requirements


# Process each frame
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        # Convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the color range for red (tip)
        lower_red1 = np.array([0, 70, 50])  # Increased saturation
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])  # Adjusted hue and increased saturation
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        red_tip = cv2.bitwise_and(frame, frame, mask=mask_red)

        # Get average of coorinates of point in red tip
        x_tip, y_tip = get_red_tip_avg(red_tip)

        # Append the current tip positions to their respective histories
        x_tip_history.append(x_tip)
        y_tip_history.append(y_tip)

        # Ensure the histories do not exceed the maximum length
        if len(x_tip_history) > tip_history_length:
            x_tip_history.pop(0)
        if len(y_tip_history) > tip_history_length:
            y_tip_history.pop(0)

        # Calculate the moving average of the tip positions
        avg_x_tip = sum(x_tip_history) / len(x_tip_history)
        avg_y_tip = sum(y_tip_history) / len(y_tip_history)
        x_tip = avg_x_tip
        y_tip = avg_y_tip

        # Fix coordinates (taken from first frame)
        # Curvature (no force) video
        #x_base = 500
        #y_base = 525

        # Diagonal force video
        x_base = 490
        y_base = 515

        # Define the color range for blue
        lower_blue = np.array([100, 50, 50])  # Blue hue range
        upper_blue = np.array([140, 255, 255])
        
        # Detect blue objects
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_objects = cv2.bitwise_and(frame, frame, mask=mask_blue)

        # Find the pixels that belong to the blue objects
        blue_points = np.column_stack(np.where(mask_blue > 0))

        if len(blue_points) > 1:
            # Apply KMeans clustering to separate the blue objects
            kmeans = KMeans(n_clusters=2, random_state=0).fit(blue_points)
            labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_

            # Swap the x and y coordinates of the cluster centers for correct positioning
            x_blue_1, y_blue_1 = cluster_centers[0][1], cluster_centers[0][0]
            x_blue_2, y_blue_2 = cluster_centers[1][1], cluster_centers[1][0]

            # Compute distance between the two blue objects
            spring_length = np.sqrt((x_blue_2 - x_blue_1)**2 + (y_blue_2 - y_blue_1)**2)

            # Draw circles on the original frame to indicate the objects
            cv2.circle(frame, (int(x_blue_1), int(y_blue_1)), 10, (255, 192, 203), 2) 
            cv2.circle(frame, (int(x_blue_2), int(y_blue_2)), 10, (255, 192, 203), 2)  


        # Draw points on the frame (for debugging)
        if not np.isnan(x_tip) and not np.isnan(y_tip):
            cv2.circle(frame, (int(x_tip), int(y_tip)), 10, (0, 255, 0), 5)       
        if not np.isnan(x_base) and not np.isnan(y_base):
            cv2.circle(frame, (int(x_base), int(y_base)), 10, (0, 255, 0), 5)    
        if not np.isnan(x_blue_1) and not np.isnan(y_blue_1):
            cv2.circle(frame, (int(x_blue_1), int(y_blue_1)), 10, (0, 255, 0), 5)
        if not np.isnan(x_blue_2) and not np.isnan(y_blue_2):
            cv2.circle(frame, (int(x_blue_2), int(y_blue_2)), 10, (0, 255, 0), 5)

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

            # Store radius, curvature and arc length in a csv file
            with open('data/cv_output.csv', 'a') as f:
                f.write('{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(radius, curvature, arc_length, x_tip, y_tip, spring_length))

            # Check radius thresold
            if radius < 600:               
                # Draw the fitted circle on the frame (for debugging)
                cv2.circle(frame, center, radius, (255, 204, 204), 2)

            # Display the curvature (for debugging)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'Radius: {:.2f} px'.format(radius) , (10, 50), font, 0.5, (204, 229, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Spring Length: {:.2f} px'.format(spring_length), (10, 90), font, 0.5, (204, 229, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Arc Length: {:.2f} px'.format(arc_length), (10, 130), font, 0.5, (204, 229, 255), 1, cv2.LINE_AA)
            

            # Draw the tangent line on the edges frame to display the tip orientation
            if not np.isnan(x_tip) and not np.isnan(y_tip):
                # Draw red point on tip coordinates 
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                cv2.circle(edges, (int(x_tip), int(y_tip)), 1, (206, 0, 88), 5)

                # Filter slope
                # Get the slope of the line between the center and the tip
                m1 = (y_tip - y_c) / (x_tip - x_c)

                # Append the value to the history
                m1_history.append(m1)

                # Ensure the history does not exceed max length
                if len(m1_history) > history_length:
                    m1_history.pop(0)

                # Compute the average
                avg_m1 = np.mean(m1_history)

                # Get tangent line equation                
                m2 = -1 / avg_m1
                b = y_tip - m2 * x_tip

                # Fixed length of the arrow
                arrow_length = 50

                # Calculate angle of the vector using the slope
                theta = np.arctan(avg_m1) - np.pi / 2 # Subtract Pi/2 for 90 degree rotation

                # Calculate the x and y coordinates for the end of the vector
                x2 = x_tip + arrow_length * np.cos(theta)
                y2 = y_tip + arrow_length * np.sin(theta) # Add because the y-axis is inverted in image coordinates

                # Draw arrowed line
                cv2.arrowedLine(edges, (int(x_tip), int(y_tip)), (int(x2), int(y2)), (206, 0, 88), 2, tipLength=0.2)

            # Draw the tip position predicted by the model
            #x_tip_pred, y_tip_pred, theta_pred = tip_cartesian(k_coeff, p, arc_length, x_base, y_base)
            # ADDED in 
                
        
        # Display the frame (for debugging)
        cv2.imshow('Frame', frame)
        cv2.imshow('Edges', edges)

        # Save video of frame and edges
        out_circle.write(frame)
        out_edges.write(edges)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

