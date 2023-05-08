from scipy import optimize
from utils import *

# Read the cropped video
cap = cv2.VideoCapture('data/video/curvature_crop.mov')

# Get the video's width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or use 'XVID'
out_circle = cv2.VideoWriter('data/video/circle.mp4', fourcc, 20.0, (width, height))
out_edges = cv2.VideoWriter('data/video/edges.mp4', fourcc, 20.0, (width, height))


# Process each frame
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        # Preprocess the image (convert to grayscale, blur, threshold)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        # Detect edges
        edges = cv2.Canny(thresh, 50, 150)
        
        # Filter the red color of the robot tip (you might need to adjust the color range)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        red_tip = cv2.bitwise_and(frame, frame, mask=mask)

        # Get average of coorinates of point in red tip
        x_tip, y_tip = get_red_tip_avg(red_tip)

        # Draw the average point on the frame (for debugging)
        if not np.isnan(x_tip) and not np.isnan(y_tip):
            cv2.circle(frame, (int(x_tip), int(y_tip)), 1, (0, 255, 0), 5)        
        
        # Detect circle and compute the curvature
        # Extract a set of points from the edges of the robot
        points = np.where((edges > 0))
        points = np.column_stack((points[1], points[0]))
           
        if len(points) > 0:
            # Get the coordinates of the base of the robot
            # Get the 10 points with the highets y and average their coordinates
            points = points[points[:, 1].argsort()][-10:]
            x_base = np.mean(points[:, 0])
            y_base = np.mean(points[:, 1])

            # Draw the average point on the frame (for debugging)
            cv2.circle(frame, (int(x_base), int(y_base)), 1, (0, 255, 0), 5)

            # Reduce the number of points using cv2.reduce
            rows, cols = edges.shape[:2]
            rows_per_step = 2  # Reduce the number of points by a factor of rows_per_step
            reduced_points = []
            for row in range(0, rows, rows_per_step):
                column_indices = np.where(edges[row, :] > 0)[0]
                if len(column_indices) > 0:
                    col = int(np.mean(column_indices))
                    reduced_points.append([col, row])
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
            with open('curvature.csv', 'a') as f:
                f.write('{:.2f},{:.2f},{:.2f}\n'.format(radius, curvature, arc_length))

            # Check radius thresold
            if radius < 600:               
                # Draw the fitted circle on the frame (for debugging)
                cv2.circle(frame, center, radius, (255, 204, 204), 2)

            # Display the curvature (for debugging)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'Radius: {:.2f} px'.format(radius) , (10, 50), font, 0.5, (204, 229, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Curvature: {:.2f}'.format(curvature), (10, 90), font, 0.5, (204, 229, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Arc Length: {:.2f}'.format(arc_length), (10, 130), font, 0.5, (204, 229, 255), 1, cv2.LINE_AA)

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

