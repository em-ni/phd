import pandas as pd
import numpy as np
import cv2
from kin_model import tip_cartesian

# Import pressure data (value of the pressure every 0.005 seconds)
pressure = pd.read_csv('data/Pressure.csv', header=None)
print("\nPressure data shape: ", pressure.shape)
# print(pressure) 
#          0     0.005      0.01     0.015      0.02  ...    82.365     82.37    82.375     82.38    82.385
#0  2.090457  2.089145  2.090457  2.087833  2.089145  ...  2.044536  2.043224  2.043224  2.037976  2.039288

# Number of data points
n = pressure.shape[1]

# First pressure value
p = pressure.iloc[1, 0]
print("\nFirst pressure value: ", p)

# Import cv output (radius, curvature, arc length, x_base, y_base)
cv_output = pd.read_csv('data/cv_output.csv', header=None)
arc_length = cv_output.iloc[:, 2].values
x_base = cv_output.iloc[:, 3].values
y_base = cv_output.iloc[:, 4].values

print("\narc length shape: ", arc_length)
print("x_base shape: ", x_base)
print("y_base shape: ", y_base)

# Compute average of arc_length , x_base and y_base
arc_length_avg = np.mean(arc_length)
x_base_avg = np.mean(x_base)
y_base_avg = np.mean(y_base)

print("\narc length avg: ", arc_length_avg, " px")
print("x_base avg: ", x_base_avg, " px")
print("y_base avg: ", y_base_avg, " px")

# Tip length 
L = 5 #mm

# Conversion factor mm to px
mm2px = arc_length_avg / L #px/mm

# Import curvature from optimized model
opt_results = pd.read_csv('data/optimization_results.csv', header=None)
opt_results = opt_results.astype(str)
opt_results = opt_results[0].str.split(',', expand=True)
opt_results = opt_results.apply(pd.to_numeric)

#print("shape: ", opt_results.shape)

epsilon_coeff = opt_results.iloc[0, 0]  # This gives you the first row value
k_coeff = opt_results.iloc[1, 0]  # This gives you the second row value

print("\nOptimal Values")
print("epsilon coeff:", epsilon_coeff)
print("curvature coeff:", k_coeff)

# Open data/video/edges.mp4 and count the frames
cap = cv2.VideoCapture('data/video/edges.mp4')
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Frame count: ", frame_count)

# Compute step size
step = n // frame_count 

# Get the video's width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or use 'XVID'
out_video = cv2.VideoWriter('data/video/model_validation.mp4', fourcc, 20.0, (frame_width, frame_height))

# Init index to get the right pressure value
index = 0

# Loop over all frames in the video
while(cap.isOpened()):
    # Read the current frame
    ret, frame = cap.read()

    if ret:
        # Get the right pressure value for the current frame
        p_index = round(index * step)
        p = pressure.iloc[1, p_index]

        # Compute model prediction
        x_tip_pred, y_tip_pred, theta_pred = tip_cartesian(k_coeff, p, L)
        
        # Convert to px
        x_tip_pred = x_tip_pred * mm2px 
        y_tip_pred = y_tip_pred * mm2px 

        print("x_tip_pred: ", x_tip_pred, " px")
        print("y_tip_pred: ", y_tip_pred, " px")
        print("Lpx: ", L*mm2px, " px")

        # transform coordinates in opencv frame
        x_tip_pred = x_base_avg 
        y_tip_pred = y_base_avg - x_tip_pred

        # Draw the dot at (x_base_avg, y_base_avg)
        cv2.circle(frame, (int(x_base_avg), int(y_base_avg)), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(frame, (int(x_tip_pred), int(y_tip_pred)), radius=5, color=(0, 255, 0), thickness=-1)

        # Write the frame with the drawn dot into the output video
        out_video.write(frame)
        
        # Display the frame (for debugging)
        cv2.imshow('Model vs Experiment', frame)

        # Increment index
        index += 1

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and writer objects
cap.release()
out_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()


