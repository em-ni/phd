import cv2
import numpy as np
import yaml

# Camera name
# camera_name = 'Misumi_200x200p'
# camera_name = "Misumi_400x380p"
# camera_name = "videoscope_1280x720p"
camera_name = "videoscope_940x970p"

# Define the chessboard size
chessboard_size = (9, 6)
# Take frame size from name
frame_size = (
    int(camera_name.split("_")[1].split("x")[0]),
    int(camera_name.split("_")[1].split("x")[1].split("p")[0]),
)
FPS = 60
# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []
imgpoints = []

# Folder name calibration_images + camera name
folder_name = "calibration_images_" + camera_name
first = 1
last = 15

# Capture images of the calibration pattern
images = [
    cv2.imread(f"{folder_name}/image_{i}.png") for i in range(first, last + 1)
]  # Assuming 20 images

for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow("img", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, frame_size, None, None
)

print(f"Camera Matrix: \n{camera_matrix}")
print(f"Distortion Coefficients: \n{dist_coeffs}")


# Create YAML file
calibration_data = {
    "File.version": "1.0",
    "Camera.type": "PinHole",
    # These parameters specify the width and height of the camera frame in pixels. They are set based on frame_size.
    "Camera.width": frame_size[0],
    "Camera.height": frame_size[1],
    # The focal lengths of the camera in pixels, measured in the x and y directions.
    # These values are derived from the camera matrix (camera_matrix) which is calculated during the calibration process.
    "Camera1.fx": float(camera_matrix[0, 0]),
    "Camera1.fy": float(camera_matrix[1, 1]),
    # The optical centers of the camera in the x and y coordinates, respectively. These are also obtained from the camera matrix.
    # The optical center is the point in the image where the camera's line of sight intersects the image plane.
    "Camera1.cx": float(camera_matrix[0, 2]),
    "Camera1.cy": float(camera_matrix[1, 2]),
    # Radial distortion coefficients. These parameters correct for radial distortion in the camera lens, where straight lines appear curved in the image.
    # k3 is included only if there are more than 4 distortion coefficients available.
    "Camera1.k1": float(dist_coeffs[0, 0]),
    "Camera1.k2": float(dist_coeffs[0, 1]),
    "Camera1.k3": float(dist_coeffs[0, 4]) if len(dist_coeffs) > 4 else 0.0,
    # Tangential distortion coefficients. These parameters adjust for lens distortion that occurs when the lens and the image plane are not parallel.
    "Camera1.p1": float(dist_coeffs[0, 2]),
    "Camera1.p2": float(dist_coeffs[0, 3]),
    "Camera.fps": FPS,
    # Indicates whether the camera outputs RGB color images (1 for true, 0 for false).
    "Camera.RGB": 1,
    # Represents the baseline times focal length, used in stereo cameras for depth perception.
    "Camera.bf": 40.0,
    # Is the depth threshold, which might be used to determine how far objects can be from the camera to be considered for processing
    "Camera.ThDepth": 40.0,
    # The number of features to be extracted by the ORB algorithm.
    # More features allow for more detailed image analysis but require more processing power.
    "ORBextractor.nFeatures": 500,
    # The scale factor between levels in the image pyramid used by ORB for feature detection.
    # Smaller values mean more levels and finer details.
    "ORBextractor.scaleFactor": 1.2,
    # The number of levels in the pyramid. More levels allow the detector to find features at multiple scales.
    "ORBextractor.nLevels": 8,
    # Thresholds for detecting corners using the FAST algorithm. iniThFAST is the initial threshold, and minThFAST is the minimum threshold.
    # Higher values mean fewer corners are detected.
    "ORBextractor.iniThFAST": 1,
    "ORBextractor.minThFAST": 1,
    # These parameters control the visual appearance of various elements (like keyframes, graphs, points) in the visualization or viewer module
    # that might be used to display the camera's output or analysis results.
    "Viewer.KeyFrameSize": 0.05,
    "Viewer.KeyFrameLineWidth": 1.0,
    "Viewer.GraphLineWidth": 0.9,
    "Viewer.PointSize": 2.0,
    "Viewer.CameraSize": 0.08,
    "Viewer.CameraLineWidth": 3.0,
    "Viewer.ViewpointX": 0.0,
    "Viewer.ViewpointY": -0.7,
    "Viewer.ViewpointZ": -1.8,
    "Viewer.ViewpointF": 500.0,
}

yaml_filename = f"calibration_{camera_name}.yaml"
with open(yaml_filename, "w") as yaml_file:
    yaml_file.write("%YAML:1.0\n")
    yaml.dump(calibration_data, yaml_file, default_flow_style=False)

print(f"Calibration data saved to {yaml_filename}")
