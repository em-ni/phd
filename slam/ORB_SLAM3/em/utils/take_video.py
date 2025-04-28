import cv2

# Set up the camera source. Adjust the RTSP URL as needed.
# camera_index = "rtsp://:@192.168.1.1:8554/session0.mpg"
camera_index = 2
cap = cv2.VideoCapture(camera_index)

# Attempt to use the original video's frame rate and resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # Capture the original fps from the stream

# If there's an error capturing the correct fps (e.g., returns zero), set a default fps
if fps <= 0:
    print("Warning: Unable to retrieve FPS from the camera stream. Using default FPS.")
    fps = (
        59.94  # Default fps, you can adjust this based on your camera's specifications
    )

# Define the codec and create a VideoWriter object to write the video
# Using a format that supports high-quality output, like 'MP4V' for .mp4 files
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
out = cv2.VideoWriter("o.mp4", fourcc, fps, (frame_width, frame_height))

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Write the frame into the file 'output.mp4'
        out.write(frame)

        # Display the resulting frame
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            print("Exiting: 'q' key pressed.")
            break
finally:
    # When everything is done, release the capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()
