import cv2

# Open the video
cap = cv2.VideoCapture('curvature_crop.mov')

# Get the video's width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or use 'XVID'
out = cv2.VideoWriter('curvature_crop_no_wm.mp4', fourcc, 20.0, (2*width//3, height))

# Define the region to crop to (here we cut the video size in half)
x_start = 0
x_end = x_start + 2 * width // 3
y_start = 0
y_end = height

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # Crop the frame
        frame = frame[y_start:y_end, x_start:x_end]

        # Write the cropped frame to the new video
        out.write(frame)

        # Display the resulting frame (for debugging)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
