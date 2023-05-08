import cv2

video = cv2.VideoCapture('data/video/edges.mp4')
fps = video.get(cv2.CAP_PROP_FPS)  # Getting the frames per second

# Start frame number for X seconds
start_frame_number = int(65 * fps) 

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/video/edges_cut.mp4', fourcc, fps, (int(video.get(3)), int(video.get(4))))

video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

while True:
    ret, frame = video.read()
    if not ret:
        break

    out.write(frame)  # Write out frame to video

# Release everything if job is finished
video.release()
out.release()
cv2.destroyAllWindows()
