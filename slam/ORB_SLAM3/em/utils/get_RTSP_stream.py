import cv2


def capture_rtsp_stream(rtsp_url):
    # Create a VideoCapture object with the RTSP URL
    try:
        cap = cv2.VideoCapture(rtsp_url)
    except:
        print("Error: Unable to open the video stream.", rtsp_url)
        return

    # Get frame info
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frame Width:", frame_width)
    print("Frame Height:", frame_height)
    print("FPS:", fps)

    try:
        while True:
            # Capture frame-by-frame from the stream
            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to fetch the frame.")
                break

            # Display the resulting frame
            cv2.imshow("RTSP Stream", frame)

            # Press 'q' on the keyboard to exit the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # When everything done, release the capture object and destroy all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()


# Replace the below URL with your actual RTSP stream URL
rtsp_url = "rtsp://:@192.168.1.1:8554/session0.mpg"
capture_rtsp_stream(rtsp_url)
