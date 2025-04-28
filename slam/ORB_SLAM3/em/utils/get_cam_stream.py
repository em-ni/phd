import cv2
import numpy as np
import argparse
import time
import os


def list_available_cameras():
    """List all available camera devices."""
    available = []
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            available.append((i, f"{width}x{height}@{fps}fps"))
            cap.release()
    return available


def capture_stream(
    camera_index=0,
    width=None,
    height=None,
    fps=None,
    record=False,
    output_folder="output",
):
    """Capture video stream from specified camera."""
    # Create a VideoCapture object
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        return False

    # Set camera properties if specified
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)

    # Get actual camera properties
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        f"Camera {camera_index} opened with resolution: {actual_width}x{actual_height}, FPS: {actual_fps}"
    )

    # Setup video writer if recording
    video_writer = None
    if record:
        os.makedirs(output_folder, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_file = f"{output_folder}/camera_{camera_index}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            video_file, fourcc, actual_fps, (int(actual_width), int(actual_height))
        )
        print(f"Recording to: {video_file}")

    # Create window
    window_name = f"Camera {camera_index} Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture frame")
                break

            # Calculate actual FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                current_fps = frame_count / elapsed_time
                print(
                    f"Current resolution: {current_width}x{current_height} Current FPS: {current_fps:.1f}",
                    end="\r",
                    flush=True,
                )
                frame_count = 0
                start_time = time.time()

            # Display the resulting frame
            cv2.imshow(window_name, frame)

            # Record frame if enabled
            if video_writer:
                video_writer.write(frame)

            # Press 'q' to exit, 's' to save a snapshot
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                snapshot_file = (
                    f"{output_folder}/snapshot_{camera_index}_{timestamp}.jpg"
                )
                cv2.imwrite(snapshot_file, frame)
                print(f"Saved snapshot to: {snapshot_file}")

    finally:
        # Release resources
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Camera released and resources cleaned up")
        return True


def main():
    parser = argparse.ArgumentParser(description="HDMI Camera Stream Capture")
    parser.add_argument(
        "-l", "--list", action="store_true", help="List available cameras"
    )
    parser.add_argument(
        "-c", "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument("-W", "--width", type=int, help="Set capture width")
    parser.add_argument("-H", "--height", type=int, help="Set capture height")
    parser.add_argument("-f", "--fps", type=int, help="Set capture FPS")
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record video to file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        help="Output folder for recordings",
    )

    args = parser.parse_args()

    if args.list:
        print("Scanning for available cameras...")
        cameras = list_available_cameras()
        if cameras:
            print("Available cameras:")
            for idx, spec in cameras:
                print(f"  Camera index {idx}: {spec}")
        else:
            print("No cameras detected")
        return

    capture_stream(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        record=args.record,
        output_folder=args.output,
    )


if __name__ == "__main__":
    main()
