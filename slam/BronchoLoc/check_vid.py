import os
import glob
import numpy as np
import cv2
import time

def verify_videos(dataset, fps=30):
    """
    Plays back video.npy files found in dataset/seq_XXX/
    """
    # Find all sequence folders
    seq_pattern = os.path.join(dataset, "seq_*")
    sequences = sorted(glob.glob(seq_pattern))
    
    if not sequences:
        print(f"[ERROR] No sequences found in {dataset}")
        return

    print(f"Found {len(sequences)} sequences. Press 'q' to quit, 'n' for next.")

    for seq_path in sequences:
        seq_name = os.path.basename(seq_path)
        video_path = os.path.join(seq_path, "video.npy")
        
        if not os.path.exists(video_path):
            print(f"[SKIP] No video.npy in {seq_name}")
            continue
            
        print(f"\n--- Playing: {seq_name} ---")
        
        try:
            # Load Video (N, H, W, 3)
            video_data = np.load(video_path)
            N, H, W, C = video_data.shape
            print(f"    Shape: {video_data.shape} | Frames: {N}")
            
            # Calculate delay for playback
            delay = int(1000 / fps)
            
            for i in range(N):
                frame = video_data[i]
                
                # Panda3D usually captures RGB, OpenCV expects BGR
                # If your video looks blue/orange inverted, remove/add this line
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Add info text
                cv2.putText(frame_bgr, f"{seq_name}: {i}/{N}", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imshow("Dataset Verification", frame_bgr)
                
                # Controls
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('q'):
                    print("Quitting.")
                    return
                if key == ord('n'):
                    print("Skipping to next sequence...")
                    break
                    
        except Exception as e:
            print(f"[ERROR] Failed to load/play {video_path}: {e}")

    cv2.destroyAllWindows()
    print("\nDone.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_ROOT = os.path.join(BASE_DIR, "dataset", "sequences")
    # ---------------------
    
    verify_videos(DATASET_ROOT)