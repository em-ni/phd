import os
import glob
import numpy as np
import cv2
import argparse
from tqdm import tqdm

def verify_videos(args):
    """
    Converts video.npy files found in dataset/seq_XXX/ to .mp4 for inspection.
    """
    dataset = args.dataset
    
    # Find all sequence folders
    seq_pattern = os.path.join(dataset, "seq_*")
    sequences = sorted(glob.glob(seq_pattern))
    
    if not sequences:
        print(f"[ERROR] No sequences found in {dataset}")
        return

    print(f"Found {len(sequences)} sequences.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If img_size is specified, create subdirectory and override scale logic logic later
    if args.img_size:
        save_dir = os.path.join(args.output_dir, str(args.img_size))
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving videos to: {save_dir} (Resolution: {args.img_size}x{args.img_size})")
    else:
        save_dir = args.output_dir
        print(f"Saving videos to: {save_dir} (Scale: {args.scale})")

    for seq_path in sequences:
        seq_name = os.path.basename(seq_path)
        video_path = os.path.join(seq_path, "video.npy")
        
        if not os.path.exists(video_path):
            print(f"[SKIP] No video.npy in {seq_name}")
            continue
            
        print(f"\n--- Processing: {seq_name} ---")
        
        try:
            # Load Video (N, H, W, 3)
            # Use mmap_mode to avoid loading entire file into RAM
            video_data = np.load(video_path, mmap_mode='r')
            N, H, W, C = video_data.shape
            print(f"    Original Shape: {video_data.shape}")
            
            # Determine Output Size
            if args.img_size:
                out_w = args.img_size
                out_h = args.img_size
                print(f"    Output Size: {out_w}x{out_h} (Forced)")
            else:
                out_w = int(W * args.scale)
                out_h = int(H * args.scale)
                print(f"    Output Size: {out_w}x{out_h} (Scale: {args.scale})")
            
            save_path = os.path.join(save_dir, f"{seq_name}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, args.fps, (out_w, out_h))
            
            # Process frames
            # We iterate through all frames
            for i in tqdm(range(N), desc=f"    Converting {seq_name}"):
                frame = video_data[i]
                
                # Resize
                if args.img_size:
                     frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                elif args.scale != 1.0:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                
                # Assuming BGR input from cv2/npy (standard for this dataset apparently)
                # If it looks wrong (blue lungs), we might need to swap.
                # Based on previous steps, input is BGR.
                frame_bgr = frame 
                
                # Ensure uint8
                if frame_bgr.dtype != np.uint8:
                     if frame_bgr.max() <= 1.0:
                         frame_bgr = (frame_bgr * 255).astype(np.uint8)
                     else:
                         frame_bgr = frame_bgr.astype(np.uint8)
                else:
                     # Make a copy if we need to draw on it (though we aren't drawing text anymore to be clean)
                     # Actually, let's add frame number for debugging
                     frame_bgr = frame_bgr.copy()

                # Add info text (Frame Number)
                cv2.putText(frame_bgr, f"Fr: {i}/{N}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Check for black frame
                if frame_bgr.mean() < 1.0:
                    cv2.putText(frame_bgr, "BLACK FRAME", (out_w//2 - 100, out_h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                out.write(frame_bgr)
            
            out.release()
            print(f"    Saved to {save_path}")
                    
        except Exception as e:
            print(f"[ERROR] Failed to process {seq_name}: {e}")

    print("\nAll Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./dataset/sequences', help="Path to sequences directory")
    parser.add_argument('--output_dir', type=str, default='./dataset_check_videos', help="Where to save mp4s")
    parser.add_argument('--scale', type=float, default=0.5, help="Downscale factor (e.g. 0.5 for half size). Ignored if img_size is set.")
    parser.add_argument('--img_size', type=int, default=None, help="Force output size (WxH), overrides scale. Creates subfolder.")
    parser.add_argument('--fps', type=int, default=30, help="Output FPS")
    
    args = parser.parse_args()
    
    verify_videos(args)