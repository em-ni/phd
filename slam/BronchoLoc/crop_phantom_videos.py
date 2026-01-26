"""
Script to center-crop phantom videos from 1280x720 to 720x720 (square).
Creates new sequence folders with _crop suffix.
"""
import os
import numpy as np
import cv2
from tqdm import tqdm

sequences_dir = 'dataset/sequences'

# Find phantom sequences (exclude already cropped ones)
phantom_dirs = [d for d in os.listdir(sequences_dir) 
                if d.startswith('seq_phantom') and not d.endswith('_crop')]

print(f"Found {len(phantom_dirs)} phantom sequences to crop")
print("-" * 60)

for seq in sorted(phantom_dirs):
    src_dir = os.path.join(sequences_dir, seq)
    dst_dir = os.path.join(sequences_dir, f"{seq}_sq")  # _sq for square
    
    vid_path = os.path.join(src_dir, 'video.npy')
    traj_path = os.path.join(src_dir, 'trajectory.npy')
    
    if not os.path.exists(vid_path):
        print(f"[SKIP] {seq}: No video.npy found")
        continue
    
    # Load video
    vid = np.load(vid_path, mmap_mode='r')
    frames, h, w, c = vid.shape
    print(f"\n[{seq}] Original: {w}x{h}, {frames} frames")
    
    # Skip if already square
    if h == w:
        print(f"  Already square, skipping")
        continue
    
    # Calculate center crop coordinates
    # For 1280x720 -> crop to 720x720 (center)
    crop_size = min(h, w)  # 720
    x_start = (w - crop_size) // 2  # (1280-720)//2 = 280
    y_start = (h - crop_size) // 2  # (720-720)//2 = 0
    
    print(f"  Cropping to {crop_size}x{crop_size} (x_start={x_start}, y_start={y_start})")
    
    # Create output directory
    os.makedirs(dst_dir, exist_ok=True)
    
    # Process video in chunks to avoid memory issues
    cropped_frames = []
    chunk_size = 500
    for i in tqdm(range(0, frames, chunk_size), desc=f"  Processing"):
        end_i = min(i + chunk_size, frames)
        chunk = np.array(vid[i:end_i])
        
        # Center crop each frame
        cropped = chunk[:, y_start:y_start+crop_size, x_start:x_start+crop_size, :]
        cropped_frames.append(cropped)
    
    # Concatenate and save
    cropped_video = np.concatenate(cropped_frames, axis=0)
    print(f"  Cropped shape: {cropped_video.shape}")
    
    # Save cropped video
    dst_vid_path = os.path.join(dst_dir, 'video.npy')
    np.save(dst_vid_path, cropped_video)
    print(f"  Saved: {dst_vid_path}")
    
    # Copy trajectory (unchanged)
    if os.path.exists(traj_path):
        traj = np.load(traj_path)
        dst_traj_path = os.path.join(dst_dir, 'trajectory.npy')
        np.save(dst_traj_path, traj)
        print(f"  Copied trajectory: {dst_traj_path}")
    
    # Save a sample frame as image for visual verification
    sample_idx = frames // 2  # middle frame
    sample_frame = cropped_video[sample_idx]
    sample_path = os.path.join(dst_dir, 'sample_frame.png')
    cv2.imwrite(sample_path, cv2.cvtColor(sample_frame, cv2.COLOR_RGB2BGR))
    print(f"  Saved sample frame: {sample_path}")

print("\n" + "=" * 60)
print("Done! Cropped sequences saved with _sq suffix")
