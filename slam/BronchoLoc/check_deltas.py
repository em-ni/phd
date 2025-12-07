import numpy as np
import glob
import os
import argparse

def analyze_dataset(data_root, frame_skip=1):
    seq_dirs = sorted(glob.glob(os.path.join(data_root, "seq_*")))
    print(f"Found {len(seq_dirs)} sequences in {data_root}")

    all_frame_distances = []
    
    print("\n--- Individual Sequence Stats (First 5) ---")
    for i, seq_dir in enumerate(seq_dirs):
        traj_path = os.path.join(seq_dir, "trajectory.npy")
        if not os.path.exists(traj_path):
            continue
            
        # Load trajectory
        traj = np.load(traj_path) # (T, 7)
        positions = traj[:, :3]   # (T, 3)
        
        # Calculate distances between frames separated by frame_skip
        # d[i] = dist(p[i], p[i+frame_skip])
        if len(positions) > frame_skip:
            diffs = positions[frame_skip:] - positions[:-frame_skip]
            dists = np.linalg.norm(diffs, axis=1)
            all_frame_distances.append(dists)
        else:
            dists = np.array([])
        
        if i < 5:
            print(f"Seq: {os.path.basename(seq_dir)}")
            print(f"  Frames: {len(positions)}")
            if len(dists) > 0:
                print(f"  Mean Move (skip={frame_skip}): {np.mean(dists):.4f} mm")
                print(f"  Max Move  (skip={frame_skip}): {np.max(dists):.4f} mm")
            
    if not all_frame_distances:
        print("No trajectory data found or sequences too short for frame_skip.")
        return

    # Aggregate
    all_dists = np.concatenate(all_frame_distances)
    
    print("\n" + "="*40)
    print(f"GLOBAL STATISTICS (Movement per sample, skip={frame_skip})")
    print("="*40)
    print(f"Total Samples Analyzed: {len(all_dists)}")
    print(f"Mean Movement:       {np.mean(all_dists):.4f} mm")
    print(f"Median Movement:     {np.median(all_dists):.4f} mm")
    print(f"Std Dev:             {np.std(all_dists):.4f} mm")
    print("-" * 40)
    print(f"Min Movement:        {np.min(all_dists):.4f} mm")
    print(f"Max Movement:        {np.max(all_dists):.4f} mm")
    print("-" * 40)
    print("Percentiles:")
    print(f"  90th: {np.percentile(all_dists, 90):.4f} mm")
    print(f"  95th: {np.percentile(all_dists, 95):.4f} mm")
    print(f"  99th: {np.percentile(all_dists, 99):.4f} mm")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset/sequences')
    parser.add_argument('--frame_skip', type=int, default=1, help="Frame skipping interval")
    args = parser.parse_args()
    
    analyze_dataset(args.data_root, args.frame_skip)
