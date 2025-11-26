import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class DeepLungDataset(Dataset):
    """
    PyTorch Dataset for the Deep-Lung-ST model.
    
    Expects a data root structure like:
    /dataset/
      /sequences/
        /seq_1/
          video.npy      (N, H, W, 3) or (N, 3, H, W)
          trajectory.npy (N, 7) -> [x, y, z, qx, qy, qz, qw]
        /seq_2/
          ...
          
    This dataset performs a sliding window operation to generate overlapping
    sequences of length T (t_frames) for training.
    """
    def __init__(self, data_root, t_frames=16, mode='train', stride=1):
        """
        Args:
            data_root (str): Path to the 'sequences' folder.
            t_frames (int): Length of the temporal window (T).
            mode (str): 'train' or 'val'. Currently affects debug prints.
            stride (int): Step size for sliding window. 
                          1 = Maximum overlap (best for training).
                          t_frames = No overlap (best for validation/inference).
        """
        self.data_root = data_root
        self.t_frames = t_frames
        self.mode = mode
        self.stride = stride
        
        # List to store metadata for every valid window found in the dataset
        # Format: (video_path, traj_path, start_index)
        self.samples = []
        
        self._index_dataset()

    def _index_dataset(self):
        """
        Scans the directory, validates files, and builds the index of valid windows.
        """
        # Find all sequence folders
        seq_dirs = sorted(glob.glob(os.path.join(self.data_root, "seq_*")))
        
        if not seq_dirs:
            print(f"[DATASET] Warning: No sequences found in {self.data_root}")
            return

        print(f"[DATASET] Indexing {self.mode} data from {len(seq_dirs)} sequences (T={self.t_frames}, Stride={self.stride})...")
        
        valid_windows = 0
        skipped_seqs = 0
        
        for seq_dir in seq_dirs:
            if not os.path.isdir(seq_dir):
                continue
                
            # Expected file paths
            vid_path = os.path.join(seq_dir, "video.npy")
            traj_path = os.path.join(seq_dir, "trajectory.npy")
            
            # Validate existence
            if not (os.path.exists(vid_path) and os.path.exists(traj_path)):
                skipped_seqs += 1
                continue
            
            try:
                # Use mmap_mode='r' to read shape without loading the whole file to RAM
                # This makes indexing huge datasets extremely fast
                vid_data = np.load(vid_path, mmap_mode='r')
                traj_data = np.load(traj_path, mmap_mode='r')
                
                N_vid = vid_data.shape[0]
                N_traj = traj_data.shape[0]
                
                # Validate synchronization
                if N_vid != N_traj:
                    print(f"[DATASET] Error: Sync mismatch in {os.path.basename(seq_dir)}. Vid={N_vid}, Traj={N_traj}")
                    skipped_seqs += 1
                    continue
                
                # Generate sliding windows
                # We need at least T frames
                if N_vid >= self.t_frames:
                    # Range is [0, N - T]
                    # E.g., if N=20, T=16, indexes: 0, 1, 2, 3, 4.
                    # Slice 4: 4 to 4+16=20 (End)
                    count_for_seq = 0
                    for start_idx in range(0, N_vid - self.t_frames + 1, self.stride):
                        self.samples.append((vid_path, traj_path, start_idx))
                        count_for_seq += 1
                    valid_windows += count_for_seq
                else:
                    # Sequence too short
                    skipped_seqs += 1
                    
            except Exception as e:
                print(f"[DATASET] Error reading {os.path.basename(seq_dir)}: {e}")
                skipped_seqs += 1

        print(f"[DATASET] Done. Found {len(self.samples)} valid windows.")
        if skipped_seqs > 0:
            print(f"[DATASET] Skipped {skipped_seqs} invalid/short sequences.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'video': tensor (T, 3, 128, 128),
                'gt_pos': tensor (T, 3),
                'gt_rot': tensor (T, 4)
            }
        """
        vid_path, traj_path, start_idx = self.samples[idx]
        
        # 1. Load Data Window using Memory Mapping
        # This reads ONLY the bytes for the specific window from disk
        vid_mmap = np.load(vid_path, mmap_mode='r')
        traj_mmap = np.load(traj_path, mmap_mode='r')
        
        end_idx = start_idx + self.t_frames
        
        # Numpy Slicing
        video_clip = vid_mmap[start_idx : end_idx] # Shape: (T, H, W, 3) usually
        traj_clip  = traj_mmap[start_idx : end_idx] # Shape: (T, 7)
        
        # 2. Convert to Tensor and Float
        # Copy is necessary here because torch doesn't support negative strides 
        # which sometimes happen with mmap slicing, and to detach from file handle.
        video_tensor = torch.from_numpy(np.array(video_clip)).float()
        traj_tensor  = torch.from_numpy(np.array(traj_clip)).float()
        
        # 3. Video Preprocessing
        # Check channel dimension. 
        # BronchoSim saves as (N, H, W, 3)
        # PyTorch Conv2d expects (N, 3, H, W)
        if video_tensor.shape[-1] == 3:
            video_tensor = video_tensor.permute(0, 3, 1, 2)
            
        # Normalize to [-1, 1]
        # Assuming input is [0, 255]
        video_tensor = (video_tensor / 127.5) - 1.0
        
        # 4. Trajectory Splitting
        # traj_clip is [x, y, z, qx, qy, qz, qw]
        gt_pos = traj_tensor[:, :3]
        gt_rot = traj_tensor[:, 3:]
        
        return {
            "video": video_tensor,   # (T, 3, H, W)
            "gt_pos": gt_pos,        # (T, 3)
            "gt_rot": gt_rot         # (T, 4)
        }

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Simple sanity check
    import sys
    
    # Default root for testing
    test_root = "./dataset/sequences"
    if len(sys.argv) > 1:
        test_root = sys.argv[1]
        
    if not os.path.exists(test_root):
        print(f"Path {test_root} does not exist. Run BronchoSim to generate data first.")
        exit()
        
    ds = DeepLungDataset(test_root, t_frames=16)
    
    if len(ds) > 0:
        sample = ds[0]
        print("\nSample 0 Shapes:")
        print(f"  Video: {sample['video'].shape} (Expected: 16, 3, H, W)")
        print(f"  Pose:  {sample['gt_pos'].shape} (Expected: 16, 3)")
        print(f"  Min/Max Video: {sample['video'].min():.2f} / {sample['video'].max():.2f}")
    else:
        print("Dataset empty.")