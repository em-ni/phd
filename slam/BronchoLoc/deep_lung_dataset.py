import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 
import os
from tqdm import tqdm
from constants import NORM_CENTER, NORM_SCALE

def get_stats(data_root):
    traj_files = glob.glob(os.path.join(data_root, "seq_*", "trajectory.npy"))
    
    all_mins = []
    all_maxs = []
    
    print(f"[INFO] Scanning {len(traj_files)} sequences...")
    
    for f in tqdm(traj_files):
        # Load only xyz columns (0,1,2)
        data = np.load(f)[:, :3] 
        all_mins.append(data.min(axis=0))
        all_maxs.append(data.max(axis=0))
    
    # Global stats
    global_min = np.min(all_mins, axis=0)
    global_max = np.max(all_maxs, axis=0)
    
    # Calculate scale to fit in [-1, 1]
    # Center = (Max + Min) / 2
    # Scale = (Max - Min) / 2
    center = (global_max + global_min) / 2
    scale = (global_max - global_min) / 2
    
    # Use the largest scale dimension to preserve aspect ratio
    max_scale = scale.max()
    
    # Add 10% padding so we don't hit the edge
    max_scale *= 1.1
    
    print("\n" + "="*40)
    print("DATASET STATISTICS")
    print("="*40)
    print(f"Global Min (mm): {global_min}")
    print(f"Global Max (mm): {global_max}")
    print("-" * 20)
    print(f"Normalization Center: {center}")
    print(f"Normalization Scale:  {max_scale}")
    print("="*40)
    print("\nCopy these values into your constants.py!")

class DeepLungDataset(Dataset):
    """
    PyTorch Dataset for the Deep-Lung-ST model.
    """
    def __init__(self, data_root, t_frames=16, mode='train', stride=1):
        self.data_root = data_root
        self.t_frames = t_frames
        self.mode = mode
        self.stride = stride
        self.samples = []
        
        # --- NORMALIZATION CONSTANTS ---
        # Imported from constants.py
        self.norm_center = torch.from_numpy(NORM_CENTER).float()
        self.norm_scale  = NORM_SCALE
        
        self._index_dataset()

    def _index_dataset(self):
        seq_dirs = sorted(glob.glob(os.path.join(self.data_root, "seq_*")))
        if not seq_dirs: return

        print(f"[DATASET] Indexing {self.mode} data...")
        for seq_dir in seq_dirs:
            if not os.path.isdir(seq_dir): continue
            vid_path = os.path.join(seq_dir, "video.npy")
            traj_path = os.path.join(seq_dir, "trajectory.npy")
            
            if not (os.path.exists(vid_path) and os.path.exists(traj_path)): continue
            
            try:
                vid_data = np.load(vid_path, mmap_mode='r')
                N_vid = vid_data.shape[0]
                
                if N_vid >= self.t_frames:
                    for start_idx in range(0, N_vid - self.t_frames + 1, self.stride):
                        self.samples.append((vid_path, traj_path, start_idx))
            except Exception as e:
                print(f"Error: {e}")

        print(f"[DATASET] Found {len(self.samples)} valid windows.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_path, traj_path, start_idx = self.samples[idx]
        vid_mmap = np.load(vid_path, mmap_mode='r')
        traj_mmap = np.load(traj_path, mmap_mode='r')
        end_idx = start_idx + self.t_frames
        
        video_clip = vid_mmap[start_idx : end_idx] 
        traj_clip  = traj_mmap[start_idx : end_idx].copy()
        
        # Resize to 128x128
        resized_frames = []
        for frame in video_clip:
            if frame.shape[0] == 3: 
                frame = np.transpose(frame, (1, 2, 0))
                frame = cv2.resize(frame, (128, 128))
                frame = np.transpose(frame, (2, 0, 1))
            else:
                frame = cv2.resize(frame, (128, 128))
            resized_frames.append(frame)
        video_clip = np.array(resized_frames)
        
        video_tensor = torch.from_numpy(video_clip).float()
        traj_tensor  = torch.from_numpy(traj_clip).float()
        
        if video_tensor.shape[-1] == 3:
            video_tensor = video_tensor.permute(0, 3, 1, 2)
        
        # Normalize Video to [-1, 1]
        video_tensor = (video_tensor / 127.5) - 1.0
        
        gt_pos = traj_tensor[:, :3]
        
        # --- NORMALIZE TRAJECTORY ---
        # Converts World (mm) -> Network Space (approx -1 to 1)
        gt_pos_norm = (gt_pos - self.norm_center) / self.norm_scale
        
        return {
            "video": video_tensor,
            "gt_pos": gt_pos_norm, # NORMALIZED
            "gt_pos_raw": gt_pos   # RAW
        }

if __name__ == "__main__":
    get_stats("./dataset/sequences")