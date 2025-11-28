import numpy as np
import torch

# --- DATASET NORMALIZATION STATISTICS ---
# Update these values after running deep_lung_dataset.py on your full dataset!
# Current values based on initial small dataset.
NORM_CENTER = np.array([-19.00057, 9.937485, 17.185833])
NORM_SCALE  = 32.52529

# Helper to get as tensor (useful for models)
def get_norm_center_tensor(device=None):
    return torch.from_numpy(NORM_CENTER).float().to(device)
