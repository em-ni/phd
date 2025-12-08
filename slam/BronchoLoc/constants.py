# --- MAP CONFIGURATION ---
MAP_QUERY_RADIUS = 10.0 # mm

# Normalization for Map Points and Targets (Decoupled from Query Radius)
NORM_MAP_SCALE = 10.0 # mm (10.0 means 10mm -> 1.0 normalized)

# Maximum map points to pass to model (after FPS downsampling)
DEFAULT_MAX_MAP_POINTS = 16

# Heuristic threshold for connectivity
# Increased to 2.0mm to be robust against gaps in the centerline point cloud
CONNECTIVITY_THRESHOLD = 2.0 

# --- WINDOW CONFIGURATION ---
# Path to the window config file (created by check_win.py)
import os
WINDOW_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "window_config.json")

def load_window_config():
    """
    Load window_size and frame_skip from the config file.
    Returns (window_size, frame_skip) or raises FileNotFoundError if not set.
    """
    import json
    if not os.path.exists(WINDOW_CONFIG_PATH):
        raise FileNotFoundError(
            f"Window config not found at {WINDOW_CONFIG_PATH}.\n"
            "Please run check_win.py first to set window_size and frame_skip."
        )
    with open(WINDOW_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    return config['window_size'], config['frame_skip']
