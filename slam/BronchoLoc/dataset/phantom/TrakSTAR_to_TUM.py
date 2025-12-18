# Convert TrakSTAR .mat data to TUM Trajectory Format

import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

# --- Configuration ---
INPUT_FOLDER = "data"           # Folder containing .mat files
EULER_ORDER = 'ZYX'             # TrakSTAR: A=Yaw(Z), E=Pitch(Y'), R=Roll(X'')
ANGLES_IN_DEGREES = True
CONVERT_TO_METERS = True
INCH_TO_METER = 0.0254


def load_mat_data(mat_path):
    """Load and extract data from .mat file."""
    loaded = loadmat(mat_path)
    
    if 'temp' in loaded:
        return loaded['temp']
    
    # Fallback: find suitable variable
    fields = [k for k in loaded.keys() if not k.startswith('__')]
    if len(fields) == 1:
        return loaded[fields[0]]
    
    for f in fields:
        if isinstance(loaded[f], np.ndarray) and loaded[f].ndim == 2 and loaded[f].shape[1] == 8:
            return loaded[f]
    
    raise ValueError(f'Could not find 8-column data matrix in {mat_path}')


def euler_to_quaternion(angles, order='zyx'):
    """Convert Euler angles to quaternions (TUM format: qx, qy, qz, qw)."""
    rotations = Rotation.from_euler(order, angles)
    return rotations.as_quat()  # Returns [qx, qy, qz, qw]


def convert_mat_to_tum(mat_path, output_path):
    """Convert a single .mat file to TUM format."""
    print(f'Converting: {mat_path}')
    
    data = load_mat_data(mat_path)
    if data.shape[1] != 8:
        raise ValueError(f'Expected 8 columns, found {data.shape[1]}')
    
    # Extract columns: pos(0-2), angles(3-5), timestamp(6), quality(7)
    positions = data[:, 0:3]
    angles = data[:, 3:6]
    timestamps = data[:, 6]
    
    # Convert units
    if CONVERT_TO_METERS:
        positions = positions * INCH_TO_METER
    
    if ANGLES_IN_DEGREES:
        angles = np.deg2rad(angles)
    
    # Convert to quaternions
    quaternions = euler_to_quaternion(angles, EULER_ORDER.lower())
    
    # Write TUM file
    with open(output_path, 'w') as f:
        for i in range(len(timestamps)):
            f.write(f'{timestamps[i]:.6f} '
                    f'{positions[i, 0]:.6f} {positions[i, 1]:.6f} {positions[i, 2]:.6f} '
                    f'{quaternions[i, 0]:.6f} {quaternions[i, 1]:.6f} {quaternions[i, 2]:.6f} {quaternions[i, 3]:.6f}\n')
    
    print(f'  -> Saved: {output_path} ({len(timestamps)} poses)')


def convert_folder(input_folder):
    """Convert all .mat files in a folder to TUM format."""
    mat_pattern = os.path.join(input_folder, '*.mat')
    mat_files = glob.glob(mat_pattern)
    
    if not mat_files:
        print(f'No .mat files found in {input_folder}')
        return
    
    print(f'Found {len(mat_files)} .mat file(s) in {input_folder}\n')
    
    for mat_path in sorted(mat_files):
        basename = os.path.splitext(os.path.basename(mat_path))[0]
        output_path = os.path.join(input_folder, f'{basename}_gt.txt')
        try:
            convert_mat_to_tum(mat_path, output_path)
        except Exception as e:
            print(f'  Error: {e}')
    
    print(f'\nDone. Converted {len(mat_files)} file(s).')


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, INPUT_FOLDER)
    convert_folder(input_folder)
