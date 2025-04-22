#!/usr/bin/env python3

import argparse
import numpy as np
from scipy.spatial.transform import Rotation
import os


def convert_fs_to_tum(self, input_file, output_file, convention="wTc"):
    """
    Convert Frenet-Serret frames (px, py, pz, Tx, Ty, Tz, Nx, Ny, Nz, Bx, By, Bz)
    into TUM format:

        timestamp px py pz qx qy qz qw

    To get the series of w_T_ci matrices of the ground truth trajectory

    o is the origin frame (CAD origin)
    w is the world frame (SLAM origin)
    The w frame has to correspond to the first FS frame in the centerline

    The matrix saved in the FS frame file is built as
    o_T_fsi = [Tx_i, Nx_i, Bx_i, px_i]
                [Ty_i, Ny_i, By_i, py_i]
                [Tz_i, Nz_i, Bz_i, pz_i]
                [0, 0, 0, 1]
    so is the transformation from the i-th fs frame to the origin frame

    To align the fs frame to the camera convention, rotate the frame 90 degrees around the n axis
    (t forward, n down, b left) -> (z forward, x right, y down)
    R_n(90) =   [0, 0, 1]
                [0, 1, 0]
                [-1, 0, 0]

    fsi_T_ci =  [0, 0, 1, 0]
                [0, 1, 0, 0]
                [-1, 0, 0, 0]
                [0, 0, 0, 1]

    Then
    o_T_ci = o_T_fsi * fsi_T_ci

    The first point of the centerline (i = 0) corresponds to the transformation from the world frame to the origin frame
    o_T_w = o_T_fs0 * fs0_T_c0

    The camera trajectory of the SLAM is saved as a series of Twc (w_T_ci) transformations,
    so to get the ground truth data from here I have to obtain all the Twc_i (w_T_ci) matrices

    w_T_ci = o_T_w^-1 * o_T_ci


    """

    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        lines = fin.readlines()
        first = True
        o_T_w = np.eye(4)
        w_T_o = np.eye(4)

        for i, line in enumerate(lines):
            # Each line has 12 floats: px, py, pz, Tx, Ty, Tz, Nx, Ny, Nz, Bx, By, Bz
            vals = line.strip().split(",")
            if len(vals) != 12:
                continue

            # Parse floats
            px, py, pz = map(float, vals[0:3])
            # x = x / 1000.0
            # y = y / 1000.0
            # z = z / 1000.0
            tx, ty, tz = map(float, vals[3:6])
            nx, ny, nz = map(float, vals[6:9])
            bx, by, bz = map(float, vals[9:12])

            # Build rotation matrix as in file: t forward, n down, b left
            o_R_fsi = np.array([[tx, nx, bx], [ty, ny, by], [tz, nz, bz]])

            # Check determinant and adjust if necessary
            if np.linalg.det(o_R_fsi) < 0:
                print("Determinant is negative. Adjusting rotation matrix.")
                continue

            # Verify orthogonality
            RtR = o_R_fsi.T @ o_R_fsi
            I = np.eye(3)
            error = RtR - I
            max_error = np.abs(error).max()
            if max_error > 1e-6:
                print(f"Rotation matrix not orthogonal enough. Max error: {max_error}")
                continue

            # Build o_T_fsi (FS frame as originally set)
            o_T_fsi = np.eye(4)
            o_T_fsi[:3, :3] = o_R_fsi
            o_T_fsi[:3, 3] = [px, py, pz]

            # Apply rotation to match camera convention build fsi_T_ci (from FS frame to camera frame)
            # Rotate 90 degrees around n axis
            Rn = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            fsi_T_ci = np.eye(4)
            fsi_T_ci[:3, :3] = Rn
            fsi_T_ci[:3, 3] = [0, 0, 0]

            # Build o_T_ci = o_T_fsi * fsi_T_ci (from origin to camera frame)
            o_T_ci = o_T_fsi @ fsi_T_ci

            # Handle first frame
            if first:
                o_T_w = o_T_ci
                w_T_o = np.linalg.inv(o_T_w)
                first = False

            # Compute w_T_ci = (o_T_w)^-1 * o_T_ci
            w_T_ci = w_T_o @ o_T_ci
            w_R_ci = w_T_ci[:3, :3]

            if convention == "wTc":
                final = w_T_ci
                final_rot = w_R_ci
            elif convention == "cTw":
                final = np.linalg.inv(w_T_ci)
                final_rot = final[:3, :3]

            # Convert to quaternion
            rot = Rotation.from_matrix(final_rot.astype(np.float64))
            qx, qy, qz, qw = rot.as_quat()

            # Write TUM format: timestamp px py pz qx qy qz qw
            timestamp = float(i)
            fout.write(
                f"{timestamp:.6f} {final[0, 3]:.6f} {final[1, 3]:.6f} {final[2, 3]:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Convert Frenet-Serret frames to TUM trajectory format."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input .txt file or folder containing .txt files.",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the output folder for TUM-format trajectories.",
    )
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    if os.path.isfile(args.input_path) and args.input_path.endswith(".txt"):
        # Single file processing
        basename = os.path.splitext(os.path.basename(args.input_path))[0]
        output_file = os.path.join(args.output_folder, f"{basename}_gt.txt")
        convert_fs_to_tum(args.input_path, output_file)
        print(f"Converted {args.input_path} -> {output_file}")

    elif os.path.isdir(args.input_path):
        # Process all .txt files in the folder
        for filename in os.listdir(args.input_path):
            if filename.endswith(".txt"):
                input_file = os.path.join(args.input_path, filename)
                basename = os.path.splitext(filename)[0]
                output_file = os.path.join(args.output_folder, f"{basename}_gt.txt")
                convert_fs_to_tum(input_file, output_file)
                print(f"Converted {input_file} -> {output_file}")

    else:
        print("Error: Input path must be a .txt file or a folder containing .txt files")


if __name__ == "__main__":
    main()


# /home/emanuele/Desktop/github/ORB_SLAM3/em/run/centerline_frames/sim/b21/b21.txt
# /home/emanuele/Desktop/github/ORB_SLAM3/em/logs/b21_gt.txt
