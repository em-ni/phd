#!/usr/bin/env python3
"""
Script to prepare logs for evaluation with rpg_trajectory_evaluation
"""

import os
import re
import sys
import shutil
import glob
import yaml


def set_timestamps_in_reference(ref_path, traj_path, dst_gt):
    """
    Set timestamps in the reference (groundtruth) trajectory based on the recorded trajectory timestamps.
    The new groundtruth file is written to dst_gt.
    """
    # Read the recorded trajectory and calculate total duration.
    with open(traj_path, "r") as f:
        traj_lines = f.readlines()
    traj_duration = float(traj_lines[-1].split()[0]) - float(traj_lines[0].split()[0])
    print(f"Trajectory duration: {traj_duration:.2f} s")

    # Read the reference trajectory and count the number of lines.
    with open(ref_path, "r") as f:
        ref_lines = f.readlines()
    n_ref = len(ref_lines)
    print(f"Reference trajectory has {n_ref} lines")

    # Divide the total duration by the number of lines to get the interval.
    ref_interval = traj_duration / (n_ref - 1)

    # Write the new groundtruth file with adjusted timestamps.
    with open(dst_gt, "w") as f:
        for i in range(n_ref):
            new_timestamp = float(traj_lines[0].split()[0]) + i * ref_interval
            # Replace the timestamp in the reference line with the new timestamp.
            f.write(f"{new_timestamp:.6f} " + " ".join(ref_lines[i].split()[1:]) + "\n")
    return dst_gt


def main():
    if len(sys.argv) != 4:
        print("Usage: {} <platform> <algorithm> <slam_logs_folder>".format(sys.argv[0]))
        sys.exit(1)

    platform = sys.argv[1]
    alg = sys.argv[2]
    slam_logs_folder = sys.argv[3]

    # Define output structure:
    # <current working directory>/logs/<platform>/<alg>/<platform>_<alg>_<branch_or_sequence_name>
    output_root = os.path.join(os.getcwd(), "logs", platform)
    alg_folder = os.path.join(output_root, alg)
    os.makedirs(alg_folder, exist_ok=True)

    # Regular expression to match record folders of the form: record_NAME_TIMESTAMP
    record_pattern = re.compile(r"record_(.+)_\d+$")
    # Map identifier -> list of record folder paths
    datasets = {}

    # Iterate over items in the slam_logs_folder to find record folders.
    for item in os.listdir(slam_logs_folder):
        item_path = os.path.join(slam_logs_folder, item)
        if os.path.isdir(item_path):
            match = record_pattern.match(item)
            if match:
                branch_raw = match.group(
                    1
                )  # e.g., "b001" or "real_seq_000_part_1_dif_1"
                # Normalize branch name if it's in "b<digits>" format, otherwise use raw.
                if (
                    branch_raw.startswith("b")
                    and branch_raw[1:].isdigit()
                    and len(branch_raw) > 1
                ):
                    branch_normalized = "b" + str(int(branch_raw[1:]))
                else:
                    branch_normalized = branch_raw
                datasets.setdefault(branch_normalized, []).append(item_path)

    if not datasets:
        print("No record folders found in the slam logs folder.")
        sys.exit(1)

    # Helper function for sorting keys: numeric for "bX", then alphabetical
    def sort_key_func(key_str):
        if key_str.startswith("b") and key_str[1:].isdigit() and len(key_str) > 1:
            return (0, int(key_str[1:]))  # Sort numerically first
        return (1, key_str)  # Sort alphabetically after

    print(
        "\nFound the following dataset groups:\n",
        ", ".join(sorted(datasets.keys(), key=sort_key_func)),
    )

    # Process each group: merge all record folders for that group into one dataset folder.
    for branch, record_folders in sorted(
        datasets.items(), key=lambda item: sort_key_func(item[0])
    ):
        dataset_folder_name = f"{platform}_{alg}_{branch}"
        dataset_folder = os.path.join(alg_folder, dataset_folder_name)
        os.makedirs(dataset_folder, exist_ok=True)

        traj_count = 0  # counter for numbering stamped_traj_estimate files
        gt_copied = False
        first_traj_path = (
            None  # store first valid trajectory file for timestamp reference
        )

        print("Processing branch", branch)

        for record_folder in record_folders:
            # Look for all CameraTrajectory files in the record folder's logs subfolder.
            traj_files = glob.glob(
                os.path.join(record_folder, "logs", "CameraTrajectory_*.txt")
            )
            if not traj_files:
                print(
                    f"No CameraTrajectory file found in {record_folder}. Skipping this record folder."
                )
                continue
            traj_files = sorted(traj_files)
            for traj_file in traj_files:
                dst_traj = os.path.join(
                    dataset_folder, f"stamped_traj_estimate{traj_count}.txt"
                )
                shutil.copy2(traj_file, dst_traj)
                print(f"Copied trajectory file from {traj_file} to {dst_traj}")
                if first_traj_path is None:
                    first_traj_path = dst_traj
                traj_count += 1

            print(f"Processed {len(traj_files)} trajectory files in {record_folder}")

            # Copy the ground truth from the first record folder that contains it.
            if not gt_copied:
                gt_file = os.path.join(record_folder, "gt", "gt_wTc.txt")
                if os.path.exists(gt_file):
                    dst_gt = os.path.join(dataset_folder, "stamped_groundtruth.txt")
                    # Instead of a simple copy, update the timestamp column using the first trajectory file.
                    if first_traj_path is not None:
                        set_timestamps_in_reference(gt_file, first_traj_path, dst_gt)
                        print(
                            f"Adjusted and copied ground truth file from {gt_file} to {dst_gt}"
                        )
                    else:
                        # Fallback: simple copy if no trajectory file is available
                        shutil.copy2(gt_file, dst_gt)
                        print(f"Copied ground truth file from {gt_file} to {dst_gt}")
                    gt_copied = True
                else:
                    print(
                        f"Ground truth file not found in {record_folder}/gt. Skipping ground truth copy for this record folder."
                    )

        if traj_count == 0:
            print(
                f"No trajectory files copied for branch {branch}. Removing empty dataset folder {dataset_folder}."
            )
            shutil.rmtree(dataset_folder)
            continue

        # Create eval_cfg.yaml with the specified content in the dataset folder.
        eval_cfg = {"align_type": "sim3", "align_num_frames": -1}
        eval_cfg_path = os.path.join(dataset_folder, "eval_cfg.yaml")
        with open(eval_cfg_path, "w") as f:
            yaml.dump(eval_cfg, f, default_flow_style=False)
        print(f"Created evaluation config file at {eval_cfg_path}\n")

    # Create the configuration file for multiple trajectory evaluation.
    # Note: The expected algorithm function key is "traj_est"
    config_data = {
        "Datasets": {},
        "Algorithms": {alg: {"fn": "traj_est", "label": alg}},
        "RelDistances": [],
        "RelDistancePercentages": [],
    }
    # Add each branch/sequence as a dataset entry.
    # Use the same sorting key function for consistent ordering.
    for branch in sorted(datasets.keys(), key=sort_key_func):
        config_data["Datasets"][branch] = {"label": branch}

    # Write the configuration file to the root of the <platform> folder.
    config_filename = f"{platform}_{alg}_config.yaml"
    config_path = os.path.join(output_root, config_filename)
    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)
    print(f"\nCreated configuration file at {config_path}\n")

    print("\nDataset preparation completed.")
    print(
        f"- Copy the {config_filename} into rpg_trajectory_evaluation/analyze_trajectories_config"
    )
    print(
        f"- Copy the newly created folder under {output_root}/ into the rpg_trajectory_evaluation/em/broncho."
    )
    print("- Run the evaluation script with the following command:\n")
    mul_trials = len(traj_files)
    print(
        f"python scripts/analyze_trajectories.py {config_filename} --output_dir=./em/broncho --results_dir ./em/broncho --platform {platform} --odometry_error_per_dataset --plot_trajectories --rmse_table --rmse_boxplot --mul_trials={mul_trials}"
    )


if __name__ == "__main__":
    main()
