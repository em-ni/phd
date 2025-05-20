import os
import argparse
import math


def calculate_single_trajectory_length(filepath):
    """
    Calculates the total trajectory length in meters from a TUM format file.
    Each line in the TUM format is expected to be: timestamp tx ty tz qx qy qz qw.
    The length is the sum of Euclidean distances between consecutive camera positions (tx, ty, tz).
    """
    total_distance = 0.0
    previous_pose_translation = None

    try:
        with open(filepath, "r") as f:
            for line_number, line_content in enumerate(f, 1):
                line_content = line_content.strip()
                if not line_content or line_content.startswith(
                    "#"
                ):  # Skip empty or comment lines
                    continue

                parts = line_content.split()
                if len(parts) == 8:
                    try:
                        # timestamp tx ty tz qx qy qz qw
                        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                        current_pose_translation = (tx, ty, tz)

                        if previous_pose_translation:
                            dist = math.sqrt(
                                (
                                    current_pose_translation[0]
                                    - previous_pose_translation[0]
                                )
                                ** 2
                                + (
                                    current_pose_translation[1]
                                    - previous_pose_translation[1]
                                )
                                ** 2
                                + (
                                    current_pose_translation[2]
                                    - previous_pose_translation[2]
                                )
                                ** 2
                            )
                            total_distance += dist

                        previous_pose_translation = current_pose_translation
                    except ValueError:
                        print(
                            f"Warning: Skipping malformed line (non-numeric translation) in {filepath} at line {line_number}: {line_content}"
                        )
                        continue
                else:
                    print(
                        f"Warning: Skipping malformed line (incorrect number of parts) in {filepath} at line {line_number}: {line_content}"
                    )

    except Exception as e:
        print(f"Error reading or processing file {filepath}: {e}")
        return None  # Return None if there was an error reading the file

    return total_distance


def process_gt_files(input_folder):
    """
    Calculates and prints the trajectory length for ground truth files (gt_wTc.txt)
    in subdirectories of the input folder.

    The expected structure is:
    input_folder/
    ├── record_SOMETHING1/
    │   └── gt/
    │       └── gt_wTc.txt
    ...
    └── record_SOMETHINGN/
        └── gt/
            └── gt_wTc.txt
    """
    gt_file_count = 0
    grand_total_length = 0.0

    for record_dir in sorted(os.listdir(input_folder)):  # Sort for consistent order
        record_path = os.path.join(input_folder, record_dir)
        if os.path.isdir(record_path) and record_dir.startswith("record_"):
            gt_file_path = os.path.join(record_path, "gt", "gt_wTc.txt")
            if os.path.isfile(gt_file_path):
                trajectory_length = calculate_single_trajectory_length(gt_file_path)
                if trajectory_length is not None:
                    print(
                        f"File: {gt_file_path}, Trajectory Length: {trajectory_length:.3f} meters"
                    )
                    grand_total_length += trajectory_length
                    gt_file_count += 1
                # If trajectory_length is None, an error message was already printed by calculate_single_trajectory_length
            else:
                print(
                    f"Ground truth file not found in: {os.path.join(record_path, 'gt')}"
                )

    if gt_file_count == 0:
        print(
            f"\nNo ground truth files (gt_wTc.txt) found or processed in the specified structure under {input_folder}"
        )
    else:
        print(f"\nFound and processed {gt_file_count} ground truth file(s).")
        print(
            f"Grand total trajectory length for all processed files: {grand_total_length:.3f} meters."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the trajectory length in meters from ground truth files (gt_wTc.txt) in TUM format."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the input folder containing record_SOMETHING subdirectories.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' not found.")
    else:
        process_gt_files(args.input_folder)
