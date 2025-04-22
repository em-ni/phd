import os
import sys
import argparse


def set_timestamps_in_reference(ref_path, traj_path):
    """
    Set timestamps in the reference trajectory based on the recorded trajectory timestamps.
    """

    # Read the recorded trajectory and take the total duration as the difference between the last and first timestamps
    with open(traj_path, "r") as f:
        lines = f.readlines()
        traj_duration = float(lines[-1].split()[0]) - float(lines[0].split()[0])
    f.close()

    print(f"Trajectory duration: {traj_duration:.2f} s")

    # Read the reference trajectory and count the number of lines
    with open(ref_path, "r") as f:
        ref_lines = f.readlines()
        n_ref = len(ref_lines)
    f.close()

    print(f"Reference trajectory has {n_ref} lines")

    # Divide the total duration by the number of lines to get the interval
    ref_interval = traj_duration / (n_ref - 1)

    # New ref file
    # Create a new file by adding "_timestamps" suffix to the original reference file
    new_ref_path = os.path.splitext(ref_path)[0] + "_reference.txt"
    print(f"New reference file: {new_ref_path}")

    # Create the new reference file and Write the new reference trajectory with timestamps (for each line replace the first column with the new timestamp)
    with open(new_ref_path, "w") as f:
        for i in range(n_ref):
            new_timestamp = float(lines[0].split()[0]) + i * ref_interval
            f.write(
                f"{new_timestamp:.6f} {ref_lines[i].split()[1]} {ref_lines[i].split()[2]} {ref_lines[i].split()[3]} {ref_lines[i].split()[4]} {ref_lines[i].split()[5]} {ref_lines[i].split()[6]} {ref_lines[i].split()[7]}\n"
            )
    f.close()
    return new_ref_path


def run_evo_traj_subprocess(traj_path, ref_path, plot_mode="xy", do_plot=True):
    """
    Run the evo_traj subprocess with the new aligned trajectories.
    """
    import subprocess

    # Run evo_traj subprocess
    cmd = f"evo_traj tum {traj_path} {ref_path} -p --plot_mode {plot_mode}"
    if not do_plot:
        cmd += " --no-plots"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare camera trajectory to ground truth."
    )
    parser.add_argument(
        "traj_path",
        type=str,
        help="Path to the recorded trajectory",
    )
    parser.add_argument(
        "ref_path",
        type=str,
        help="Path to the reference trajectory",
    )
    args = parser.parse_args()

    new_ref_path = set_timestamps_in_reference(args.ref_path, args.traj_path)
    run_evo_traj_subprocess(args.traj_path, new_ref_path)
