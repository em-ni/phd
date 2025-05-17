import os
import sys


def convert_inches_to_meters_and_rename(input_filepath):
    """
    Reads a TUM file with positions in inches (expected name *_inches.txt),
    converts positions to meters, and saves it by removing '_inches'
    from the filename (e.g., gt_wTc.txt).

    Args:
        input_filepath (str): Path to the input TUM file (e.g., gt_wTc_inches.txt).

    Returns:
        bool: True if successful, False otherwise.
    """
    inch_to_meter = 0.0254
    original_header_lines = []
    converted_data_lines = []

    print(f"Processing file: {input_filepath}")

    if not input_filepath.endswith("_inches.txt"):
        print(
            f"  Skipping '{input_filepath}': Filename does not end with '_inches.txt'."
        )
        return False

    try:
        with open(input_filepath, "r") as f_in:
            for line_num, line_content in enumerate(f_in):
                stripped_line = line_content.strip()
                if not stripped_line:
                    continue
                if stripped_line.startswith("#"):
                    original_header_lines.append(
                        line_content
                    )  # Keep original line ending
                else:
                    parts = stripped_line.split()
                    if len(parts) == 8:
                        try:
                            timestamp = float(parts[0])
                            tx_inch = float(parts[1])
                            ty_inch = float(parts[2])
                            tz_inch = float(parts[3])
                            qx, qy, qz, qw = [float(p) for p in parts[4:8]]

                            tx_meter = tx_inch * inch_to_meter
                            ty_meter = ty_inch * inch_to_meter
                            tz_meter = tz_inch * inch_to_meter

                            converted_data_lines.append(
                                f"{timestamp:.6f} {tx_meter:.6f} {ty_meter:.6f} {tz_meter:.6f} "
                                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
                            )
                        except ValueError:
                            print(
                                f"  Warning: Could not parse data line {line_num + 1}: '{stripped_line}'. Skipping."
                            )
                    else:
                        print(
                            f"  Warning: Line {line_num + 1} does not have 8 columns: '{stripped_line}'. Skipping."
                        )

        if not converted_data_lines:
            print(
                f"  No valid data lines found or converted in {input_filepath}. Output file will not be created."
            )
            return False

        # Determine output filename by removing "_inches"
        directory = os.path.dirname(input_filepath)
        filename = os.path.basename(input_filepath)

        name_part, ext_part = os.path.splitext(filename)  # ext_part will be ".txt"
        if name_part.endswith("_inches"):
            new_name_part = name_part[: -len("_inches")]  # Remove "_inches"
            output_filename = new_name_part + ext_part
            output_filepath = os.path.join(directory, output_filename)
        else:
            # This case should ideally be caught by the initial check, but as a safeguard:
            print(
                f"  Error: Could not determine output filename for {input_filepath} by removing '_inches'."
            )
            return False

        print(f"  Saving converted data to: {output_filepath} (OVERWRITING if exists)")
        with open(output_filepath, "w") as f_out:

            for data_line in converted_data_lines:
                f_out.write(data_line)

        print(f"  Successfully processed and saved: {output_filepath}")
        return True

    except FileNotFoundError:
        print(f"  Error: Input file not found: {input_filepath}")
        return False
    except Exception as e:
        print(f"  An unexpected error occurred processing file {input_filepath}: {e}")
        return False


if __name__ == "__main__":
    input_files_from_args = sys.argv[1:]

    if not input_files_from_args:
        print(
            "Usage: python this_script_name.py <file1_inches.txt> [file2_inches.txt ...]"
        )
        print("No input files (expected to end with '_inches.txt') provided.")
        sys.exit(1)

    print(f"Processing {len(input_files_from_args)} file(s):")
    for f in input_files_from_args:
        print(f"  - {f}")
    print("-" * 30)

    successful_ops = 0
    failed_ops = 0

    for in_file_path in input_files_from_args:
        if convert_inches_to_meters_and_rename(in_file_path):
            successful_ops += 1
        else:
            failed_ops += 1

    print("-" * 30)
    print(f"Processing complete.")
    print(f"Successfully processed and saved: {successful_ops} file(s).")
    if failed_ops > 0:
        print(f"Failed or skipped: {failed_ops} file(s).")

"""
em/run/datasets/phantom_branches/record_002_1747195986/gt/gt_wTc_inches.txt em/run/datasets/phantom_branches/record_003_1747196017/gt/gt_wTc_inches.txt em/run/datasets/phantom_branches/record_004_1747196049/gt/gt_wTc_inches.txt em/run/datasets/phantom_branches/record_005_1747196071/gt/gt_wTc_inches.txt em/run/datasets/phantom_branches/record_006_1747196079/gt/gt_wTc_inches.txt em/run/datasets/phantom_branches/record_007_1747196104/gt/gt_wTc_inches.txt em/run/datasets/phantom_branches/record_008_1747196123/gt/gt_wTc_inches.txt em/run/datasets/phantom_branches/record_009_1747196143/gt/gt_wTc_inches.txt em/run/datasets/phantom_branches/record_010_1747196160/gt/gt_wTc_inches.txt em/run/datasets/phantom_branches/record_011_1747196176/gt/gt_wTc_inches.txt

"""
