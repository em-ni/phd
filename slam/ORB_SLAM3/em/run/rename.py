import os
import sys
import shutil


def rename_remove_meters_suffix(filepath):
    """
    Renames a file by removing '_meters' from its name before the extension.
    Example: 'path/to/gt_wTc_meters.txt' becomes 'path/to/gt_wTc.txt'

    Args:
        filepath (str): The full path to the file to be renamed.

    Returns:
        bool: True if rename was successful, False otherwise.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False

    if not os.path.isfile(filepath):
        print(f"Error: Path is not a file: {filepath}")
        return False

    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    # We expect filenames like 'name_part_meters.ext'
    # We want to change it to 'name_part.ext'

    # Check if "_meters" is actually in the filename part before the extension
    name_part_before_ext, ext_part = os.path.splitext(filename)
    suffix_to_remove = "_meters"

    if name_part_before_ext.endswith(suffix_to_remove):
        # Remove the suffix
        new_name_part = name_part_before_ext[
            : -len(suffix_to_remove)
        ]  # Slicing to remove the suffix
        new_filename = f"{new_name_part}{ext_part}"
        new_filepath = os.path.join(directory, new_filename)

        # Check if the new filename already exists to prevent accidental overwrite
        if os.path.exists(new_filepath):
            print(
                f"Error: Target filename '{new_filename}' already exists in '{directory}'. Skipping rename for '{filename}'."
            )
            return False

        try:
            print(f"Renaming '{filepath}' to '{new_filepath}'")
            shutil.move(filepath, new_filepath)
            print(f"Successfully renamed to '{new_filename}'")
            return True
        except Exception as e:
            print(f"Error renaming '{filepath}': {e}")
            return False
    else:
        print(
            f"Skipping '{filename}': does not end with '{suffix_to_remove}' before the extension."
        )
        return False


if __name__ == "__main__":
    # sys.argv[0] is the script name itself.
    # Actual file arguments start from sys.argv[1]
    input_files_to_rename = sys.argv[1:]

    if not input_files_to_rename:
        print(
            "Usage: python rename_script_remove_meters.py <file1_meters.txt> [file2_meters.txt ...]"
        )
        print("No input files provided on the command line to rename.")
        sys.exit(1)

    print(
        f"Attempting to rename {len(input_files_to_rename)} file(s) by removing '_meters' suffix:"
    )
    for f_path in input_files_to_rename:
        print(f"  - {f_path}")
    print("-" * 30)

    successful_renames = 0
    failed_renames_skipped = 0

    for file_path in input_files_to_rename:
        if rename_remove_meters_suffix(file_path):
            successful_renames += 1
        else:
            failed_renames_skipped += 1

    print("-" * 30)
    print(f"Renaming process complete.")
    print(f"Successfully renamed: {successful_renames} file(s).")
    if failed_renames_skipped > 0:
        print(f"Failed to rename or skipped: {failed_renames_skipped} file(s).")
