import csv
import os

# input_path = "/home/emanuele/Desktop/github/ORB_SLAM3/em/run/videos/datasets/sim/ca_data_b1_1738215363.9776962.csv"
input_path = "/home/emanuele/Desktop/github/ORB_SLAM3/em/run/videos/datasets/sim/"
output_path = "/home/emanuele/Desktop/github/ORB_SLAM3/em/run/videos/datasets/sim/ca_data_b1_1738215363.9776962_mm.csv"

try:
    if os.path.isdir(input_path):
        # Process all CSV files in the directory
        for filename in os.listdir(input_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(input_path, filename)
                # Read the file content
                with open(file_path, "r") as fin:
                    reader = csv.reader(fin)
                    next(reader)  # Skip header
                    rows = [
                        [float(row[0]), float(row[1]), float(row[2]) / 1000]
                        for row in reader
                    ]

                # Write back to the same file
                with open(file_path, "w", newline="") as fout:
                    writer = csv.writer(fout)
                    writer.writerow(["x", "y", "z"])  # Write header
                    writer.writerows(rows)
    else:
        # Process single file
        with open(input_path, "r") as fin:
            reader = csv.reader(fin)
            next(reader)  # Skip header
            rows = [
                [float(row[0]), float(row[1]), float(row[2]) / 1000] for row in reader
            ]

        with open(input_path, "w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(["x", "y", "z"])  # Write header
            writer.writerows(rows)

except FileNotFoundError:
    print("File or directory not found")
except Exception as e:
    print("An error occurred: ", e)
else:
    print("Conversion completed successfully")
