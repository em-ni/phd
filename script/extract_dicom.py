import os
import pydicom
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support

# Read the DICOMDIR file
dicomdir = pydicom.dcmread("../data/skull_dicom/DICOMDIR")
base_dir = os.path.dirname("../data/skull_dicom/DICOMDIR")

series_files = {}
for patient_record in dicomdir.patient_records:
    for study in patient_record.children:
        for series in study.children:
            if series.SeriesNumber not in series_files:
                series_files[series.SeriesNumber] = []
            for image in series.children:
                if hasattr(image, "ReferencedFileID"):
                    file_path = os.path.join(base_dir, *image.ReferencedFileID)
                    series_files[series.SeriesNumber].append(file_path)

# Create output directory
output_dir = os.path.join(base_dir, "extracted_data")
os.makedirs(output_dir, exist_ok=True)

# Process each series individually
for series_number, files in series_files.items():
    print(f"Processing series {series_number} with {len(files)} files.")

    if len(files) < 2:  # Change this to your desired minimum number of slices
        print(f"Series {series_number} has less than 2 slices, skipping.")
        continue

    # Read the DICOM files
    datasets = [pydicom.dcmread(file) for file in files]

    # Convert the DICOM image data to a volume file
    volume_data = np.stack([ds.pixel_array for ds in datasets])
    volume_image = sitk.GetImageFromArray(volume_data)
    sitk.WriteImage(volume_image, os.path.join(output_dir, f"series_{series_number}_volume.mhd"))

    # Extract a surface mesh from the DICOM image data
    # Convert the volume data to a VTK image
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(volume_data.shape)
    vtk_image.SetSpacing([1.0, 1.0, 1.0])  # Adjust this if your DICOM files have different spacing
    vtk_image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(volume_data.ravel(), deep=True, array_type=vtk.VTK_FLOAT))

    # Use marching cubes to extract a surface
    mcubes = vtk.vtkMarchingCubes()
    mcubes.SetInputData(vtk_image)
    mcubes.ComputeNormalsOn()
    mcubes.SetValue(0, 100)  # Set this to the appropriate threshold value for your data

    # Write the surface to an OBJ file
    writer = vtk.vtkOBJWriter()
    writer.SetInputConnection(mcubes.GetOutputPort())
    writer.SetFileName(os.path.join(output_dir, f"series_{series_number}_surface.obj"))
    writer.Write()
