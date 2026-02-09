#! /usr/bin/env python

"""
Standalone Airway Processing Pipeline
=====================================

This script performs the complete workflow for extracting and repairing airways from CT scans.

Pipeline Steps:
1.  Load DICOM series.
2.  Extract airways using Tubular Enhancement (Frangi Vesselness).
3.  Convert segmentation to 3D Mesh (STL).
4.  Repair topology using Contact-Based Smart Repair (Genus 0, Watertight).
5.  Save processed mesh.

Usage:
    python standalone_pipeline.py --input <dicom_folder> --output <output_folder>
    python standalone_pipeline.py --batch --root <archive_root>
"""

import sys
import os
import glob
import time
import fnmatch
import math
import argparse
import traceback

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import trimesh
from scipy import ndimage
import pyvista as pv

# ==========================================
# SECTION 1: UTILITIES (Inlined from dicom2stl)
# ==========================================

# --- dicomutils.py ---

from pydicom.filereader import read_file_meta_info
from pydicom.errors import InvalidDicomError

def testDicomFile(file_path):
    try:
        read_file_meta_info(file_path)
        return True
    except InvalidDicomError:
        return False

def scanDirForDicom(dicomdir):
    matches = []
    found_dirs = []
    try:
        for root, _, filenames in os.walk(dicomdir):
            for filename in fnmatch.filter(filenames, "*.dcm"):
                matches.append(os.path.join(root, filename))
                if root not in found_dirs:
                    found_dirs.append(root)
    except OSError as e:
        print("Error in scanDirForDicom: ", e)
    return (matches, found_dirs)

def getAllSeries(target_dirs):
    isr = sitk.ImageSeriesReader()
    found_series = []
    for d in target_dirs:
        series = isr.GetGDCMSeriesIDs(d)
        for s in series:
            found_files = isr.GetGDCMSeriesFileNames(d, s)
            found_series.append([s, d, found_files])
    return found_series

def getModality(img):
    modality = ""
    if (sitk.Version.MinorVersion() > 8) or (sitk.Version.MajorVersion() > 0):
        try:
            modality = img.GetMetaData("0008|0060")
        except RuntimeError:
            modality = ""
    return modality

def loadLargestSeries(dicomdir):
    files, dirs = scanDirForDicom(dicomdir)
    if (len(files) == 0) or (len(dirs) == 0):
        # Fallback: check if dir itself contains dcm files without recursion
        files = glob.glob(os.path.join(dicomdir, "*.dcm"))
        if files:
            dirs = [dicomdir]
        else:
            print("Error in loadLargestSeries. No files found.")
            return None, None

    seriessets = getAllSeries(dirs)
    maxsize = 0
    maxindex = -1
    count = 0
    for ss in seriessets:
        size = len(ss[2])
        if size > maxsize:
            maxsize = size
            maxindex = count
        count = count + 1
    if maxindex < 0:
        print("Error: no series found")
        return None, None
        
    isr = sitk.ImageSeriesReader()
    ss = seriessets[maxindex]
    files = ss[2]
    isr.SetFileNames(files)
    print(f"Loading series {ss[0]} in directory {ss[1]} ({len(files)} slices)")
    img = isr.Execute()
    firstslice = sitk.ReadImage(files[0])
    modality = getModality(firstslice)
    return img, modality

# --- sitk2vtk.py ---

def sitk2vtk(img, debugOn=False):
    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    ncomp = img.GetNumberOfComponentsPerPixel()
    direction = img.GetDirection()
    i2 = sitk.GetArrayFromImage(img)
    vtk_image = vtk.vtkImageData()
    if len(size) == 2: size.append(1)
    if len(origin) == 2: origin.append(0.0)
    if len(spacing) == 2: spacing.append(spacing[0])
    if len(direction) == 4:
        direction = [direction[0], direction[1], 0.0, direction[2], direction[3], 0.0, 0.0, 0.0, 1.0]
    vtk_image.SetDimensions(size)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
    if vtk.vtkVersion.GetVTKMajorVersion() >= 9:
        vtk_image.SetDirectionMatrix(direction)
    depth_array = numpy_support.numpy_to_vtk(i2.ravel())
    depth_array.SetNumberOfComponents(ncomp)
    vtk_image.GetPointData().SetScalars(depth_array)
    vtk_image.Modified()
    return vtk_image

# --- vtkutils.py ---

def elapsedTime(start_time):
    dt = time.perf_counter() - start_time
    # print(f"    {dt:4.3f} seconds")

def extractSurface(vol, isovalue=0.0):
    try:
        t = time.perf_counter()
        iso = vtk.vtkContourFilter()
        if vtk.vtkVersion.GetVTKMajorVersion() >= 6:
            iso.SetInputData(vol)
        else:
            iso.SetInput(vol)
        iso.SetValue(0, isovalue)
        iso.Update()
        mesh = iso.GetOutput()
        elapsedTime(t)
        return mesh
    except RuntimeError:
        print("Iso-surface extraction failed")
        return None

def cleanMesh(mesh, connectivityFilter=False):
    try:
        t = time.perf_counter()
        connect = vtk.vtkPolyDataConnectivityFilter()
        clean = vtk.vtkCleanPolyData()
        if connectivityFilter:
            if vtk.vtkVersion.GetVTKMajorVersion() >= 6:
                connect.SetInputData(mesh)
            else:
                connect.SetInput(mesh)
            connect.SetExtractionModeToLargestRegion()
            clean.SetInputConnection(connect.GetOutputPort())
        else:
            if vtk.vtkVersion.GetVTKMajorVersion() >= 6:
                clean.SetInputData(mesh)
            else:
                clean.SetInput(mesh)
        clean.Update()
        m2 = clean.GetOutput()
        elapsedTime(t)
        return m2
    except RuntimeError:
        print("Surface cleaning failed")
        return None

def smoothMesh(mesh, nIterations=10):
    try:
        t = time.perf_counter()
        smooth = vtk.vtkWindowedSincPolyDataFilter()
        smooth.SetNumberOfIterations(nIterations)
        if vtk.vtkVersion.GetVTKMajorVersion() >= 6:
            smooth.SetInputData(mesh)
        else:
            smooth.SetInput(mesh)
        smooth.Update()
        m2 = smooth.GetOutput()
        elapsedTime(t)
        return m2
    except RuntimeError:
        print("Surface smoothing failed")
        return None

def writeMesh(mesh, name):
    if name.endswith(".vtk"): writeVTKMesh(mesh, name)
    elif name.endswith(".ply"): writePLY(mesh, name)
    elif name.endswith(".stl"): writeSTL(mesh, name)
    else: print("Unknown file type: ", name)

def writeVTKMesh(mesh, name):
    writer = vtk.vtkPolyDataWriter()
    if vtk.vtkVersion.GetVTKMajorVersion() >= 6: writer.SetInputData(mesh)
    else: writer.SetInput(mesh)
    writer.SetFileTypeToBinary()
    writer.SetFileName(name)
    writer.Write()

def writeSTL(mesh, name):
    writer = vtk.vtkSTLWriter()
    if vtk.vtkVersion.GetVTKMajorVersion() >= 6: writer.SetInputData(mesh)
    else: writer.SetInput(mesh)
    writer.SetFileTypeToBinary()
    writer.SetFileName(name)
    writer.Write()

def writePLY(mesh, name):
    writer = vtk.vtkPLYWriter()
    if vtk.vtkVersion.GetVTKMajorVersion() >= 6: writer.SetInputData(mesh)
    else: writer.SetInput(mesh)
    writer.SetFileTypeToBinary()
    writer.SetFileName(name)
    writer.Write()

# ==========================================
# SECTION 2: AIRWAY EXTRACTION (Tubular)
# ==========================================

def compute_frangi_manual(img, sigma, alpha=0.5, beta=0.5, gamma=5.0):
    print(f"Computing Hessian Manual (Sigma={sigma})...")
    
    def apply_gauss(image, direction, order):
        filt = sitk.RecursiveGaussianImageFilter()
        filt.SetSigma(sigma)
        filt.SetDirection(direction)
        filt.SetOrder(order)
        return filt.Execute(image)

    ixx = apply_gauss(apply_gauss(apply_gauss(img, 0, 2), 1, 0), 2, 0)
    iyy = apply_gauss(apply_gauss(apply_gauss(img, 0, 0), 1, 2), 2, 0)
    izz = apply_gauss(apply_gauss(apply_gauss(img, 0, 0), 1, 0), 2, 2)
    ixy = apply_gauss(apply_gauss(apply_gauss(img, 0, 1), 1, 1), 2, 0)
    ixz = apply_gauss(apply_gauss(apply_gauss(img, 0, 1), 1, 0), 2, 1)
    iyz = apply_gauss(apply_gauss(apply_gauss(img, 0, 0), 1, 1), 2, 1)
    
    print("Converting to numpy for Eigenvalue computation (Chunked)...")
    a_xx = sitk.GetArrayViewFromImage(ixx)
    a_yy = sitk.GetArrayViewFromImage(iyy)
    a_zz = sitk.GetArrayViewFromImage(izz)
    a_xy = sitk.GetArrayViewFromImage(ixy)
    a_xz = sitk.GetArrayViewFromImage(ixz)
    a_yz = sitk.GetArrayViewFromImage(iyz)
    
    depth = a_xx.shape[0]
    height = a_xx.shape[1]
    width = a_xx.shape[2]
    
    V_final = np.zeros((depth, height, width), dtype=np.float32)
    
    chunk_size = 10 
    
    for i in range(0, depth, chunk_size):
        end = min(i + chunk_size, depth)
        
        c_xx = a_xx[i:end]
        c_yy = a_yy[i:end]
        c_zz = a_zz[i:end]
        c_xy = a_xy[i:end]
        c_xz = a_xz[i:end]
        c_yz = a_yz[i:end]
        
        shape = c_xx.shape
        H = np.zeros(shape + (3, 3), dtype=np.float32)
        H[..., 0, 0] = c_xx
        H[..., 1, 1] = c_yy
        H[..., 2, 2] = c_zz
        H[..., 0, 1] = H[..., 1, 0] = c_xy
        H[..., 0, 2] = H[..., 2, 0] = c_xz
        H[..., 1, 2] = H[..., 2, 1] = c_yz
        
        evals, _ = np.linalg.eigh(H)
        idx = np.argsort(np.abs(evals), axis=-1)
        evals = np.take_along_axis(evals, idx, axis=-1)
        
        lambda1 = evals[..., 0]
        lambda2 = evals[..., 1]
        lambda3 = evals[..., 2]
        
        epsilon = 1e-10
        Ra = np.abs(lambda2) / (np.abs(lambda3) + epsilon)
        Rb = np.abs(lambda1) / (np.sqrt(np.abs(lambda2 * lambda3)) + epsilon)
        S = np.sqrt(lambda1**2 + lambda2**2 + lambda3**2)
        
        term1 = 1 - np.exp(-(Ra**2) / (2 * alpha**2))
        term2 = np.exp(-(Rb**2) / (2 * beta**2))
        term3 = 1 - np.exp(-(S**2) / (2 * gamma**2))
        
        V = term1 * term2 * term3
        condition = (lambda2 < 0) & (lambda3 < 0)
        V[~condition] = 0
        
        V_final[i:end] = V
        del H, evals, V
    
    vesselness_img = sitk.GetImageFromArray(V_final)
    vesselness_img.CopyInformation(img)
    return vesselness_img

def extract_airways_tubular(img, sigma=1.0, alpha=0.5, beta=0.5, gamma=100.0, vessel_thresh=0.01, air_thresh=-800):
    print("Preprocessing...")
    img_float = sitk.Cast(img, sitk.sitkFloat32)
    img_inv = img_float * -1.0
    
    vesselness = compute_frangi_manual(img_inv, sigma, alpha, beta, gamma)
    
    print(f"Thresholding Vesselness > {vessel_thresh}...")
    vessel_mask = sitk.BinaryThreshold(vesselness, lowerThreshold=vessel_thresh, upperThreshold=100.0, insideValue=1, outsideValue=0)
    
    print(f"Masking with Air < {air_thresh} HU...")
    air_mask = sitk.BinaryThreshold(img, lowerThreshold=-3000, upperThreshold=air_thresh, insideValue=1, outsideValue=0)
    
    final_mask = sitk.And(vessel_mask, air_mask)
    
    print("Keeping largest component...")
    final_mask = sitk.RelabelComponent(sitk.ConnectedComponent(final_mask))
    final_mask = sitk.BinaryThreshold(final_mask, 1, 1, 1, 0)
    
    return final_mask

# ==========================================
# SECTION 3: TOPOLOGICAL REPAIR (Contact-Based)
# ==========================================

def get_contact_ratio(component, base_matrix):
    struct = ndimage.generate_binary_structure(3, 1)
    dilated = ndimage.binary_dilation(component, structure=struct)
    neighbors = dilated & ~component
    contact_voxels = neighbors & base_matrix
    n_contact = np.sum(contact_voxels)
    exposed_voxels = neighbors & ~base_matrix
    n_exposed = np.sum(exposed_voxels)
    
    if n_exposed == 0:
        return n_contact, 0, float('inf')
    return n_contact, n_exposed, n_contact / n_exposed

def smart_repair(mesh, target_euler=2, smoothing_iters=100):
    print(f"[Repair] Starting Smart Repair. Initial Euler: {mesh.euler_number}")
    
    if not mesh.is_watertight:
        print("[Repair] Input not watertight. Voxelizing...")
        mesh = mesh.voxelized(pitch=mesh.extents.max()/300).marching_cubes

    length = mesh.extents.max()
    pitch = length / 400.0
    print(f"[Voxel] Voxelizing with pitch {pitch:.4f}...")
    
    voxelized = mesh.voxelized(pitch=pitch)
    voxelized.fill()
    
    if hasattr(voxelized.encoding, 'dense'):
        base_matrix = voxelized.encoding.dense
    else:
        base_matrix = voxelized.matrix
    if not isinstance(base_matrix, np.ndarray):
        base_matrix = np.array(base_matrix)
    base_matrix = base_matrix.astype(bool)
    
    current_matrix = base_matrix.copy()
    
    i = 0
    while True:
        i += 1
        closing_size = i * 2 
        
        if (closing_size * pitch) > length:
             print("[Stop] Kernel size exceeds mesh size. Stopping.")
             break
             
        print(f"  [Iter {i}] Closing Size: {closing_size} (approx {closing_size*pitch:.2f}mm)")
        
        closed = ndimage.binary_dilation(current_matrix, iterations=closing_size)
        closed = ndimage.binary_erosion(closed, iterations=closing_size)
        
        added_voxels = closed & ~current_matrix
        labeled, num_features = ndimage.label(added_voxels)
        
        if num_features == 0:
            continue
            
        kept_plugs_mask = np.zeros_like(added_voxels, dtype=bool)
        kept_count = 0
        objects = ndimage.find_objects(labeled)
        
        for idx, slices in enumerate(objects):
            if slices is None: continue
            s_expanded = tuple(slice(max(0, s.start-1), min(d, s.stop+1)) for s, d in zip(slices, base_matrix.shape))
            local_label = labeled[s_expanded]
            local_component = (local_label == (idx + 1))
            local_base = current_matrix[s_expanded]
            n_c, n_e, ratio = get_contact_ratio(local_component, local_base)
            
            if ratio > 1.2:
                kept_plugs_mask[s_expanded] |= local_component
                kept_count += 1

        print(f"    -> Kept {kept_count}/{num_features} plugs.")
        
        if kept_count == 0:
            continue
            
        current_matrix |= kept_plugs_mask
        
        temp_vox = trimesh.voxel.VoxelGrid(current_matrix, transform=voxelized.transform)
        temp_mesh = temp_vox.marching_cubes
        euler = temp_mesh.euler_number
        print(f"    -> New Euler: {euler}")
        
        if euler >= target_euler:
            print("[Success] Target Euler achieved!")
            mesh = temp_mesh
            break
            
        mesh = temp_mesh

    print(f"[Post] Smoothing (Taubin, {smoothing_iters} iterations)...")
    trimesh.smoothing.filter_taubin(mesh, iterations=smoothing_iters)
    
    # Final Watertight Check
    if not mesh.is_watertight:
        print("[Post] Mesh not watertight. Repairing...")
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_winding(mesh)
        
    return mesh

# ==========================================
# SECTION 4: VISUALIZATION
# ==========================================

def visualize_files(files):
    files = [f for f in files if os.path.exists(f)]
    files.sort()
    
    if not files:
        print("No valid files found for visualization.")
        return

    n_files = len(files)
    cols = int(math.ceil(math.sqrt(n_files)))
    rows = int(math.ceil(n_files / cols))
    
    print(f"Visualizing {n_files} files in a {rows}x{cols} grid.")
    plotter = pv.Plotter(shape=(rows, cols))

    for i, input_file in enumerate(files):
        r = i // cols
        c = i % cols
        plotter.subplot(r, c)
        try:
            mesh = pv.read(input_file)
            plotter.add_mesh(mesh)
            plotter.add_text(os.path.basename(input_file), font_size=10)
            plotter.reset_camera()
        except Exception as e:
            print(f"Error reading {input_file}: {e}")
            plotter.add_text(f"Error:\n{os.path.basename(input_file)}", font_size=8, color='red')

    plotter.show()

# ==========================================
# SECTION 5: MAIN EXECUTION
# ==========================================

import multiprocessing

def process_combination_safe(img, folder_name, scan_output_dir, sigma, thresh, gamma, count, total_combinations):
    # Naming convention: 1003/1003_s1.0_t0.01_g100.stl
    filename = f"{folder_name}_s{sigma}_t{thresh}_g{int(gamma)}.stl"
    output_file = os.path.join(scan_output_dir, filename)
    
    print(f"\n--- Combination {count}/{total_combinations}: Sigma={sigma}, Thresh={thresh}, Gamma={gamma} ---")
    
    if os.path.exists(output_file):
        print(f"[Skip] {output_file} already exists.")
        return

    try:
        # 2. Extract Airways
        print("[Step 1] Extracting Airways...")
        mask = extract_airways_tubular(img, sigma=sigma, vessel_thresh=thresh, gamma=gamma)
        
        # Check if mask is empty
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask)
        if stats.GetNumberOfLabels() == 0:
            print("[Warning] Extraction resulted in empty mask. Skipping.")
            return

        # Convert to Mesh
        print("  -> Converting to mesh...")
        vtkimg = sitk2vtk(mask)
        mesh_vtk = extractSurface(vtkimg, isovalue=0.5)
        if mesh_vtk is None or mesh_vtk.GetNumberOfPoints() == 0:
             print("[Warning] Mesh extraction failed or empty. Skipping.")
             return
             
        mesh_vtk = cleanMesh(mesh_vtk)
        mesh_vtk = smoothMesh(mesh_vtk, nIterations=10)
        
        # Save temp STL to load into Trimesh
        temp_stl = f"temp_{folder_name}_{count}.stl"
        writeSTL(mesh_vtk, temp_stl)
        
        # 3. Repair Topology
        print("[Step 2] Repairing Topology...")
        mesh_tri = trimesh.load(temp_stl)
        
        # Skip repair if mesh is empty (trimesh might load it as empty Scene or empty Trimesh)
        if isinstance(mesh_tri, trimesh.Scene):
            if len(mesh_tri.geometry) == 0:
                 print("[Warning] Trimesh loaded empty scene. Skipping.")
                 if os.path.exists(temp_stl): os.remove(temp_stl)
                 return
            mesh_tri = mesh_tri.dump(concatenate=True)
        
        if len(mesh_tri.vertices) == 0:
             print("[Warning] Empty mesh. Skipping.")
             if os.path.exists(temp_stl): os.remove(temp_stl)
             return

        mesh_fixed = smart_repair(mesh_tri, target_euler=2, smoothing_iters=100)
        
        # 4. Save
        print(f"[Save] Saving to {output_file}")
        mesh_fixed.export(output_file)
        
        if os.path.exists(temp_stl):
            os.remove(temp_stl)
            
    except Exception as e:
        print(f"[Error] Failed processing combination s{sigma}_t{thresh}_g{gamma}: {e}")
        traceback.print_exc()

# Specific Parameter Triplets (Sigma, Thresh, Gamma)
# Extracted from user image (red dots)
TRIPLETS = [
    (0.8, 0.001, 100.0),
    (0.8, 0.001, 300.0),
    (0.8, 0.001, 500.0),
    (0.8, 0.01, 100.0),
    (1.0, 0.001, 100.0),
    (1.0, 0.001, 300.0),
    (1.0, 0.001, 500.0),
    (1.0, 0.01, 300.0),
    (1.2, 0.0005, 100.0),
    (1.2, 0.001, 300.0),
    (1.2, 0.001, 500.0),
    (1.2, 0.01, 100.0),
    (1.5, 0.0005, 100.0),
    (1.5, 0.0005, 300.0),
    (1.5, 0.005, 100.0),
    (1.5, 0.005, 300.0)
]

def process_combination_subprocess(folder_path, folder_name, scan_output_dir, sigma, thresh, gamma, count, total_combinations):
    """Wrapper that loads DICOM in the subprocess to avoid pickle issues on Windows."""
    # Check if already exists first (skip early)
    filename = f"{folder_name}_s{sigma}_t{thresh}_g{int(gamma)}.stl"
    output_file = os.path.join(scan_output_dir, filename)
    
    if os.path.exists(output_file):
        print(f"\n--- Combination {count}/{total_combinations}: Sigma={sigma}, Thresh={thresh}, Gamma={gamma} ---")
        print(f"[Skip] {output_file} already exists.")
        return
    
    # Load DICOM in subprocess
    img, mod = loadLargestSeries(folder_path)
    if img is None:
        print(f"[Error] Could not load DICOM from {folder_path}")
        return
    
    # Now process
    process_combination_safe(img, folder_name, scan_output_dir, sigma, thresh, gamma, count, total_combinations)


def process_folder(folder_path, output_dir):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    
    # Create a subdirectory for this scan's results
    scan_output_dir = os.path.join(output_dir, folder_name)
    if not os.path.exists(scan_output_dir):
        os.makedirs(scan_output_dir)

    # Check if all outputs already exist to skip this folder entirely
    remaining_triplets = []
    for sigma, thresh, gamma in TRIPLETS:
        filename = f"{folder_name}_s{sigma}_t{thresh}_g{int(gamma)}.stl"
        output_file = os.path.join(scan_output_dir, filename)
        if not os.path.exists(output_file):
            remaining_triplets.append((sigma, thresh, gamma))
    
    if not remaining_triplets:
        print(f"[Skip] Scan {folder_name} already fully processed.")
        return

    print(f"\n=== Processing {folder_name} ({len(remaining_triplets)}/{len(TRIPLETS)} remaining) ===")
    
    total_combinations = len(TRIPLETS)
    
    for i, (sigma, thresh, gamma) in enumerate(TRIPLETS):
        count = i + 1
        
        # Check if this specific file already exists (for resume)
        filename = f"{folder_name}_s{sigma}_t{thresh}_g{int(gamma)}.stl"
        output_file = os.path.join(scan_output_dir, filename)
        if os.path.exists(output_file):
            print(f"[Skip] Combination {count}/{total_combinations} (s{sigma}_t{thresh}_g{gamma}) already exists.")
            continue
        
        # Run in a separate process to prevent OOM kills from stopping the batch
        # Pass folder_path instead of img to avoid pickle issues on Windows
        p = multiprocessing.Process(
            target=process_combination_subprocess,
            args=(folder_path, folder_name, scan_output_dir, sigma, thresh, gamma, count, total_combinations)
        )
        p.start()
        p.join()
        
        if p.exitcode != 0:
            print(f"[Error] Process for combination {count} (s{sigma}_t{thresh}_g{gamma}) crashed or was killed (Exit Code: {p.exitcode}). Continuing...")


def main():
    parser = argparse.ArgumentParser(description="Standalone Airway Pipeline")
    
    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch", action="store_true", help="Process all scans in archive/CT_Scans")
    group.add_argument("--single", action="store_true", help="Process a single DICOM folder")
    group.add_argument("--visualize", action="store_true", help="Visualize STL files")
    
    # Arguments
    parser.add_argument("--input", help="Input DICOM folder (for --single) or STL files (for --visualize)")
    parser.add_argument("--output", default="processed_airways", help="Output directory")
    parser.add_argument("--root", default="archive/CT_Scans", help="Root directory for batch processing")
    
    # Visualization args (if multiple files passed to input)
    parser.add_argument("files", nargs="*", help="Files to visualize")

    args = parser.parse_args()
    
    if args.batch:
        # Direct use of root, no download
        root_dirs = [
            os.path.join(args.root, "EXP1_blind"),
            os.path.join(args.root, "EXP2_open")
        ]
        
        if not os.path.exists(args.output):
            os.makedirs(args.output)
            
        for root in root_dirs:
            if not os.path.exists(root):
                print(f"[Warning] Directory {root} not found.")
                continue
            subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            print(f"Found {len(subdirs)} scans in {root}")
            for subdir in subdirs:
                process_folder(subdir, args.output)
                
    elif args.single:
        if not args.input:
            print("Error: --input required for single mode")
            return
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        process_folder(args.input, args.output)
        
    elif args.visualize:
        targets = []
        if args.input: targets.append(args.input)
        targets.extend(args.files)
        visualize_files(targets)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) # Safer for CUDA/OpenCV/VTK if used
    main()
