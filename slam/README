# BronchoLoc & BronchoSim

[![Website](https://img.shields.io/badge/Website-surgym.com-blue)](https://surgym.com)

<!-- Note: Author information and institutional affiliations are hidden for double-blind peer review -->

**End-to-end Learning-based Bronchoscope Tracking Enhanced with Medical CT Imaging and a Novel Open-Source Simulator**

This repository contains the implementation of BronchoLoc, a novel learning-based framework for bronchoscopy localization, and BronchoSim, an open-source bronchoscopy simulator for synthetic dataset generation and algorithm benchmarking.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{anonymous2025broncholoc,
  title={End-to-end learning-based bronchoscope tracking enhanced with medical CT imaging and a novel open-source simulator},
  author={Anonymous},
  journal={Under Review},
  year={2025},
  note={Author information hidden for double-blind review}
}
```

## Table of Contents

- [Overview](#overview)
- [BronchoLoc](#broncholoc)
  - [Architecture](#architecture)
  - [ANT (Airways Neighborhood Tracker)](#ant-airways-neighborhood-tracker)
  - [BIRD (Bronchial Intraoperative Route Discriminator)](#bird-bronchial-intraoperative-route-discriminator)
  - [Training](#training)
  - [Testing](#testing)
  - [Phantom Dataset Processing](#phantom-dataset-processing)
- [BronchoSim](#bronchosim)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Data Preparation](#data-preparation)
  - [Configuration](#configuration)
  - [Running the Simulator](#running-the-simulator)
  - [Utilities](#utilities)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

Robotic bronchoscopy platforms offer improved precision and stability over manual procedures, yet achieving autonomous or semi-autonomous navigation remains limited by the lack of accurate intraoperative tracking within the complex bronchial tree. 

This project introduces:

1. **BronchoLoc**: A learning-based localization framework that synergizes anatomical priors from preoperative CT imaging with:
   - A local visual perception module (ANT) for frame-to-frame motion estimation
   - A global anatomical reasoning module (BIRD) utilizing neural memory for trajectory context and drift correction

2. **BronchoSim**: An open-source bronchoscopy simulator for:
   - Generating synthetic datasets with ground truth poses
   - Algorithm development and benchmarking
   - Real-time visualization of tracking results

---

## BronchoLoc

BronchoLoc is located in the `BronchoLoc/` directory and provides end-to-end localization by constraining predictions to lie on the patient-specific anatomical manifold extracted from CT imaging.

### Architecture

The system consists of two coupled modules:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BronchoLoc Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────────────────┐    ┌─────────────┐  │
│  │ Video Input │───▶│  ANT (Local Tracking)   │───▶│    BIRD     │  │
│  │  (B,T,C,H,W)│    │  - ST-ViViT Encoder     │    │  (Global    │  │
│  └─────────────┘    │  - VO Module            │    │  Refinement)│  │
│                     │  - Local Map Encoder    │    │             │  │
│  ┌─────────────┐    │  - Cross-Attention      │    │  - Neural   │  │
│  │ Centerline  │───▶│    Selection            │───▶│    Memory   │  │
│  │  Map (CT)   │    └─────────────────────────┘    │  - Global   │  │
│  └─────────────┘                                   │    Attention│  │
│                                                    └─────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### ANT (Airways Neighborhood Tracker)

The local prediction module (`ant.py`) that estimates bronchoscope movement relative to immediate airway anatomy.

**Key Components:**
- **Spatio-Temporal Vision Transformer (ST-ViViT)**: Processes video windows with spatial and temporal attention mechanisms
- **Visual Odometry Module**: Predicts position deltas and quaternion rotations
- **Local Map Encoder**: Encodes nearby centerline candidate points
- **Cross-Attention Selection**: Selects position from local candidates based on visual-motion features

**Model Configurations:**
| Mode | Embed Dim | Heads | Layers | Parameters |
|------|-----------|-------|--------|------------|
| xs   | 128       | 2     | 4      | ~1.5M      |
| s    | 256       | 4     | 8      | ~6M        |
| b    | 384       | 6     | 12     | ~20M       |
| m    | 512       | 8     | 16     | ~50M       |
| l    | 1024      | 16    | 24     | ~150M      |

### BIRD (Bronchial Intraoperative Route Discriminator)

The global refinement module (`bird.py`) that corrects drift using the full anatomical context.

**Key Components:**
- **Titans Neural Memory**: Maintains trajectory history with persistent memory state
- **Global Centerline Encoder**: Pre-encodes the entire airway centerline
- **Global Cross-Attention**: Attends to complete anatomical map for corrections
- **Distance-Penalized Selection**: Balances local consistency with global corrections

### Training

#### Stage 1: Train ANT (Local Pre-training)

```bash
cd BronchoLoc
python train_ant.py \
    --data_root ./dataset \
    --mode m \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --img_size 128
```

**Arguments:**
- `--data_root`: Path to training data directory
- `--mode`: Model size (xs, s, b, m, l)
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--img_size`: Input image resolution
- `--resume`: Path to checkpoint for resuming training
- `--debug`: Enable debug mode with single sample

#### Stage 2: Train BIRD (Global Refinement)

```bash
python train_bird.py \
    --data_root ./dataset \
    --ant_checkpoint ./checkpoints/ant_model_m.pth \
    --mode m \
    --epochs 50 \
    --batch_size 1
```

**Arguments:**
- `--ant_checkpoint`: Path to pretrained ANT model (required)
- `--num_centerline_pts`: Size of downsampled centerline (default: 1024)
- Other arguments similar to train_ant.py

### Testing

#### Test ANT Only

```bash
python test_ant.py \
    --checkpoint ./checkpoints/ant_model_m.pth \
    --data_root ./dataset \
    --output_dir ./results
```

#### Test ANT + BIRD

```bash
python test_bird.py \
    --ant_checkpoint ./checkpoints/ant_model_m.pth \
    --bird_checkpoint ./checkpoints/bird_model_m.pth \
    --data_root ./dataset \
    --output_dir ./results
```

**Test outputs include:**
- 3D trajectory visualization (PyVista)
- Side-by-side video with predictions
- Quantitative metrics (ATE, etc.)

### Phantom Dataset Processing

For processing real phantom bronchoscopy recordings:

#### 1. Align Phantom Trajectory

Compute similarity transformation between sensor trajectory and 3D lung model:

```bash
python align_phantom_traj.py <sequence_name> \
    --annotate  # Interactive correspondence annotation
```

#### 2. Refine Alignment (Optional)

Manually adjust transformation with real-time visualization:

```bash
python refine_phantom_traj.py <sequence_name>
```

#### 3. Build Dataset

Convert phantom recordings to training format:

```bash
python build_phantom_dataset.py \
    --phantom_dir ./dataset/phantom/data \
    --output_root ./dataset/sequences
```

### Phantom Sensor Data Utilities

Located in `BronchoLoc/dataset/phantom/`:

| Script | Description |
|--------|-------------|
| `TrakSTAR_to_TUM.py` | Convert TrakSTAR magnetic tracker .mat files to TUM trajectory format |
| `trim_traj.py` | Interactive GUI tool to trim trajectory recordings (select start/end points) |
| `vis_traj.py` | Synchronized video + trajectory visualization with split functionality |

### Static Dataset Building

Build the static centerline dataset for inference:

```bash
python build_static_dataset.py
```

This script (`build_static_dataset.py`):
- Loads VTK skeleton and CAD model
- Exports `centerline.npz` for the ANT network
- Provides visualization for verification

### Window Configuration

The `window_config.json` file stores hyperparameters for video window processing:

```json
{
  "window_size": 10,
  "frame_skip": 40
}
```

Use `check/check_win.py` to interactively set these values.

### Utility Scripts

Located in `BronchoLoc/check/`:

| Script | Description |
|--------|-------------|
| `check_win.py` | Interactive tool to set window size and frame skip configuration |
| `check_traj.py` | Visualize trajectory predictions with 3D rendering |
| `check_ball.py` | Visualize local ball query regions and candidate points |
| `check_attention.py` | Visualize cross-attention weights over centerline candidates |
| `check_data_values.py` | Validate dataset statistics (min, max, normalization) |
| `check_centerlines.py` | Visualize and validate centerline data |
| `check_convex_hull.py` | Visualize convex hull of candidate points |
| `check_downsample.py` | Test and visualize density-based downsampling |
| `check_smoothing.py` | Visualize trajectory smoothing effects |
| `check_vid.py` | Video playback and inspection tool |

Located in `BronchoLoc/utils/`:
- `utils.py`: Core utility functions (centerline loading, FPS sampling, path finding, trajectory interpolation)

---

## BronchoSim

BronchoSim is located in the `BronchoSim/` directory and provides a simulation environment built on the Panda3D engine.

### Features

- **Patient-Specific Models**: Load 3D airway meshes from CT scans
- **Multiple View Modes**: First-person (FPV) and third-person (TPV) views
- **Respiratory Motion**: Configurable breathing simulation with sinusoidal deformation
- **Ground Truth Generation**: RGB images, depth maps, and 6-DOF poses in TUM format
- **Navigation Modes**:
  - Manual keyboard control
  - Autopilot mode for dataset generation
  - Live mode for external localization visualization
- **Configurable Camera**: Full control over intrinsic parameters

### Prerequisites

- **FreeCAD**: For STL to STEP conversion
- **Gmsh**: For mesh generation
- **VMTK**: For centerline extraction (`conda install -c vmtk vmtk`)
- **Python Dependencies**: Panda3D, NumPy, SciPy, PyVista

### Data Preparation

#### 1. Convert STL to VTK

```bash
# Using FreeCAD GUI:
# 1. Import STL file
# 2. Part menu → Create shape from mesh
# 3. Part menu → Convert to solid
# 4. Export as .step file

# Using Gmsh:
# 1. Import .step file
# 2. Mesh → 3D to generate volumetric mesh
# 3. Export as .vtk
```

#### 2. Convert VTK to VTP

```bash
python utils/format_3d.py -i path/to/input.vtk -o path/to/output.vtp
```

#### 3. Extract Centerlines

Using the provided script:
```bash
./centerline.sh
```

Or manually with VMTK:
```bash
conda activate vmtk
vmtkcenterlines -ifile path/to/output.vtp -ofile path/to/centerline.vtp
# GUI: Select end points (space), press q, select start points (space), press q
```

#### 4. Create Negative Model (for FPV)

In FreeCAD:
1. Import the .obj model
2. Create shape from mesh
3. Make solid from shape
4. Create a containing sphere
5. Boolean difference: sphere - solid
6. Export the result as .obj

### Configuration

Edit `config.ini` to customize:

```ini
[PATHS]
data_folder = data/mesh/lungs/sim/
path_name = centerlines/b1.vtp
negative_model_name = model_negative.obj
model_name = model.obj
texture_name = data/textures/mucosa_diffuse.png

[CAMERA]
width = 360
height = 360
fx = 100.0
fy = 100.0
cx = 180.0
cy = 180.0
np = 0.1
fp = 1000.0
depth_map_factor = 5000.0
depth_bool = 1

[RECORD]
legacy_record_method = 0

[BREATHING]
enabled = true
min_scale = 0.95
max_scale = 1.05
period_seconds = 3.0
```

### Running the Simulator

#### Interactive Mode
```bash
python main.py
```

#### Recording Mode (Dataset Generation)
```bash
python main.py -record
```

#### Record All Branches
```bash
python main.py -record -all
```

#### Batch Recording (Shell Script)
```bash
./run_random.sh
```

#### Live Mode (External Localization)
Enable `sim_server_bool = 1` in config and connect via socket to send 6-DOF poses.

#### Results Visualization Mode
```bash
python main.py -results
```

### Utilities

Located in `BronchoSim/utils/`:

| Script | Description |
|--------|-------------|
| `format_3d.py` | Convert between mesh formats (STL, VTK, VTP, OBJ, MSH) |
| `centerline_variations.py` | Generate centerline variations with smooth noise for data augmentation |
| `set_FS_frame.py` | Compute Rotation Minimizing Frames (RMF) along centerlines |
| `align_trajectory.py` | Align trajectories using Umeyama similarity transformation |
| `pc_viewer.py` | Interactive point cloud visualization |
| `test_depth.py` | Test and validate depth map rendering |
| `visualize_3d.py` | Quick 3D model visualization |
| `generate_normal_map.py` | Generate normal maps from diffuse textures for realistic rendering |

Located in `BronchoSim/src/`:

| Module | Description |
|--------|-------------|
| `BronchoSim.py` | Main simulator class with rendering, navigation, and recording |
| `draw.py` | 3D drawing utilities (trajectories, frames, cones, paths) |
| `utils.py` | Core utilities (FS frames, trajectory snapping, curvilinear abscissa) |
| `server.py` | Socket server for receiving external 6-DOF poses in live mode |

### CT-to-Dataset Pipeline

Located in `BronchoSim/dataset/`:

| Script | Description |
|--------|-------------|
| `build_dataset.py` | Complete airway extraction pipeline from CT DICOM to watertight STL meshes |
| `visualize_airways.py` | Grid visualization of all STL variations for parameter selection |

The `build_dataset.py` script performs:
1. DICOM series loading (SimpleITK)
2. Frangi vesselness filtering for tubular structure extraction
3. Air-intensity masking and connected component analysis
4. Marching cubes surface extraction
5. Smart topological repair (contact-based morphological closing)
6. Multi-parameter grid search (sigma, threshold, gamma)

```bash
cd BronchoSim/dataset
python build_dataset.py --input /path/to/dicom/folder --output processed_airways/
```

Visualize and select the best parameter combination:
```bash
python visualize_airways.py 1003  # Folder name
```

#### Generate Centerline Variations

```bash
python utils/centerline_variations.py \
    --input centerlines/b1.vtp \
    --output variations/ \
    --num_variations 10 \
    --amplitude 0.05 \
    --mesh model.obj  # Optional: constrain to mesh
```

#### Generate Normal Maps

```bash
python utils/generate_normal_map.py
# Generates mucosa_diffuse_normal.png from mucosa_diffuse.png
```

---

## Project Structure

```
slam/
├── README.md                     # This file
│
├── BronchoLoc/                   # Localization framework
│   ├── ant.py                    # ANT model (Spatio-Temporal ViViT + VO + Map Selection)
│   ├── bird.py                   # BIRD model (Titans Neural Memory + Global Attention)
│   ├── ant_dataset.py            # PyTorch dataset with augmentation
│   ├── train_ant.py              # ANT training (Stage 1)
│   ├── train_bird.py             # BIRD training (Stage 2, frozen ANT)
│   ├── test_ant.py               # ANT evaluation with visualization
│   ├── test_bird.py              # ANT+BIRD joint evaluation
│   ├── constants.py              # Hyperparameters (radii, spacing, thresholds)
│   ├── window_config.json        # Window size and frame skip settings
│   ├── align_phantom_traj.py     # Sensor-to-CT alignment via correspondences
│   ├── refine_phantom_traj.py    # Interactive transform refinement GUI
│   ├── build_phantom_dataset.py  # Convert phantom recordings to training format
│   ├── build_static_dataset.py   # Export centerline.npz for inference
│   ├── check/                    # Visualization and debugging tools
│   │   ├── check_win.py          # Set window_size and frame_skip
│   │   ├── check_traj.py         # 3D trajectory visualization
│   │   ├── check_ball.py         # Ball query visualization
│   │   ├── check_attention.py    # Attention weight visualization
│   │   ├── check_data_values.py  # Dataset statistics validation
│   │   ├── check_centerlines.py  # Centerline visualization
│   │   ├── check_convex_hull.py  # Candidate convex hull
│   │   ├── check_downsample.py   # Density-based sampling test
│   │   ├── check_smoothing.py    # Trajectory smoothing effects
│   │   └── check_vid.py          # Video inspection
│   ├── utils/
│   │   └── utils.py              # Core utilities (loading, sampling, paths)
│   ├── dataset/
│   │   ├── phantom/              # Phantom data processing
│   │   │   ├── TrakSTAR_to_TUM.py    # Magnetic tracker format conversion
│   │   │   ├── trim_traj.py          # Interactive trajectory trimming
│   │   │   ├── vis_traj.py           # Video+trajectory sync viewer
│   │   │   └── data/                 # Raw .mat and .mkv files
│   │   └── static/               # Exported centerline.npz
│   ├── patient/                  # Patient-specific anatomical data
│   │   ├── centerline.vtk        # Full centerline point cloud
│   │   ├── lungs.obj             # 3D lung mesh
│   │   └── centerlines/          # Individual branch VTPs
│   ├── checkpoints/              # Saved model weights
│   └── res/                      # Reference papers and notes
│
├── BronchoSim/                   # Bronchoscopy simulator (Panda3D)
│   ├── main.py                   # Entry point
│   ├── config.ini                # Full configuration file
│   ├── centerline.sh             # VMTK centerline extraction script
│   ├── run_random.sh             # Batch recording script (Linux)
│   ├── run_random.bat            # Batch recording script (Windows)
│   ├── src/
│   │   ├── BronchoSim.py         # Main simulator (2274 lines)
│   │   ├── draw.py               # 3D rendering (trajectories, frames, cones)
│   │   ├── utils.py              # FS frames, trajectory snapping, path building
│   │   └── server.py             # Socket server for live 6-DOF poses
│   ├── utils/
│   │   ├── format_3d.py          # Mesh format conversion (STL/VTK/VTP/OBJ/MSH)
│   │   ├── centerline_variations.py  # Data augmentation via smooth noise
│   │   ├── set_FS_frame.py       # Rotation Minimizing Frame computation
│   │   ├── align_trajectory.py   # Umeyama similarity alignment
│   │   ├── pc_viewer.py          # Point cloud viewer
│   │   ├── test_depth.py         # Depth rendering test
│   │   ├── visualize_3d.py       # Quick mesh visualization
│   │   └── generate_normal_map.py # Normal map generation from textures
│   ├── dataset/
│   │   ├── build_dataset.py      # CT DICOM → STL pipeline (Frangi + repair)
│   │   ├── visualize_airways.py  # Grid view for parameter selection
│   │   └── processed_airways/    # Generated STL variations
│   ├── data/
│   │   ├── mesh/                 # 3D models (.obj, .vtp)
│   │   └── textures/             # Diffuse and normal maps
│   └── shaders/                  # Custom GLSL shaders
│
├── ORB_SLAM3/                    # ORB-SLAM3 integration (baseline comparison)
└── rpg_trajectory_evaluation/    # Trajectory evaluation tools (ATE, RPE)
```
---

## License

This project is open-source. See the LICENSE file for details.

---

## Contact

<!-- Contact information hidden for double-blind review -->
*Author and institutional contact information will be provided upon acceptance.*
