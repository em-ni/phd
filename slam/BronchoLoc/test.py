import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv

from deep_lung_st import DeepLungST
from deep_lung_dataset import DeepLungDataset

def test(args):
    # --- 1. SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Testing on {device}")
    
    static_dir = os.path.join(args.data_root, "static")
    sequences_dir = os.path.join(args.data_root, "sequences")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. LOAD STATIC DATA ---
    print("[INFO] Loading Static Map Data...")
    try:
        sdf_vol = torch.load(os.path.join(static_dir, "lung_sdf.pt"), map_location=device)
        grid_trans = torch.from_numpy(np.load(os.path.join(static_dir, "grid_transform.npy"))).float().to(device)
        graph_data = np.load(os.path.join(static_dir, "deep_lung_graph.npz"))
    except FileNotFoundError:
        print(f"[ERROR] Static data missing in {static_dir}")
        return

    # Graph Tensors
    node_pos = torch.from_numpy(graph_data['node_pos']).float().to(device)
    edge_index = torch.from_numpy(graph_data['edge_index']).long().to(device)
    edge_attr = torch.from_numpy(graph_data['edge_attr']).float().to(device)

    # --- 3. MODEL ---
    print(f"[INFO] Loading Model ({args.model_mode})...")
    model = DeepLungST(
        t_frames=args.t_frames, 
        sdf_volume_tensor=sdf_vol, 
        grid_transform_matrix=grid_trans,
        mode=args.model_mode
    ).to(device)
    
    # Load Checkpoint
    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found at {ckpt_path}")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"[INFO] Loaded weights from {ckpt_path}")

    # --- 4. DATASET ---
    # Stride=T means no overlap, testing distinct segments
    test_dataset = DeepLungDataset(data_root=sequences_dir, t_frames=args.t_frames, mode='val', stride=args.t_frames)
    print(f"[INFO] Test Samples: {len(test_dataset)}")

    # --- 5. INFERENCE LOOP ---
    all_ade = [] # Average Displacement Error
    all_viol = [] # Wall Violations
    
    # Visualization limit
    viz_count = 0
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            
            # Add Batch Dim
            video = sample['video'].unsqueeze(0).to(device)   # (1, T, 3, H, W)
            gt_pos = sample['gt_pos'].unsqueeze(0).to(device) # (1, T, 3)
            
            initial_pose = gt_pos[:, 0, :]
            
            # Forward
            pred_traj, violations = model(video, node_pos, edge_index, edge_attr, initial_pose)
            
            # Metrics
            # ADE: Mean Euclidean distance over time
            diff = pred_traj - gt_pos
            ade = torch.norm(diff, dim=2).mean().item()
            all_ade.append(ade)
            
            # Violation Ratio: % of points strictly inside wall (viol > 0)
            viol_count = (violations > 0).float().sum()
            viol_ratio = viol_count / (args.t_frames)
            all_viol.append(viol_ratio.item())
            
            # Visualize first few samples
            if viz_count < args.num_viz:
                visualize_result(
                    pred_traj[0].cpu().numpy(), 
                    gt_pos[0].cpu().numpy(), 
                    os.path.join(output_dir, f"result_{i}.png")
                )
                viz_count += 1
            
            if i % 10 == 0:
                print(f"  Sample {i}: ADE={ade:.2f}mm, Viol={viol_ratio:.2%}")

    # --- 6. SUMMARY ---
    mean_ade = np.mean(all_ade)
    mean_viol = np.mean(all_viol)
    
    print("\n" + "="*30)
    print(f"TEST RESULTS ({len(test_dataset)} samples)")
    print(f"Mean ADE: {mean_ade:.4f} mm")
    print(f"Wall Collision Rate: {mean_viol:.2%}")
    print("="*30)

def visualize_result(pred, gt, save_path):
    """ Saves a 3D plot of Predicted vs GT """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot GT
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], c='green', label='Ground Truth', linewidth=2, marker='o', markersize=3)
    # Plot Pred
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], c='blue', label='Predicted', linewidth=2, marker='^', markersize=3)
    
    # Start/End
    ax.scatter(gt[0,0], gt[0,1], gt[0,2], c='black', marker='x', s=50, label='Start')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    
    # Equal aspect ratio hack
    all_pts = np.concatenate([pred, gt])
    max_range = np.array([all_pts[:,0].max()-all_pts[:,0].min(), 
                          all_pts[:,1].max()-all_pts[:,1].min(), 
                          all_pts[:,2].max()-all_pts[:,2].min()]).max() / 2.0
    mid_x = (all_pts[:,0].max()+all_pts[:,0].min()) * 0.5
    mid_y = (all_pts[:,1].max()+all_pts[:,1].min()) * 0.5
    mid_z = (all_pts[:,2].max()+all_pts[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--output_dir', type=str, default='./test_results')
    parser.add_argument('--model_mode', type=str, default='tiny', choices=['tiny', 'big'])
    parser.add_argument('--t_frames', type=int, default=16)
    parser.add_argument('--num_viz', type=int, default=5, help='Number of plots to save')
    
    args = parser.parse_args()
    test(args)