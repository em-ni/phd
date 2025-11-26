import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from einops import rearrange, repeat
import numpy as np

# ==============================================================================
# CONFIGURATIONS
# ==============================================================================
MODEL_CONFIGS = {
    'tiny': {
        'embed_dim': 16,       # Small dimension for fast debugging
        'num_heads': 2,
        'vi_layers': 1,
        'gat_heads': 2,
        'lstm_hidden': 32
    },
    'big': {
        'embed_dim': 512,      # Standard dimension for deployment
        'num_heads': 8,
        'vi_layers': 6,
        'gat_heads': 4,
        'lstm_hidden': 256
    }
}

# ==============================================================================
# PART 1: DIFFERENTIABLE PHYSICS LAYER (The "BREA" Constraint)
# ==============================================================================
class DifferentiableSDFConstraint(nn.Module):
    """
    This layer enforces the geometric prior: "The bronchoscope must be inside the lungs."
    It calculates wall violations and pushes predictions back inside using gradients.
    """
    def __init__(self, sdf_volume_tensor, grid_transform_matrix):
        """
        sdf_volume_tensor: (D, H, W) float32 tensor from lung_sdf.pt
        grid_transform_matrix: (4, 4) float32 matrix from grid_transform.npy
        """
        super().__init__()
        
        # Ensure shape is (1, 1, D, H, W) for grid_sample
        if sdf_volume_tensor.dim() == 3:
            sdf_volume_tensor = sdf_volume_tensor.unsqueeze(0).unsqueeze(0)
            
        self.register_buffer('sdf', sdf_volume_tensor)
        
        # We need World (mm) -> Voxel Index. Input is usually Voxel -> World.
        inv_transform = torch.linalg.inv(grid_transform_matrix)
        self.register_buffer('world_to_vox_transform', inv_transform)
        
        # Store grid dimensions to normalize coordinates to [-1, 1]
        depth, height, width = sdf_volume_tensor.shape[2:]
        self.grid_dims = torch.tensor([width, height, depth], device=sdf_volume_tensor.device)

    def world_to_norm_grid(self, points):
        B = points.shape[0]
        ones = torch.ones(B, 1, device=points.device)
        pts_homo = torch.cat([points, ones], dim=1)
        
        # World -> Voxel
        vox_coords = (self.world_to_vox_transform @ pts_homo.T).T
        vox_xyz = vox_coords[:, :3]
        
        # Normalize to [-1, 1]
        norm_coords = 2.0 * (vox_xyz / (self.grid_dims - 1.0)) - 1.0
        return norm_coords.view(1, 1, 1, B, 3)

    def forward(self, pred_positions):
        # Enable gradient calculation for the "Push" logic
        if not pred_positions.requires_grad:
            pred_positions.requires_grad_(True)
        
        # 1. Map World -> Grid
        grid_coords = self.world_to_norm_grid(pred_positions)
        
        # 2. Sample SDF
        # padding_mode='zeros' is differentiable and safe.
        # 'border' caused the double-backward error.
        sdf_val = F.grid_sample(self.sdf, grid_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
        sdf_val = sdf_val.view(-1, 1)

        # 3. Collision Logic (SDF > 0 is OUTSIDE)
        violation = F.relu(sdf_val) 
        
        # 4. Compute Gradient (Wall Normal)
        grad_outputs = torch.ones_like(violation)
        
        if violation.requires_grad:
            # CRITICAL FIX: create_graph=False because we detach the result anyway.
            # This avoids the "derivative for grid_sampler_3d_backward not implemented" error.
            sdf_grad = torch.autograd.grad(
                outputs=violation, 
                inputs=pred_positions, 
                grad_outputs=grad_outputs, 
                create_graph=False, 
                retain_graph=True, 
                only_inputs=True
            )[0]
        else:
            sdf_grad = torch.zeros_like(pred_positions)
        
        # 5. Apply Hard Constraint
        # We detach() sdf_grad to treat the wall normal as a fixed geometric property.
        # The network learns to minimize 'violation' (which is differentiable).
        correction = violation * sdf_grad.detach() 
        
        constrained_pos = pred_positions - correction
        
        return constrained_pos, violation

# ==============================================================================
# PART 2: VISUAL STREAM (Genie ST-ViViT Adaptation)
# ==============================================================================
class SpatioTemporalBlock(nn.Module):
    def __init__(self, dim, num_heads, t_frames):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        self.t_frames = t_frames

    def forward(self, x):
        # x: (B*T, N_patches, D)
        
        # 1. Spatial Attention
        attn_out, _ = self.spatial_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # 2. Reshape for Temporal Attention
        # (B*T, N, D) -> (B, T, N, D) -> (B, N, T, D) -> (B*N, T, D)
        bt, n, d = x.shape
        b = bt // self.t_frames
        
        # Correct reshape logic for temporal dimension
        x = rearrange(x, '(b t) n d -> (b n) t d', b=b, t=self.t_frames)
        
        # 3. Temporal Attention
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.norm2(x + attn_out)
        
        # 4. Restore Shape
        x = rearrange(x, '(b n) t d -> (b t) n d', b=b)
        
        # 5. MLP
        x = self.norm3(x + self.mlp(x))
        return x

class STViViT(nn.Module):
    def __init__(self, config, img_size=128, t_frames=16):
        super().__init__()
        dim = config['embed_dim']
        layers = config['vi_layers']
        heads = config['num_heads']
        
        self.patch_size = 16
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Calculate expected patches
        self.num_patches = (img_size // self.patch_size) ** 2
        
        # Learnable Positional Embeddings: (1, T, N, D)
        # We initialize for the config img_size, but will interpolate if input differs
        self.pos_embed = nn.Parameter(torch.randn(1, t_frames, self.num_patches, dim))
        
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(dim, heads, t_frames) for _ in range(layers)
        ])
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, video_tensor):
        # video: (B, T, 3, H, W)
        B, T, C, H, W = video_tensor.shape
        
        # Flatten batch and time
        x = video_tensor.view(B*T, C, H, W)
        
        # Patch Embedding
        x = self.patch_embed(x) # (BT, D, h_patches, w_patches)
        
        # Dynamic Positional Embedding Interpolation
        curr_patches_h = x.shape[2]
        curr_patches_w = x.shape[3]
        curr_num_patches = curr_patches_h * curr_patches_w
        
        x = x.flatten(2).transpose(1, 2) # (BT, N_curr, D)
        
        if curr_num_patches != self.num_patches:
            # Interpolate pos_embed to match current input resolution
            pos = self.pos_embed # (1, T, N, D)
            T_dim = pos.shape[1]
            D_dim = pos.shape[3]
            
            # Reshape to spatial grid
            orig_size = int(self.num_patches ** 0.5)
            pos = rearrange(pos, '1 t (h w) d -> (1 t) d h w', h=orig_size, w=orig_size)
            
            # Interpolate
            pos = F.interpolate(pos, size=(curr_patches_h, curr_patches_w), mode='bicubic', align_corners=False)
            
            # Flatten back
            pos = rearrange(pos, '(b t) d h w -> b t (h w) d', b=1, t=T_dim)
        else:
            pos = self.pos_embed

        # Add Positional Embeddings
        # pos shape: (1, T, N, D) -> expand to (B, T, N, D) -> flatten to (B*T, N, D)
        pos = repeat(pos, '1 t n d -> (b t) n d', b=B)
        x = x + pos
        
        # Apply Transformer Blocks
        for block in self.blocks:
            x = block(x)
            
        # Global Average Pooling (over spatial patches)
        x = x.mean(dim=1) # (BT, D)
        
        # Reshape back to Sequence
        x = x.view(B, T, -1) # (B, T, D)
        
        x = self.proj_out(x)
        return x

# ==============================================================================
# PART 3: MAP STREAM (Topological Encoder)
# ==============================================================================
class MapEncoderGAT(nn.Module):
    def __init__(self, config, in_channels=11):
        super().__init__()
        dim = config['embed_dim']
        gat_heads = config['gat_heads']
        
        self.conv1 = GATConv(in_channels, dim // gat_heads, heads=gat_heads, concat=True)
        self.conv2 = GATConv(dim, dim, heads=1, concat=False)
        self.pos_enc = nn.Linear(3, in_channels)

    def forward(self, node_pos, edge_index, edge_attr):
        # Encode absolute positions
        x = self.pos_enc(node_pos)
        # GAT Layers
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x

# ==============================================================================
# PART 4: MAIN ARCHITECTURE (DeepLungST)
# ==============================================================================
class DeepLungST(nn.Module):
    def __init__(self, t_frames, sdf_volume_tensor, grid_transform_matrix, mode='tiny'):
        super().__init__()
        self.t_frames = t_frames
        self.config = MODEL_CONFIGS[mode]
        embed_dim = self.config['embed_dim']
        
        # Encoders
        self.visual_encoder = STViViT(self.config, img_size=128, t_frames=t_frames)
        self.map_encoder = MapEncoderGAT(self.config, in_channels=11)
        
        # Fusion (Cross-Attention)
        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=self.config['num_heads'], batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Kinematics
        self.kinematic_rnn = nn.LSTM(embed_dim, self.config['lstm_hidden'], batch_first=True)
        self.velocity_head = nn.Linear(self.config['lstm_hidden'], 3)
        
        # Physics
        self.constraint = DifferentiableSDFConstraint(sdf_volume_tensor, grid_transform_matrix)

    def forward(self, video, map_nodes, map_edges, map_edge_attr, initial_pose=None):
        B = video.shape[0]
        
        # A. Map Encoding
        map_embed_single = self.map_encoder(map_nodes, map_edges, map_edge_attr)
        map_embed = repeat(map_embed_single, 'n d -> b n d', b=B)
        
        # B. Visual Encoding
        visual_tokens = self.visual_encoder(video)
        
        # C. Fusion
        context, _ = self.fusion(visual_tokens, map_embed, map_embed)
        context = self.norm(context + visual_tokens)
        
        # D. Kinematics
        rnn_out, _ = self.kinematic_rnn(context)
        velocities = self.velocity_head(rnn_out)
        
        # E. Integration & Constraint
        if initial_pose is not None: 
            current_pos = initial_pose 
        else:
            current_pos = torch.zeros(B, 3, device=video.device)
        
        traj_pos = []
        violations = []
        
        for t in range(self.t_frames):
            vel = velocities[:, t, :]
            proposal = current_pos + vel
            constrained, viol = self.constraint(proposal)
            current_pos = constrained
            traj_pos.append(current_pos)
            violations.append(viol)
            
        return torch.stack(traj_pos, dim=1), torch.stack(violations, dim=1)

# ==============================================================================
# PART 5: LOSS FUNCTION
# ==============================================================================
def deep_lung_loss(pred_traj, gt_traj, violations, sdf_lambda=10.0):
    loss_pose = F.mse_loss(pred_traj, gt_traj)
    loss_geo = violations.mean()
    
    vel = pred_traj[:, 1:] - pred_traj[:, :-1]
    accel = vel[:, 1:] - vel[:, :-1]
    loss_smooth = torch.mean(accel**2)
    
    total_loss = loss_pose + (sdf_lambda * loss_geo) + (0.1 * loss_smooth)
    return total_loss, {"pose": loss_pose, "geo": loss_geo, "smooth": loss_smooth}