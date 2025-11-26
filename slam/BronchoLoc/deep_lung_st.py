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
    's': {
        'embed_dim': 16,
        'num_heads': 2,
        'vi_layers': 1,
        'gat_heads': 2,
        'lstm_hidden': 32,
        'num_particles': 20  
    },
    'm': {
        'embed_dim': 256,   
        'num_heads': 4,   
        'vi_layers': 4,
        'gat_heads': 4,     
        'lstm_hidden': 128,
        'num_particles': 20
    },
    'l': {
        'embed_dim': 1024,
        'num_heads': 16,
        'vi_layers': 12,
        'gat_heads': 8,
        'lstm_hidden': 512,
        'num_particles': 20
    }
}

# ==============================================================================
# PART 1: DIFFERENTIABLE PHYSICS LAYER (SDF + Gravity Patch)
# ==============================================================================
class DifferentiableSDFConstraint(nn.Module):
    """
    Enforces geometric priors. 
    Patch: Adds 'Gravity Well' for points outside the voxel grid.
    """
    def __init__(self, sdf_volume_tensor, grid_transform_matrix):
        super().__init__()
        
        if sdf_volume_tensor.dim() == 3:
            sdf_volume_tensor = sdf_volume_tensor.unsqueeze(0).unsqueeze(0)
            
        self.register_buffer('sdf', sdf_volume_tensor)
        
        # World (mm) -> Voxel Index
        inv_transform = torch.linalg.inv(grid_transform_matrix)
        self.register_buffer('world_to_vox_transform', inv_transform)
        
        depth, height, width = sdf_volume_tensor.shape[2:]
        self.grid_dims = torch.tensor([width, height, depth], device=sdf_volume_tensor.device)

    def world_to_norm_grid(self, points):
        # points: (B * K, 3)
        B = points.shape[0]
        ones = torch.ones(B, 1, device=points.device)
        pts_homo = torch.cat([points, ones], dim=1)
        
        vox_coords = (self.world_to_vox_transform @ pts_homo.T).T
        vox_xyz = vox_coords[:, :3]
        
        # Normalize to [-1, 1] for grid_sample
        norm_coords = 2.0 * (vox_xyz / (self.grid_dims - 1.0)) - 1.0
        return norm_coords.view(1, 1, 1, B, 3)

    def forward(self, pred_positions):
        if not pred_positions.requires_grad:
            pred_positions.requires_grad_(True)
        
        # 1. Map World -> Grid
        grid_coords_raw = self.world_to_norm_grid(pred_positions) # (1, 1, 1, BK, 3)
        
        # --- PATCH: GRAVITY WELL FOR OUT-OF-BOUNDS ---
        # grid_sample returns 0 gradient for points outside [-1, 1].
        # We manually calculate a penalty distance for these points.
        
        # Squeeze to (BK, 3) for distance calc
        coords_flat = grid_coords_raw.view(-1, 3)
        
        # Calculate how far outside the [-1, 1] box we are
        # max(0, |x| - 1)
        dist_outside = torch.clamp(torch.abs(coords_flat) - 1.0, min=0.0)
        gravity_force = torch.norm(dist_outside, dim=1, keepdim=True) # (BK, 1)
        
        # Create a gradient direction pointing back to center (0,0,0)
        # We assume 0,0,0 is the center of the lung volume.
        # This acts as a "rubber band" pulling lost particles back.
        gravity_grad = -1.0 * coords_flat * (gravity_force > 0).float()
        
        # 2. Sample SDF (Standard Collision)
        sdf_val = F.grid_sample(self.sdf, grid_coords_raw, mode='bilinear', padding_mode='zeros', align_corners=True)
        sdf_val = sdf_val.view(-1, 1) # (BK, 1)

        # 3. Collision Logic (SDF > 0 is OUTSIDE)
        violation = F.relu(sdf_val)
        
        # 4. Compute Wall Normal
        grad_outputs = torch.ones_like(violation)
        if violation.requires_grad:
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
        
        # 5. Apply Hard Constraints
        # Correction A: Wall Push (Local SDF)
        wall_correction = violation * sdf_grad.detach() 
        
        # Correction B: Gravity Pull (Global Bounds)
        # If far outside, sdf_grad is 0, so gravity takes over.
        # We detach the direction but keep magnitude differentiable if needed, 
        # or just treat it as a hard update step. Here we mimic the wall push logic.
        gravity_correction = gravity_force * -gravity_grad.detach() # Push back towards center

        # Combine
        total_correction = wall_correction + (0.5 * gravity_correction)
        constrained_pos = pred_positions - total_correction
        
        # Return violations for loss (include gravity penalty)
        total_violation = violation + gravity_force
        
        return constrained_pos, total_violation

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
        attn_out, _ = self.spatial_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        bt, n, d = x.shape
        b = bt // self.t_frames
        x = rearrange(x, '(b t) n d -> (b n) t d', b=b, t=self.t_frames)
        
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.norm2(x + attn_out)
        
        x = rearrange(x, '(b n) t d -> (b t) n d', b=b)
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
        
        # Calculate expected patches for the default config
        self.num_patches = (img_size // self.patch_size) ** 2
        
        # Learnable Positional Embeddings: (1, T, N, D)
        # We initialize for 128x128, but the forward pass will interpolate if input is larger
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
        
        # --- FIX: Dynamic Positional Embedding Interpolation ---
        curr_patches_h = x.shape[2]
        curr_patches_w = x.shape[3]
        curr_num_patches = curr_patches_h * curr_patches_w
        
        x = x.flatten(2).transpose(1, 2) # (BT, N_curr, D)
        
        if curr_num_patches != self.num_patches:
            # The input video is NOT 128x128. We must interpolate the pos_embed.
            pos = self.pos_embed # (1, T, N, D)
            T_dim = pos.shape[1]
            
            # 1. Reshape to spatial grid: (1, T, 8, 8, D) -> (T, D, 8, 8)
            orig_size = int(self.num_patches ** 0.5)
            pos_grid = rearrange(pos, '1 t (h w) d -> (1 t) d h w', h=orig_size, w=orig_size)
            
            # 2. Interpolate to new size (e.g., 60x60)
            pos_new = F.interpolate(
                pos_grid, 
                size=(curr_patches_h, curr_patches_w), 
                mode='bicubic', 
                align_corners=False
            )
            
            # 3. Flatten back: (T, D, H_new, W_new) -> (1, T, N_new, D)
            pos = rearrange(pos_new, '(b t) d h w -> b t (h w) d', b=1, t=T_dim)
        else:
            pos = self.pos_embed

        # Expand to batch size and add
        pos = repeat(pos, '1 t n d -> (b t) n d', b=B)
        x = x + pos
        # -------------------------------------------------------
        
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
        x = self.pos_enc(node_pos)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x # (Num_Nodes, D)

# ==============================================================================
# PART 4: MAIN ARCHITECTURE (DeepLungST - Multi-Hypothesis)
# ==============================================================================
class DeepLungST(nn.Module):
    def __init__(self, t_frames, sdf_volume_tensor, grid_transform_matrix, mode='s'):
        super().__init__()
        self.t_frames = t_frames
        self.config = MODEL_CONFIGS[mode]
        embed_dim = self.config['embed_dim']
        self.K = self.config['num_particles'] # Number of particles
        
        # Encoders
        self.visual_encoder = STViViT(self.config, img_size=128, t_frames=t_frames)
        self.map_encoder = MapEncoderGAT(self.config, in_channels=11)
        
        # Improvement 3: Map Retrieval Projection Heads
        self.visual_proj = nn.Linear(embed_dim, embed_dim)
        self.map_proj = nn.Linear(embed_dim, embed_dim)
        
        # Fusion
        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=self.config['num_heads'], batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Kinematics (Particle Filter)
        self.kinematic_rnn = nn.LSTM(embed_dim, self.config['lstm_hidden'], batch_first=True)
        
        # Velocity Head: Outputs mean velocity + variance/noise scale
        self.velocity_head = nn.Linear(self.config['lstm_hidden'], 3)
        self.noise_scale_head = nn.Linear(self.config['lstm_hidden'], 3)
        
        # Physics
        self.constraint = DifferentiableSDFConstraint(sdf_volume_tensor, grid_transform_matrix)

    def forward(self, video, map_nodes, map_edges, map_edge_attr, initial_pose=None):
        B = video.shape[0]
        K = self.K
        
        # 1. Map Encoding
        map_nodes_embed = self.map_encoder(map_nodes, map_edges, map_edge_attr)
        
        # 2. Visual Encoding
        visual_tokens = self.visual_encoder(video)
        
        # --- Projection Heads ---
        visual_emb_global = visual_tokens.mean(dim=1) 
        visual_feat_proj = self.visual_proj(visual_emb_global) 
        map_feat_proj = self.map_proj(map_nodes_embed)         
        
        # 3. Expand for Particle Filter
        map_embed_expanded = repeat(map_nodes_embed, 'n d -> (b k) n d', b=B, k=K)
        visual_tokens_expanded = repeat(visual_tokens, 'b t d -> (b k) t d', k=K)
        
        # 4. Fusion
        context, _ = self.fusion(visual_tokens_expanded, map_embed_expanded, map_embed_expanded)
        context = self.norm(context + visual_tokens_expanded) 
        
        # 5. Kinematics
        rnn_out, _ = self.kinematic_rnn(context) 
        velocities = self.velocity_head(rnn_out)   
        noise_scales = torch.sigmoid(self.noise_scale_head(rnn_out))
        
        # 6. Integration & Constraint Loop
        if initial_pose is None:
            initial_pose = torch.zeros(B, 3, device=video.device)
            
        # --- FIX: Handling Independent Particle Continuation ---
        # If initial_pose is (B, 3), we spawn K particles (Start of inference/Training).
        # If initial_pose is (B, K, 3), we resume existing K particles (Mid-inference).
        if initial_pose.dim() == 3: # shape (B, K, 3)
            current_pos = rearrange(initial_pose, 'b k d -> (b k) d')
        else: # shape (B, 3)
            current_pos = repeat(initial_pose, 'b d -> (b k) d', k=K)
            # Add initial noise only when spawning new particles
            current_pos = current_pos + (torch.randn_like(current_pos) * 2.0)
        
        traj_pos = []
        violations = []
        
        for t in range(self.t_frames):
            vel_mean = velocities[:, t, :]
            noise_sigma = noise_scales[:, t, :]
            
            # Stochastic Step
            step_noise = torch.randn_like(vel_mean) * noise_sigma
            proposal = current_pos + vel_mean + step_noise
            
            # Physics Constraints
            constrained, viol = self.constraint(proposal)
            
            current_pos = constrained
            traj_pos.append(current_pos)
            violations.append(viol)
            
        traj_stack = torch.stack(traj_pos, dim=1) 
        viol_stack = torch.stack(violations, dim=1) 
        
        traj_out = rearrange(traj_stack, '(b k) t d -> b k t d', b=B, k=K)
        viol_out = rearrange(viol_stack, '(b k) t d -> b k t d', b=B, k=K)
        
        return traj_out, viol_out, visual_feat_proj, map_feat_proj