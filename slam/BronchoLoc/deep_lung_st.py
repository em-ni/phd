import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from einops import rearrange, repeat
import numpy as np
from torch.distributions import Categorical

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
        'embed_dim': 1024,
        'num_heads': 16,
        'vi_layers': 12,
        'gat_heads': 8,
        'lstm_hidden': 512,
        'num_particles': 20
    },
    'l': {
        'embed_dim': 2048,
        'num_heads': 32,
        'vi_layers': 24,
        'gat_heads': 16,
        'lstm_hidden': 1024,
        'num_particles': 20
    }
}

class DifferentiableSDFConstraint(nn.Module):
    def __init__(self, sdf_volume_tensor, grid_transform_matrix):
        super().__init__()
        if sdf_volume_tensor.dim() == 3:
            sdf_volume_tensor = sdf_volume_tensor.unsqueeze(0).unsqueeze(0)
        self.register_buffer('sdf', sdf_volume_tensor)
        inv_transform = torch.linalg.inv(grid_transform_matrix)
        self.register_buffer('world_to_vox_transform', inv_transform)
        depth, height, width = sdf_volume_tensor.shape[2:]
        self.grid_dims = torch.tensor([width, height, depth], device=sdf_volume_tensor.device)

    def world_to_norm_grid(self, points):
        B = points.shape[0]
        ones = torch.ones(B, 1, device=points.device)
        pts_homo = torch.cat([points, ones], dim=1)
        vox_coords = (self.world_to_vox_transform @ pts_homo.T).T
        vox_xyz = vox_coords[:, :3]
        norm_coords = 2.0 * (vox_xyz / (self.grid_dims - 1.0)) - 1.0
        return norm_coords.view(1, 1, 1, B, 3)

    def forward(self, pred_positions):
        if not pred_positions.requires_grad:
            pred_positions.requires_grad_(True)
        
        grid_coords_raw = self.world_to_norm_grid(pred_positions)
        coords_flat = grid_coords_raw.view(-1, 3)
        
        dist_outside = torch.clamp(torch.abs(coords_flat) - 1.0, min=0.0)
        gravity_force = torch.norm(dist_outside, dim=1, keepdim=True)
        gravity_grad = -1.0 * coords_flat * (gravity_force > 0).float()
        
        sdf_val = F.grid_sample(self.sdf, grid_coords_raw, mode='bilinear', padding_mode='zeros', align_corners=True)
        sdf_val = sdf_val.view(-1, 1)

        violation = F.relu(sdf_val)
        
        grad_outputs = torch.ones_like(violation)
        if violation.requires_grad:
            sdf_grad = torch.autograd.grad(outputs=violation, inputs=pred_positions, grad_outputs=grad_outputs, create_graph=False, retain_graph=True, only_inputs=True)[0]
        else:
            sdf_grad = torch.zeros_like(pred_positions)
        
        wall_correction = violation * sdf_grad.detach() 
        gravity_correction = gravity_force * -gravity_grad.detach()

        total_correction = wall_correction + (0.5 * gravity_correction)
        constrained_pos = pred_positions - total_correction
        
        return constrained_pos, violation + gravity_force

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
        self.num_patches = (img_size // self.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, t_frames, self.num_patches, dim))
        self.blocks = nn.ModuleList([SpatioTemporalBlock(dim, heads, t_frames) for _ in range(layers)])
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, video_tensor):
        B, T, C, H, W = video_tensor.shape
        x = video_tensor.view(B*T, C, H, W)
        x = self.patch_embed(x)
        curr_patches_h = x.shape[2]
        curr_patches_w = x.shape[3]
        curr_num_patches = curr_patches_h * curr_patches_w
        x = x.flatten(2).transpose(1, 2)
        
        if curr_num_patches != self.num_patches:
            pos = self.pos_embed
            T_dim = pos.shape[1]
            orig_size = int(self.num_patches ** 0.5)
            pos_grid = rearrange(pos, '1 t (h w) d -> (1 t) d h w', h=orig_size, w=orig_size)
            pos_new = F.interpolate(pos_grid, size=(curr_patches_h, curr_patches_w), mode='bicubic', align_corners=False)
            pos = rearrange(pos_new, '(b t) d h w -> b t (h w) d', b=1, t=T_dim)
        else:
            pos = self.pos_embed

        pos = repeat(pos, '1 t n d -> (b t) n d', b=B)
        x = x + pos
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = x.view(B, T, -1)
        x = self.proj_out(x)
        return x

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
        return x

class DeepLungST(nn.Module):
    def __init__(self, t_frames, sdf_volume_tensor, grid_transform_matrix, mode='s', norm_center=None, norm_scale=1.0):
        super().__init__()
        self.t_frames = t_frames
        self.config = MODEL_CONFIGS[mode]
        embed_dim = self.config['embed_dim']
        self.K = self.config['num_particles']
        
        # --- Normalization Params ---
        self.register_buffer('norm_center', norm_center if norm_center is not None else torch.zeros(3))
        self.norm_scale = norm_scale
        
        self.visual_encoder = STViViT(self.config, img_size=128, t_frames=t_frames)
        self.map_encoder = MapEncoderGAT(self.config, in_channels=11)
        self.visual_proj = nn.Linear(embed_dim, embed_dim)
        self.map_proj = nn.Linear(embed_dim, embed_dim)
        self.fusion = nn.MultiheadAttention(embed_dim, num_heads=self.config['num_heads'], batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.kinematic_rnn = nn.LSTM(embed_dim, self.config['lstm_hidden'], batch_first=True)
        self.velocity_head = nn.Linear(self.config['lstm_hidden'], 3)
        self.noise_scale_head = nn.Linear(self.config['lstm_hidden'], 3)
        self.constraint = DifferentiableSDFConstraint(sdf_volume_tensor, grid_transform_matrix)
        
        # --- New Components for Multi-Hypothesis ---
        self.rel_pos_proj = nn.Linear(3, embed_dim)
        self.particle_embed = nn.Linear(3, embed_dim)

    def forward(self, video, map_nodes, map_edges, map_edge_attr, initial_pose=None, physics_on=True, resample=False):
        B = video.shape[0]
        K = self.K
        
        # Normalize Map Nodes
        map_nodes_norm = (map_nodes - self.norm_center) / self.norm_scale
        
        map_nodes_embed = self.map_encoder(map_nodes_norm, map_edges, map_edge_attr)
        visual_tokens = self.visual_encoder(video)
        
        # Frame-wise projection for Dense Retrieval Loss
        visual_feat_proj = self.visual_proj(visual_tokens) # (B, T, D)
        map_feat_proj = self.map_proj(map_nodes_embed)     # (N, D)    
        
        # --- Prepare Particle States ---
        if initial_pose is None:
            initial_pose = torch.zeros(B, 3, device=video.device)
            
        if initial_pose.dim() == 3: 
            current_pos = rearrange(initial_pose, 'b k d -> (b k) d')
        else: 
            current_pos = repeat(initial_pose, 'b d -> (b k) d', k=K)
            current_pos = current_pos + (torch.randn_like(current_pos) * 0.02)

        # --- Relative Map Encoding ---
        # map_nodes_norm: (N, 3)
        # current_pos: (B*K, 3)
        # We need to normalize current_pos to match map_nodes_norm scale for meaningful relative diff
        current_pos_norm = (current_pos - self.norm_center) / self.norm_scale
        
        # (B*K, 1, 3) - (1, N, 3) -> (B*K, N, 3)
        rel_pos = map_nodes_norm.unsqueeze(0) - current_pos_norm.unsqueeze(1)
        rel_pos_embed = self.rel_pos_proj(rel_pos) # (B*K, N, D)
        
        map_embed_expanded = repeat(map_nodes_embed, 'n d -> (b k) n d', b=B, k=K)
        map_embed_with_rel = map_embed_expanded + rel_pos_embed
        
        visual_tokens_expanded = repeat(visual_tokens, 'b t d -> (b k) t d', k=K)
        
        # Fusion with Particle Awareness
        context, _ = self.fusion(visual_tokens_expanded, map_embed_with_rel, map_embed_with_rel)
        context = self.norm(context + visual_tokens_expanded) 
        
        # Inject Particle State into Context
        particle_feat = self.particle_embed(current_pos_norm) # (B*K, D)
        context = context + particle_feat.unsqueeze(1)
        
        rnn_out, _ = self.kinematic_rnn(context) 
        velocities = self.velocity_head(rnn_out)   # (B*K, T, 3)
        noise_scales = torch.sigmoid(self.noise_scale_head(rnn_out)) # (B*K, T, 3)
        
        # Reshape for easier indexing during resampling
        velocities = rearrange(velocities, '(b k) t d -> b k t d', b=B, k=K)
        noise_scales = rearrange(noise_scales, '(b k) t d -> b k t d', b=B, k=K)
        current_pos = rearrange(current_pos, '(b k) d -> b k d', b=B, k=K)
        
        traj_pos = []
        violations = []
        
        for t in range(self.t_frames):
            # 1. Resampling Step (SIR)
            # Only resample if enabled and not at the very first step (need some movement/evidence)
            if resample and t > 0 and t % 3 == 0: # Resample every 3 frames
                # Calculate Weights
                # current_pos: (B, K, 3) (Normalized)
                # map_nodes_norm: (N, 3)
                
                # Find nearest map node for each particle
                # (B, K, 1, 3) - (1, 1, N, 3) -> (B, K, N, 3) -> norm -> (B, K, N)
                # Optimization: Flatten B*K
                flat_pos = rearrange(current_pos, 'b k d -> (b k) d')
                dists = torch.cdist(flat_pos, map_nodes_norm) # (B*K, N)
                nearest_idx = torch.argmin(dists, dim=1) # (B*K)
                
                # Gather map embeddings
                # map_feat_proj: (N, D)
                nearest_map_feats = map_feat_proj[nearest_idx] # (B*K, D)
                
                # Get current visual embedding
                # visual_feat_proj: (B, T, D) -> (B, D) at time t
                curr_vis_feat = visual_feat_proj[:, t, :] 
                curr_vis_feat = repeat(curr_vis_feat, 'b d -> (b k) d', k=K)
                
                # Dot product similarity
                scores = torch.sum(nearest_map_feats * curr_vis_feat, dim=1) # (B*K)
                scores = rearrange(scores, '(b k) -> b k', b=B, k=K)
                
                # Softmax to get weights
                weights = F.softmax(scores / 0.1, dim=1) # Temperature 0.1
                
                # Resample
                # We use Categorical to sample indices
                dist = Categorical(probs=weights)
                indices = dist.sample((K,)).T # (B, K)
                
                # Update State
                # Gather from batch dimension
                batch_indices = torch.arange(B, device=video.device).unsqueeze(1).expand(B, K)
                
                current_pos = current_pos[batch_indices, indices]
                velocities = velocities[batch_indices, indices]
                noise_scales = noise_scales[batch_indices, indices]
            
            # 2. Kinematic Update
            vel_mean = velocities[:, :, t, :] # (B, K, 3)
            noise_sigma = noise_scales[:, :, t, :] * (0.05 if physics_on else 0.0)
            
            step_noise = torch.randn_like(vel_mean) * noise_sigma
            proposal_norm = current_pos + vel_mean + step_noise
            
            if physics_on:
                # Flatten for constraint
                prop_flat = rearrange(proposal_norm, 'b k d -> (b k) d')
                
                # 1. Norm -> World (mm)
                proposal_world = (prop_flat * self.norm_scale) + self.norm_center
                # 2. Check SDF
                constrained_world, viol = self.constraint(proposal_world)
                # 3. World -> Norm
                constrained_norm = (constrained_world - self.norm_center) / self.norm_scale
                
                current_pos = rearrange(constrained_norm, '(b k) d -> b k d', b=B, k=K)
                violations.append(viol)
            else:
                current_pos = proposal_norm
                violations.append(torch.zeros(B*K, 1, device=video.device))
                
            traj_pos.append(current_pos)
            
        traj_stack = torch.stack(traj_pos, dim=2) # (B, K, T, 3)
        viol_stack = torch.stack(violations, dim=1) # (B*K, T, 1) -> need reshape
        
        viol_out = rearrange(viol_stack, '(b k) t d -> b k t d', b=B, k=K)
        
        return traj_stack, viol_out, visual_feat_proj, map_feat_proj