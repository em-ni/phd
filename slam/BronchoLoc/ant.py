import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# ==============================================================================
# CONFIGURATIONS
# ==============================================================================
# Model variants with different capacities: small (s), base (b), medium (m), large (l)
# Configurable dimensions:
#   - embed_dim: Main embedding dimension (must be divisible by num_heads)
#   - num_heads: Number of attention heads (embed_dim must be divisible by this)
#   - vi_layers: Number of SpatioTemporal transformer blocks
#   - mlp_expansion: Expansion factor for transformer FFN (hidden = embed_dim * mlp_expansion)
#   - patch_size: Size of image patches for ViT (img_size must be divisible by this)
#   - map_encoder_hidden: List of hidden layer sizes for MapEncoder MLP (input is 3, output is embed_dim)
MODEL_CONFIGS = {
    # try to keep head_dim = 64
    'xs': {  # Extra-Small: ~1.5M params
        'embed_dim': 128,
        'num_heads': 2,
        'vi_layers': 4,
        'mlp_expansion': 4,
        'patch_size': 16,
        'map_encoder_hidden': [32, 64],
        'suggested_lr': 2e-4,
        'dropout': 0.1
    },
    's': {  # Small: ~7M params
        'embed_dim': 256,
        'num_heads': 4,       
        'vi_layers': 6,
        'mlp_expansion': 4,
        'patch_size': 16,
        'map_encoder_hidden': [64, 128],
        'suggested_lr': 1e-4,
        'dropout': 0.1
    },
    'b': {  # Base: ~50M params
        'embed_dim': 512,
        'num_heads': 8, 
        'vi_layers': 12,
        'mlp_expansion': 4,
        'patch_size': 16,
        'map_encoder_hidden': [128, 256],
        'suggested_lr': 5e-5,
        'dropout': 0.2
    },
    'm': {  # Medium: ~115M params
        'embed_dim': 768,
        'num_heads': 12,
        'vi_layers': 12,
        'mlp_expansion': 4,
        'patch_size': 16,
        'map_encoder_hidden': [192, 384],
        'suggested_lr': 2e-5,
        'dropout': 0.2
    },
    'l': {  # Large: ~400M params
        'embed_dim': 1024,
        'num_heads': 16,
        'vi_layers': 24,
        'mlp_expansion': 4,
        'patch_size': 16,
        'map_encoder_hidden': [256, 512],
        'suggested_lr': 1e-5,
        'dropout': 0.3
    }
}


class SpatioTemporalBlock(nn.Module):
    """
    Transformer block that processes both spatial and temporal dependencies.
    It uses separated attention mechanisms:
    1. Spatial Attention: Attends between patches within the same frame.
    2. Temporal Attention: Attends between the same spatial location across different frames.
    """
    def __init__(self, dim, num_heads, window_size, mlp_expansion=4, dropout=0.1):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        hidden_dim = dim * mlp_expansion
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.window_size = window_size

    def forward(self, x):
        # x: (B*T, N_patches, D) - Flattened batch of frames
        
        # 1. Spatial Attention
        # Standard self-attention over N_patches
        attn_out, _ = self.spatial_attn(x, x, x)
        x = self.norm1(x + self.drop1(attn_out))
        
        # Reshape for Temporal Attention
        bt, n, d = x.shape
        b = bt // self.window_size
        
        # Rearrange to group by spatial location: (B, N, T, D)
        # We treat (B*N) as the batch dimension and T as the sequence dimension.
        # This allows each patch to attend to its history/future self.
        x = rearrange(x, '(b t) n d -> (b n) t d', b=b, t=self.window_size)
        
        # 2. Temporal Attention
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.norm2(x + self.drop2(attn_out))
        
        # Restore original shape: (B*T, N, D)
        x = rearrange(x, '(b n) t d -> (b t) n d', b=b)
        
        # 3. Feed Forward Network
        x = self.norm3(x + self.mlp(x))
        return x

class STViViT(nn.Module):
    """
    Spatio-Temporal Vision Transformer (ViViT-style).
    Extracts features from a sequence of video frames.
    """
    def __init__(self, config, img_size=128, window_size=16):
        super().__init__()
        dim = config['embed_dim']
        layers = config['vi_layers']
        heads = config['num_heads']
        mlp_expansion = config.get('mlp_expansion', 4)
        dropout = config.get('dropout', 0.1)
        self.patch_size = config.get('patch_size', 16)
        
        # Convolutional Patch Embedding: Splits image into patches and projects to dim
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.num_patches = (img_size // self.patch_size) ** 2
        
        # Learnable Positional Embedding: (1, T, N, D) adds info about space and time
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, self.num_patches, dim))
        
        # Dropout after positional embedding
        self.pos_drop = nn.Dropout(dropout)
        
        # Stack of SpatioTemporal Blocks (now with dropout)
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(dim, heads, window_size, mlp_expansion, dropout=dropout) 
            for _ in range(layers)
        ])
        
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, video_tensor):
        """
        Args:
            video_tensor: (B, T, C, H, W)
        Returns:
            x: (B, T, D) - One feature vector per frame.
        """
        B, T, C, H, W = video_tensor.shape
        x = video_tensor.view(B*T, C, H, W)
        
        # Patch Embedding
        x = self.patch_embed(x) # (B*T, D, H/16, W/16)
        curr_patches_h = x.shape[2]
        curr_patches_w = x.shape[3]
        curr_num_patches = curr_patches_h * curr_patches_w
        
        # Flatten spatial dims: (B*T, N, D)
        x = x.flatten(2).transpose(1, 2)
        
        # Handle Resolution Mismatch (Interpolate Position Embeddings if needed)
        # This allows the model to work on different image sizes than trained on.
        if curr_num_patches != self.num_patches:
            pos = self.pos_embed
            T_dim = pos.shape[1]
            orig_size = int(self.num_patches ** 0.5)
            # Rearrange to 2D grid
            pos_grid = rearrange(pos, '1 t (h w) d -> (1 t) d h w', h=orig_size, w=orig_size)
            # Interpolate
            pos_new = F.interpolate(pos_grid, size=(curr_patches_h, curr_patches_w), mode='bicubic', align_corners=False)
            # Flatten back
            pos = rearrange(pos_new, '(b t) d h w -> b t (h w) d', b=1, t=T_dim)
        else:
            pos = self.pos_embed

        # Add Positional Embeddings
        pos = repeat(pos, '1 t n d -> (b t) n d', b=B)
        x = self.pos_drop(x + pos)
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
            
        # Global Average Pooling over Spatial Patches
        # (B*T, N, D) -> (B*T, D)
        x = x.mean(dim=1)
        
        # Reshape to (B, T, D)
        x = x.view(B, T, -1)
        x = self.proj_out(x)
        return x

class MapEncoder(nn.Module):
    """
    Encodes local map points (K, 3) into feature vectors (K, D).
    Used to embed the topological map context.
    No pooling is applied because we want to attend to individual points.
    
    Args:
        out_dim: Output dimension
        hidden_sizes: List of hidden layer sizes (default: [64, 128])
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(self, out_dim=64, hidden_sizes=None, dropout=0.1):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 128]
        
        # Build MLP dynamically based on hidden_sizes
        layers = []
        in_dim = 3  # Input is always 3D coordinates
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, out_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        # x: (B, T, K, 3)
        B, T, K, C = x.shape
        x = x.view(B * T, K, C)
        
        # Point-wise MLP applied to each map point
        x = self.mlp(x) # (B*T, K, out_dim)
        
        x = x.view(B, T, K, -1)
        return x

class VOHead(nn.Module):
    """
    Visual Odometry Head: Predicts delta pose (position + quaternion) from visual features.
    
    At inference, these deltas are chained to track absolute pose.
    Training uses GT delta poses as supervision.
    """
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        
        # Shared feature projection
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Position delta head: predicts (dx, dy, dz) in local frame
        self.pos_head = nn.Linear(embed_dim, 3)
        
        # Orientation delta head: predicts (dqx, dqy, dqz, dqw)
        # Output is normalized to unit quaternion
        self.quat_head = nn.Linear(embed_dim, 4)
    
    def forward(self, visual_tokens):
        """
        Args:
            visual_tokens: (B, T, D) - per-frame visual embeddings
            
        Returns:
            delta_pos: (B, T, 3) - predicted position delta
            delta_quat: (B, T, 4) - predicted quaternion delta (normalized)
        """
        x = F.gelu(self.fc1(visual_tokens))
        x = self.dropout(x)
        
        # Position delta (unnormalized, will be matched to GT)
        delta_pos = self.pos_head(x)  # (B, T, 3)
        
        # Quaternion delta (normalized to unit quaternion)
        delta_quat = self.quat_head(x)  # (B, T, 4)
        delta_quat = F.normalize(delta_quat, p=2, dim=-1)  # Unit quaternion
        
        return delta_pos, delta_quat


class ActionPredictor(nn.Module):
    """
    Main Model Class: Centerline Selector Architecture with Visual Odometry.
    
    Uses visual features to predict pose (position + orientation) and then
    attends to centerline candidates to select the best matching position.
    """
    def __init__(self, window_size, mode='s', img_size=128): 
        super().__init__()
        self.window_size = window_size
        self.config = MODEL_CONFIGS[mode]
        embed_dim = self.config['embed_dim']
        map_encoder_hidden = self.config.get('map_encoder_hidden', [64, 128])
        dropout = self.config.get('dropout', 0.1)
        
        # Visual Encoder
        self.visual_encoder = STViViT(self.config, img_size=img_size, window_size=window_size)
        
        # Visual Odometry Head (predicts delta pose from visual tokens)
        self.vo_head = VOHead(embed_dim, dropout=dropout)
        
        # Map Encoder
        # We ensure map features match visual feature dimension for dot product attention.
        self.map_dim = embed_dim 
        self.map_encoder = MapEncoder(out_dim=self.map_dim, hidden_sizes=map_encoder_hidden, dropout=dropout)
        
        # Selection Head (Attention)
        # Query includes: visual_tokens (D) + delta_pos (3) + delta_quat (4) = D+7
        # Project back to embed_dim for matching
        self.query_proj = nn.Linear(embed_dim + 7, embed_dim)
        
        # Projects map features to key space
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = embed_dim ** -0.5

    def forward(self, video, map_points=None, map_mask=None, return_features=False):
        """
        Args:
            video: Video tensor (B, T, C, H, W)
            map_points: Local map candidates (B, T, K, 3)
            map_mask: Boolean mask (B, T, K) - True for valid points, False for padding
            return_features: If True, also return visual_tokens for BIRD module
            
        Returns:
            pred_delta: Predicted position update (B, T, 3)
            visual_tokens: (optional) Visual features (B, T, D) if return_features=True
        """
        # video: (B, T, C, H, W)
        # map_points: (B, T, K, 3)
        # map_mask: (B, T, K)
        
        if map_points is None:
            # Fallback (shouldn't happen in normal flow)
            B, T = video.shape[0], video.shape[1]
            D = self.config['embed_dim']
            zeros_pos = torch.zeros(B, T, 3, device=video.device)
            zeros_quat = torch.zeros(B, T, 4, device=video.device)
            zeros_visual = torch.zeros(B, T, D, device=video.device)
            zeros_probs = torch.zeros(B, T, 1, device=video.device)
            if return_features:
                return zeros_pos, zeros_pos, zeros_quat, zeros_visual, zeros_probs
            return zeros_pos, zeros_quat
            
        # 1. Extract Visual Features
        # Get frame-wise visual embeddings: (B, T, D)
        visual_tokens = self.visual_encoder(video)
        
        # 2. Visual Odometry: Predict delta pose from visual tokens
        # delta_pos: (B, T, 3), delta_quat: (B, T, 4)
        delta_pos, delta_quat = self.vo_head(visual_tokens)
        
        # 3. Build Query: visual_tokens + delta_pos + delta_quat
        # This allows the model to use predicted pose when selecting candidates
        query_input = torch.cat([visual_tokens, delta_pos, delta_quat], dim=-1)  # (B, T, D+7)
        queries = self.query_proj(query_input)  # (B, T, D)
        
        # 4. Extract Map Features (Keys)
        # Get point-wise map embeddings: (B, T, K, D)
        map_features = self.map_encoder(map_points)
        keys = self.key_proj(map_features)  # (B, T, K, D)
        
        # 5. Compute Attention Scores (Dot Product)
        # Q: (B, T, 1, D), K: (B, T, K, D) -> Scores: (B, T, K)
        queries = queries.unsqueeze(2)  # (B, T, 1, D)
        scores = torch.sum(queries * keys, dim=-1) * self.scale  # (B, T, K)
        
        # 6. Apply Mask (set padding positions to -inf before softmax)
        if map_mask is not None:
            scores = scores.masked_fill(~map_mask, float('-inf'))
        
        # 7. Compute Probabilities
        probs = F.softmax(scores, dim=-1)  # (B, T, K)
        probs = torch.nan_to_num(probs, nan=0.0)
        
        # 8. Weighted Sum of Map Points (Soft Selection)
        # Output position is constrained to convex hull of candidates
        pred_pos = torch.sum(probs.unsqueeze(-1) * map_points, dim=2)  # (B, T, 3)
        
        if return_features:
            # Return all features for BIRD:
            # - pred_pos: selected position from candidates (B, T, 3)
            # - delta_pos: VO position estimate (B, T, 3)  
            # - delta_quat: VO orientation estimate (B, T, 4)
            # - visual_tokens: visual features (B, T, D)
            # - probs: attention weights over candidates (B, T, K)
            return pred_pos, delta_pos, delta_quat, visual_tokens, probs
        return pred_pos, delta_quat