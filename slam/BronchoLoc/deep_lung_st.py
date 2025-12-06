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
# Model variants with different capacities: small (s), base (b), medium (m), large (l)
MODEL_CONFIGS = {
    's': {
        'embed_dim': 16,
        'num_heads': 2,
        'vi_layers': 1,
        'gat_heads': 2,
        'lstm_hidden': 32
    },
    'b': {
        'embed_dim': 512,
        'num_heads': 8,
        'vi_layers': 8,
        'gat_heads': 8,
        'lstm_hidden': 256
    },
    'm': {
        'embed_dim': 1024,
        'num_heads': 16,
        'vi_layers': 12,
        'gat_heads': 8,
        'lstm_hidden': 512
    },
    'l': {
        'embed_dim': 2048,
        'num_heads': 32,
        'vi_layers': 24,
        'gat_heads': 16,
        'lstm_hidden': 1024
    }
}


class SpatioTemporalBlock(nn.Module):
    """
    Transformer block that processes both spatial and temporal dependencies.
    It uses separated attention mechanisms:
    1. Spatial Attention: Attends between patches within the same frame.
    2. Temporal Attention: Attends between the same spatial location across different frames.
    """
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
        # x: (B*T, N_patches, D) - Flattened batch of frames
        
        # 1. Spatial Attention
        # Standard self-attention over N_patches
        attn_out, _ = self.spatial_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Reshape for Temporal Attention
        bt, n, d = x.shape
        b = bt // self.t_frames
        
        # Rearrange to group by spatial location: (B, N, T, D)
        # We treat (B*N) as the batch dimension and T as the sequence dimension.
        # This allows each patch to attend to its history/future self.
        x = rearrange(x, '(b t) n d -> (b n) t d', b=b, t=self.t_frames)
        
        # 2. Temporal Attention
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.norm2(x + attn_out)
        
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
    def __init__(self, config, img_size=128, t_frames=16):
        super().__init__()
        dim = config['embed_dim']
        layers = config['vi_layers']
        heads = config['num_heads']
        self.patch_size = 16
        
        # Convolutional Patch Embedding: Splits image into 16x16 patches and projects to dim
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.num_patches = (img_size // self.patch_size) ** 2
        
        # Learnable Positional Embedding: (1, T, N, D) adds info about space and time
        self.pos_embed = nn.Parameter(torch.randn(1, t_frames, self.num_patches, dim))
        
        # Stack of SpatioTemporal Blocks
        self.blocks = nn.ModuleList([SpatioTemporalBlock(dim, heads, t_frames) for _ in range(layers)])
        
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
        x = x + pos
        
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
    """
    def __init__(self, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )
        
    def forward(self, x):
        # x: (B, T, K, 3)
        B, T, K, C = x.shape
        x = x.view(B * T, K, C)
        
        # Point-wise MLP applied to each map point
        x = self.mlp(x) # (B*T, K, out_dim)
        
        x = x.view(B, T, K, -1)
        return x

class ActionPredictor(nn.Module):
    """
    Main Model Class: Graph Selector Architecture.
    Predicts the next position by attending to visual features and selecting/weighting 
    candidates from the K local map points.
    """
    def __init__(self, t_frames, mode='s', img_size=128): 
        super().__init__()
        self.t_frames = t_frames
        self.config = MODEL_CONFIGS[mode]
        embed_dim = self.config['embed_dim']
        
        # Visual Encoder
        self.visual_encoder = STViViT(self.config, img_size=img_size, t_frames=t_frames)
        
        # Map Encoder
        # We ensure map features match visual feature dimension for dot product attention.
        self.map_dim = embed_dim 
        self.map_encoder = MapEncoder(out_dim=self.map_dim)
        
        # Selection Head (Attention)
        # Projects visual features to query space
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        
        # Projects map features to key space
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = embed_dim ** -0.5

    def forward(self, video, map_points=None):
        """
        Args:
            video: Video tensor (B, T, C, H, W)
            map_points: Local map candidates (B, T, K, 3)
            
        Returns:
            pred_delta: Predicted position update (B, T, 3)
        """
        # video: (B, T, C, H, W)
        # map_points: (B, T, K, 3)
        
        if map_points is None:
            # Fallback (shouldn't happen in normal flow)
            return torch.zeros(video.shape[0], video.shape[1], 3, device=video.device)
            
        # 1. Extract Visual Features (Query)
        # Get frame-wise visual embeddings: (B, T, D)
        visual_tokens = self.visual_encoder(video)
        queries = self.query_proj(visual_tokens) # (B, T, D)
        
        # 2. Extract Map Features (Keys)
        # Get point-wise map embeddings: (B, T, K, D)
        map_features = self.map_encoder(map_points)
        keys = self.key_proj(map_features) # (B, T, K, D)
        
        # 3. Compute Attention Scores (Dot Product)
        # We calculate similarity between visual state (Query) and each map point (Key).
        # Q: (B, T, 1, D)
        # K: (B, T, K, D)
        # Scores: (B, T, K) - Each map point gets a score at each timestep.
        queries = queries.unsqueeze(2) # (B, T, 1, D)
        scores = torch.sum(queries * keys, dim=-1) * self.scale # (B, T, K)
        
        # 4. Compute Probabilities
        # Softmax normalizes scores to sum to 1.
        probs = F.softmax(scores, dim=-1) # (B, T, K)
        
        # 5. Weighted Sum of Map Points (Soft Selection)
        # We predict the delta by taking the expected value over the map points.
        # This constrains predictions to lie in the convex hull of the map candidates
        # (effectively "selecting" a point on the centerline).
        # (B, T, K, 1) * (B, T, K, 3) -> (B, T, K, 3) -> Sum -> (B, T, 3)
        pred_delta = torch.sum(probs.unsqueeze(-1) * map_points, dim=2)
        
        # Return only translation (B, T, 3)
        # We rely on the map's geometry.
        return pred_delta