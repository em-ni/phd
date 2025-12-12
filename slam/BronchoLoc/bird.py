"""
BIRD: Bronchial Inference Route Determination

Global memory module that refines ANT's local predictions using Titans long-term memory.
Accumulates trajectory history and cross-attends to the full centerline to correct 
errors at bifurcations.
"""
import torch
import torch.nn as nn
from titans_pytorch import NeuralMemory

from ant import MODEL_CONFIGS
from constants import NORM_MAP_SCALE


# ==============================================================================
# BIRD CONFIGURATIONS
# ==============================================================================
# Configs that depend on ANT model size
BIRD_CONFIGS = {
    's': {
        'memory_dim': 128,
        'num_memory_layers': 2,
        'num_heads': 4,
    },
    'b': {
        'memory_dim': 256,
        'num_memory_layers': 2,
        'num_heads': 4,
    },
    'm': {
        'memory_dim': 256,
        'num_memory_layers': 2,
        'num_heads': 8,
    },
    'l': {
        'memory_dim': 512,
        'num_memory_layers': 3,
        'num_heads': 8,
    }
}

class CenterlineEncoder(nn.Module):
    """
    Pre-encodes the downsampled centerline points into feature vectors.
    This is applied once at the start and cached.
    """
    def __init__(self, memory_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, memory_dim)
        )
    
    def forward(self, centerline):
        """
        Args:
            centerline: (N, 3) centerline points in global frame
        Returns:
            (N, D) encoded centerline features
        """
        return self.encoder(centerline)


class BIRD(nn.Module):
    """
    Bronchial Inference Route Determination - Global Memory Module.
    
    Takes frozen ANT predictions and visual features, maintains a memory of the 
    trajectory history, and refines predictions by cross-attending to the full
    centerline for global consistency.
    
    Args:
        ant_mode: ANT model configuration ('s', 'b', 'm', 'l') - used to get embed_dim
        memory_dim: Titans memory hidden dimension
        num_memory_layers: Number of MLP layers in Titans neural memory
        num_heads: Number of attention heads for cross-attention
        num_centerline_pts: Expected size of downsampled centerline
    """
    def __init__(self, 
                 ant_mode='m',
                 memory_dim=256,
                 num_memory_layers=2,
                 num_heads=4,
                 num_centerline_pts=1024):
        super().__init__()
        
        # Get ANT visual feature dimension from config
        ant_config = MODEL_CONFIGS[ant_mode]
        self.visual_dim = ant_config['embed_dim']
        self.memory_dim = memory_dim
        self.ant_mode = ant_mode
        
        # Project ANT features (visual + local pred) to memory dimension
        # Input: visual_tokens (B, T, visual_dim) + p_local (B, T, 3)
        self.input_proj = nn.Linear(self.visual_dim + 3, memory_dim)
        
        # Titans Neural Memory
        # Uses MLP as memory that learns to store trajectory information
        # Surprise-based updates prioritize bifurcation events
        self.memory = NeuralMemory(
            dim=memory_dim,
            chunk_size=16,  # Window size - processes in chunks
            num_memory_layers=num_memory_layers,
        )
        
        # Centerline encoder (applied once, cached)
        self.centerline_encoder = CenterlineEncoder(memory_dim)
        
        # Cross-attention: memory state queries the full centerline
        # This allows global context to influence predictions
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer norms for stability
        self.norm1 = nn.LayerNorm(memory_dim)
        self.norm2 = nn.LayerNorm(memory_dim)
        
        # Refinement head
        # Predicts a RESIDUAL correction: p_global = p_local + delta
        # This makes training easier as delta starts near zero
        self.refine_head = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, 3)
        )
        
        # Initialize refinement head to output near-zero
        # So initially p_global ≈ p_local
        nn.init.zeros_(self.refine_head[-1].weight)
        nn.init.zeros_(self.refine_head[-1].bias)
        
    def encode_centerline(self, centerline):
        """
        Pre-encode the centerline for efficient cross-attention.
        Call this once at the start of inference/training.
        
        Args:
            centerline: (N, 3) downsampled centerline in normalized coordinates
        Returns:
            (N, D) encoded centerline
        """
        return self.centerline_encoder(centerline)
        
    def forward(self, p_local, visual_tokens, centerline_encoded, mem_state=None,
                first_frame_pos=None, first_frame_quat=None):
        """
        Args:
            p_local: (B, T, 3) local predictions from ANT (in local/normalized coords)
            visual_tokens: (B, T, D) visual features from frozen ANT
            centerline_encoded: (N, D) pre-encoded centerline features
            mem_state: Previous memory state (for sequential inference)
            first_frame_pos: (B, 3) position of first frame (for global transform, optional)
            first_frame_quat: (B, 4) quaternion of first frame (for global transform, optional)
            
        Returns:
            p_global: (B, T, 3) refined prediction (in same coords as p_local)
            new_mem_state: Updated memory state
        """
        B, T, _ = p_local.shape
        N, D = centerline_encoded.shape
        
        # Combine local prediction with visual features
        x = torch.cat([p_local, visual_tokens], dim=-1)  # (B, T, visual_dim + 3)
        x = self.input_proj(x)  # (B, T, memory_dim)
        
        # Update Titans memory
        # The memory learns to store trajectory history and detect "surprises"
        # (e.g., unexpected bifurcation choices)
        mem_out, new_mem_state = self.memory(x, state=mem_state)  # (B, T, memory_dim)
        mem_out = self.norm1(mem_out)
        
        # Expand centerline for batch processing
        cl_embed = centerline_encoded.unsqueeze(0).expand(B, -1, -1)  # (B, N, memory_dim)
        
        # Cross-attention: memory queries centerline for global context
        # This allows the model to "look at the whole map" and correct mistakes
        attn_out, _ = self.cross_attn(
            query=mem_out,       # (B, T, memory_dim)
            key=cl_embed,        # (B, N, memory_dim)
            value=cl_embed
        )  # (B, T, memory_dim)
        
        attn_out = self.norm2(attn_out + mem_out)  # Residual connection
        
        # Refinement: predict correction delta
        delta = self.refine_head(attn_out)  # (B, T, 3)
        
        # Residual prediction: p_global = p_local + delta
        # The delta is learned to correct errors in local predictions
        p_global = p_local + delta
        
        return p_global, new_mem_state
    
    def reset_memory(self):
        """
        Reset memory state for new trajectory.
        Call this at the start of each new bronchoscopy session.
        """
        return None  # Memory state is None initially

def create_bird(ant_mode='m', num_centerline_pts=1024):
    """
    Factory function to create a BIRD module with config matching ANT.
    
    Args:
        ant_mode: ANT model mode ('s', 'b', 'm', 'l')
        num_centerline_pts: Size of downsampled centerline
        
    Returns:
        BIRD module configured for the given ANT mode
    """
    config = BIRD_CONFIGS[ant_mode]
    return BIRD(
        ant_mode=ant_mode,
        memory_dim=config['memory_dim'],
        num_memory_layers=config['num_memory_layers'],
        num_heads=config['num_heads'],
        num_centerline_pts=num_centerline_pts
    )


