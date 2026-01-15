"""
BIRD: Bronchial Intraoperative Route Discriminator

Global memory module that refines ANT's local predictions using Titans long-term memory.
Accumulates trajectory history and cross-attends to the full centerline to SELECT
a point on the centerline (attention-based, like ANT), ensuring predictions stay on-airway.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from titans_pytorch import NeuralMemory

from ant import MODEL_CONFIGS
from constants import NORM_MAP_SCALE, load_window_config


# ==============================================================================
# BIRD CONFIGURATIONS
# ==============================================================================
# Configs that depend on ANT model size
# Note: chunk_size is not included here - it's always loaded from window_config
BIRD_CONFIGS = {
    's': {
        'memory_dim': 128,
        'num_heads': 4,
    },
    'b': {
        'memory_dim': 256,
        'num_heads': 4,
    },
    'm': {
        'memory_dim': 256,
        'num_heads': 8,
    },
    'l': {
        'memory_dim': 512,
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
    Bronchial Intraoperative Route Discriminator - Global Memory Module.
    
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
                 chunk_size=10,
                 num_heads=4,
                 num_centerline_pts=1024,
                 distance_penalty_strength=5.0,
                 dropout=0.1):
        super().__init__()
        
        # Get ANT visual feature dimension from config
        ant_config = MODEL_CONFIGS[ant_mode]
        self.visual_dim = ant_config['embed_dim']
        self.memory_dim = memory_dim
        self.ant_mode = ant_mode
        
        # Project ANT features to memory dimension
        # Input: ant_trajectory (3) + vo_trajectory [delta_pos (3) + delta_quat (4)] + visual_tokens (D)
        self.input_proj = nn.Linear(3 + 3 + 4 + self.visual_dim, memory_dim)
        
        # Titans Neural Memory
        # Uses MLP as memory that learns to store trajectory information
        # Surprise-based updates prioritize bifurcation events
        self.memory = NeuralMemory(
            dim=memory_dim,
            chunk_size=chunk_size,
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
        
        # Selection head: computes attention scores over centerline points
        # Instead of predicting delta, we predict which centerline point to select
        # Output: (B, T, memory_dim) query for dot-product attention with centerline
        self.query_head = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim, memory_dim)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Scale for attention scores
        self.scale = memory_dim ** -0.5
        
        # Distance penalty strength for soft locality constraint
        # Higher values = stronger preference for points near ANT's prediction
        # But can be overcome by strong attention (e.g., for branch correction)
        self.distance_penalty_strength = distance_penalty_strength
        
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
        
    def forward(self, ant_pos, delta_pos, delta_quat, visual_tokens, centerline_encoded, centerline_points, mem_state=None):
        """
        Args:
            ant_pos: (B, T, 3) candidate-selected positions from ANT
            delta_pos: (B, T, 3) VO position estimates from ANT
            delta_quat: (B, T, 4) VO orientation estimates from ANT
            visual_tokens: (B, T, D) visual features from frozen ANT
            centerline_encoded: (N, D) pre-encoded centerline features
            centerline_points: (N, 3) raw centerline points (normalized coordinates)
            mem_state: Previous memory state (for streaming inference)
            
        Returns:
            p_refined: (B, T, 3) refined prediction (on centerline)
            new_mem_state: Updated memory state
            attn_probs: (B, T, N) attention probabilities over centerline
        """
        B, T, _ = ant_pos.shape
        N, D = centerline_encoded.shape
        
        # Combine all ANT outputs: ant_trajectory + VO trajectory + visual features
        x = torch.cat([ant_pos, delta_pos, delta_quat, visual_tokens], dim=-1)  # (B, T, 3+3+4+D)
        x = self.input_proj(x)  # (B, T, memory_dim)
        
        # Update Titans memory with surprise signal
        # The memory learns to store trajectory history and detect "surprises"
        # (e.g., unexpected bifurcation choices)
        mem_out, new_mem_state, surprises = self.memory(x, state=mem_state, return_surprises=True)
        # surprises = (unweighted_mem_model_loss, adaptive_lr)
        # unweighted_mem_model_loss is the prediction error before memory update = surprise
        surprise_signal = surprises[0]  # (B, heads, T) or similar
        mem_out = self.norm1(mem_out)
        
        # Expand centerline for batch processing
        # CRITICAL: detach to prevent graph connection across windows during training
        cl_embed = centerline_encoded.detach().unsqueeze(0).expand(B, -1, -1)  # (B, N, memory_dim)
        
        # Cross-attention: memory queries centerline for global context
        # This allows the model to "look at the whole map" and understand global position
        attn_out, _ = self.cross_attn(
            query=mem_out,       # (B, T, memory_dim)
            key=cl_embed,        # (B, N, memory_dim)
            value=cl_embed
        )  # (B, T, memory_dim)
        
        attn_out = self.norm2(attn_out + mem_out)  # Residual connection
        attn_out = self.dropout(attn_out)  # Regularization
        
        # === ATTENTION-BASED CENTERLINE SELECTION ===
        # Instead of predicting delta, we compute attention scores over all centerline points
        # and take a weighted sum to select a point on the centerline.
        
        # Generate query vector for selection
        query = self.query_head(attn_out)  # (B, T, memory_dim)
        
        # Compute attention scores: dot product between query and centerline embeddings
        # query: (B, T, D), cl_embed: (B, N, D) -> scores: (B, T, N)
        scores = torch.bmm(query, cl_embed.transpose(1, 2)) * self.scale  # (B, T, N)
        
        # === SOFT DISTANCE PENALTY ===
        # Apply soft penalty based on distance from ANT's candidate-selected position
        # This encourages local consistency but allows branch corrections when needed
        cl_pts = centerline_points.detach().unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)
        
        # Compute L2 distance from each frame's ANT prediction to each centerline point
        dists = torch.cdist(ant_pos, cl_pts)  # (B, T, N)
        
        # Apply soft penalty: far points get lower scores but aren't masked out
        # Strong attention can overcome this penalty (e.g., for branch correction)
        scores = scores - self.distance_penalty_strength * dists
        
        # Softmax to get probabilities
        attn_probs = F.softmax(scores, dim=-1)  # (B, T, N)
        
        # Weighted sum: select refined position from full centerline
        p_refined = torch.bmm(attn_probs, cl_pts)  # (B, T, 3)
        
        # Compute average surprise for this window (scalar per batch)
        # surprise_signal shape: (B, heads, T) - average over heads and time
        if surprise_signal is not None and surprise_signal.numel() > 0:
            avg_surprise = surprise_signal.mean()  # Scalar: average surprise for this window
        else:
            avg_surprise = torch.tensor(0.0, device=p_refined.device)
        
        return p_refined, new_mem_state, attn_probs, avg_surprise
    
    def reset_memory(self):
        """
        Reset memory state for new trajectory.
        Call this at the start of each new bronchoscopy session.
        """
        return None  # Memory state is None initially

def create_bird(ant_mode='m', num_centerline_pts=1024, window_size=None):
    """
    Factory function to create a BIRD module with config matching ANT.
    
    Args:
        ant_mode: ANT model mode ('s', 'b', 'm', 'l')
        num_centerline_pts: Size of downsampled centerline
        window_size: Window size for chunk_size. If None, loads from window_config.
        
    Returns:
        BIRD module configured for the given ANT mode
    """
    config = BIRD_CONFIGS[ant_mode]
    
    # Get window_size from config file if not provided
    if window_size is None:
        window_size, _ = load_window_config()
    
    return BIRD(
        ant_mode=ant_mode,
        memory_dim=config['memory_dim'],
        chunk_size=window_size,
        num_heads=config['num_heads'],
        num_centerline_pts=num_centerline_pts
    )


