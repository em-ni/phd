"""
Diagnose attention sharpness: Is the model learning one-hot attention or staying soft?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from ant import ActionPredictor
from ant_dataset import AntDataset
from torch.utils.data import DataLoader

def analyze_attention(model, dataloader, device):
    """Analyze the attention distribution of a trained model."""
    model.eval()
    
    all_entropies = []
    all_max_probs = []
    all_correct_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            map_points = batch['map_points'].to(device)
            map_mask = batch['map_mask'].to(device)
            targets = batch['actions'][:, :, :3].to(device)  # (B, T, 3)
            
            # Get model internals - need to modify forward pass
            B, T = video.shape[0], video.shape[1]
            
            # Extract features
            visual_tokens = model.visual_encoder(video)  # (B, T, D)
            queries = model.query_proj(visual_tokens)
            
            map_features = model.map_encoder(map_points)
            keys = model.key_proj(map_features)
            
            # Compute attention scores
            queries_exp = queries.unsqueeze(2)  # (B, T, 1, D)
            scores = torch.sum(queries_exp * keys, dim=-1) * model.scale  # (B, T, K)
            scores = scores.masked_fill(~map_mask, float('-inf'))
            probs = torch.softmax(scores, dim=-1)  # (B, T, K)
            probs = torch.nan_to_num(probs, nan=0.0)
            
            # Compute entropy of attention (lower = sharper)
            # H = -sum(p * log(p))
            log_probs = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs, dim=-1)  # (B, T)
            all_entropies.extend(entropy.cpu().numpy().flatten())
            
            # Max probability (higher = sharper)
            max_prob = probs.max(dim=-1)[0]  # (B, T)
            all_max_probs.extend(max_prob.cpu().numpy().flatten())
            
            # Find which point is the target and check its probability
            for b in range(B):
                for t in range(T):
                    target = targets[b, t]  # (3,)
                    pts = map_points[b, t]  # (K, 3)
                    mask = map_mask[b, t]   # (K,)
                    
                    # Find which point matches target
                    dists = torch.norm(pts - target, dim=1)
                    dists[~mask] = float('inf')
                    correct_idx = dists.argmin()
                    correct_prob = probs[b, t, correct_idx].item()
                    all_correct_probs.append(correct_prob)
    
    return {
        'mean_entropy': np.mean(all_entropies),
        'mean_max_prob': np.mean(all_max_probs),
        'mean_correct_prob': np.mean(all_correct_probs),
        'entropy_std': np.std(all_entropies),
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./dataset')
    parser.add_argument('--model_mode', type=str, default='s')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = AntDataset(os.path.join(args.data_root, "sequences"), mode='test')
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Load model
    model = ActionPredictor(window_size=dataset.window_size, mode=args.model_mode).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Analyze
    stats = analyze_attention(model, loader, device)
    
    print("\n" + "="*50)
    print("ATTENTION ANALYSIS")
    print("="*50)
    print(f"Mean Entropy:      {stats['mean_entropy']:.4f}")
    print(f"  (Random K=16 would be: {np.log(16):.4f})")
    print(f"  (Perfect one-hot:      0.0000)")
    print(f"Mean Max Prob:     {stats['mean_max_prob']:.4f}")
    print(f"  (Random: 0.0625, Perfect: 1.0)")
    print(f"Mean Correct Prob: {stats['mean_correct_prob']:.4f}")
    print(f"  (This is the probability on the CORRECT point)")
    print("="*50)
