# ANT + BIRD Pipeline

## 1. Overview
The system consists of two cascaded models:
1.  **ANT (Airways Neighborhood Tracker):** A local window-based model that predicts 6DoF pose changes (VO) and selects a candidate position on the centerline using visual-to-map attention.
2.  **BIRD (Bronchial Intraoperative Route Discriminator):** A global streaming model that uses Titans memory to refine ANT's predictions by attending to the *full* centerline, ensuring global consistency and smooth trajectory tracking.

---

## 2. Dataset & Preprocessing

### Inputs
- **Video:** Bronchoscopy frames $I_0, ..., I_N$.
- **Trajectory:** Ground Truth (GT) camera poses $P_0, ..., P_N$ (position $p$ + orientation $q$).
- **Centerline:** Dense point cloud of the airway tree $C = \{c_1, ..., c_K\}$.

### Window Generation
The continuous stream is sliced into overlapping windows of size $T$ (e.g., 10) with a frame skip $\delta$ (e.g., 40).
- **Video Tensor:** $(T, C, H, W)$ normalized to $[-1, 1]$.
- **Local Frame:** Defined by the pose of the *first frame* in the window $(p_0, q_0)$. All targets and inputs for the window are expressed relative to this frame.

### Candidates (Map Points)
Topological map candidates are selected for the window:
1.  **Query:** Ball search radius $R$ around $p_0$ (or predicted anchor).
2.  **Filter:** Connected component analysis ensures reachable points.
3.  **Sample:** Density-based sampling reduces candidates to $K$ points.
4.  **Transform:** Converted to Local Frame: $c_{local} = q_0^{-1} * (c_{global} - p_0)$.
5.  **Normalize:** Divided by scale factor $S$.
- **Result:** `map_points` $(T, K, 3)$.

### Targets
- **ANT Position (`gt_pos`):** Nearest centerline point to $p_t$ in Local Frame.
- **VO Position (`delta_pos`):** Frame-to-frame displacement: $R_0^{-1} * (p_t - p_{t-1})$.
- **VO Orientation (`delta_quat`):** Orientation relative to start: $q_0^{-1} * q_t$.

---

## 3. ANT Model (Local Estimator)

### Inputs
- **Video:** $(B, T, C, H, W)$
- **Map Points:** $(B, T, K, 3)$

### Architecture
```
┌───────────────────────────────────────────────────────────────┐
│              ANT (Airways Neighborhood Tracker)               │
├───────────────────────────────────────────────────────────────┤
│  Video (B, T, C, H, W)                                        │
│         ↓                                                     │
│  ┌─────────────┐                                              │
│  │   STViViT   │ → visual_tokens (B, T, D)                    │
│  └─────────────┘        ↓                                     │
│         │        ┌─────────────────┐                          │
│         └──────→ │     VO Head     │ → delta_pos, delta_quat  │
│                  └─────────────────┘        ↓                 │
│         ┌───────────────────────────────────┘                 │
│         ↓                                                     │
│  [visual_tokens + deltas] → Query (B, T, D)                   │
│                                           ↓                   │
│  Map Candidates (K, 3) → [Map Encoder] → Keys (B, T, K, D)    │
│                                           ↓                   │
│  ┌────────────────────────┐                                   │
│  │       Attention        │ ← Dot Product (Q @ K)             │
│  └────────────────────────┘                                   │
│         ↓                                                     │
│  probs (B, T, K)                                              │
│         ↓                                                     │
│  ant_pos = Σ (probs * map_candidates)                         │
└───────────────────────────────────────────────────────────────┘
```
1.  **Visual Encoder (STViViT):**
    -   Extracts Spatio-Temporal features from video.
    -   **Output:** `visual_tokens` $(B, T, D)$.

2.  **Visual Odometry (VO) Head:**
    -   Parallel MLPs predict motion dynamics from `visual_tokens`.
    -   **Position:** `pred_delta_pos` $(B, T, 3)$ - Step vector in Local Frame.
    -   **Orientation:** `pred_delta_quat` $(B, T, 4)$ - Orientation relative to Local Frame (normalized).

3.  **Map Encoder:**
    -   Point-wise MLP encodes geometric layout of candidates.
    -   **Output:** `map_features` $(B, T, K, D)$.

4.  **Attention Mechanism (Selection):**
    -   **Query Construction:** Concatenation of visual and VO features.
        $$Q = \text{Proj}([ \text{visual\_tokens}, \text{delta\_pos}, \text{delta\_quat} ])$$
        Shape: $(B, T, D)$.
    -   **Key/Value:** Project `map_features` to $K$ and $V$.
    -   **Attention Scores:** Dot product $QK^T$ scaled by $\frac{1}{\sqrt{D}}$.
    -   **Probabilities:** Softmax over $K$ candidates -> `probs` $(B, T, K)$.

5.  **Output Generation:**
    -   **Refined Position:** Weighted sum of map points: $p_{ant} = \sum (\text{probs} \cdot \text{map\_points})$.
    -   **Returns:** `ant_pos` (3), `delta_pos` (3), `delta_quat` (4), `visual_tokens` (D).

### Training Loss
$$L_{ANT} = L_{MSE}(p_{ant}, p_{gt}) + \lambda_{VO}(L_{MSE}(\delta p, \delta p_{gt}) + L_{MSE}(\delta q, \delta q_{gt})) + \lambda_{CE}(L_{CE}(\text{probs}, \text{idx}_{gt}))$$

---

## 4. BIRD Model (Global Refiner)

### Inputs
- **From ANT:** `ant_pos`, `delta_pos`, `delta_quat`, `visual_tokens`.
- **Global Context:** `centerline_points` $(N, 3)$ - The full airway tree (downsampled).

### Architecture
```
┌───────────────────────────────────────────────────────────────┐
│        BIRD (Bronchial Intraoperative Route Discriminator)    │
├───────────────────────────────────────────────────────────────┤
│  ANT Outputs: ant_pos, delta_pos/quat, visual_tokens          │
│         ↓                                                     │
│  ┌─────────────┐                                              │
│  │ Input Proj  │ → input_embed (B, T, Dm)                     │
│  └─────────────┘                                              │
│         ↓                                                     │
│  ┌─────────────┐      ┌───────────────┐                       │
│  │ Titans Mem  │ ───→ │   Mem State   │ (Streaming)           │
│  └─────────────┘      └───────┬───────┘                       │
│         ↓                     │                               │
│  mem_out (B, T, Dm)           ↺ Recursive                     │
│         ↓                                                     │
│  ┌─────────────┐                                              │
│  │ Cross Attn  │ ← Full Centerline (N, 3) → [Encoder] → Keys  │
│  └─────────────┘                                              │
│         ↓                                                     │
│  attn_out (B, T, Dm)                                          │
│         ↓                                                     │
│  Q_sel = QueryHead(attn_out)                                  │
│         ↓                                                     │
│  ┌────────────────────────┐                                   │
│  │ Centerline Selection   │ ← Scores = Q_sel @ Keys           │
│  │ (with Dist Penalty)    │                                   │
│  └────────────────────────┘                                   │
│         ↓                                                     │
│  p_refined = Σ (probs * centerline_pts)                       │
└───────────────────────────────────────────────────────────────┘
```
1.  **Input Projection:**
    -   Fuses all local signals into a single stream.
    -   $x = \text{Linear}([p_{ant}, \delta p, \delta q, \text{vis}])$ -> $(B, T, D_m)$.

2.  **Titans Neural Memory:**
    -   Recurrent memory module that maintains a persistent state `mem_state` across windows.
    -   Learns sequence history and "surprise" (e.g., branching decisions).
    -   **Output:** `mem_out` $(B, T, D_m)$.

3.  **Cross-Attention (Global Context):**
    -   Memory attends to the **Full Centerline** (encoded once).
    -   Allows the model to localize the current window within the global anatomy.
    -   **Query:** `mem_out`. **Key/Value:** `centerline_embeddings`.
    -   **Output:** `attn_out` $(B, T, D_m)$.

4.  **Refinement Head:**
    -   **Selection:** Computes attention scores over all $N$ centerline points.
    -   **Distance Penalty:** Soft $L_2$ penalty suppresses physically distant points (prevents teleportation) but allows correction of branch errors.
    -   **Normalization:** Softmax over $N$.
    -   **Weighted Sum:** Selects the optimal point on the global centerline.
    -   **Output:** `p_refined` $(B, T, 3)$.

### Training Strategy: Surprise-Based Window Selection

**Problem:** Ideally BIRD should be aware of the entire trajectory, but backpropagating through all windows per sequence causes memory explosion and gradient issues.

**Solution:** Train on only the **top-K most surprising windows** per sequence, where surprise is measured by Titans memory's internal prediction error:

1.  **Forward all windows** in sequence order → collect (loss, surprise) pairs
2.  **Rank by surprise** (descending) → select top-K
3.  **Backprop only on selected windows** → focus learning on informative moments

**Rationale:** High-surprise windows are typically:
- **Bifurcations** (unexpected branch choices)
- **Direction changes** (memory state transition points)
- **Difficult regions** where the model struggles

This focuses training on the most informative windows while maintaining memory streaming across the full sequence.

### Training Loss
$$L_{BIRD} = \frac{1}{K} \sum_{i \in \text{top-K}} L_{MSE}(p_{refined}^{(i)}, p_{gt}^{(i)})$$
- $K$ = `--top_k_surprise` (default: 4)
- Gradients are **not** backpropagated to ANT (ANT is frozen).

---

## 5. Inference Pipeline

### Closed-Loop Feedback
At inference, BIRD provides the anchor for the *next* window, creating a closed loop:
1.  **Window $t$:** ANT predicts → BIRD refines → $P_{refined}$.
2.  **Feedback:** The last point of $P_{refined}$ becomes the center for candidate selection (ball search) for Window $t+1$.
3.  **Streaming:** BIRD passes `mem_state` to the next window, maintaining continuous tracking history.

### Surprise: Training vs. Inference

| Aspect | Training | Inference |
|--------|----------|-----------|
| Window selection | Top-K surprising only | All windows, sequentially |
| Surprise usage | Explicit (gradient selection) | Implicit (baked into memory) |

**How training transfers to inference:**
-   Training focuses on high-surprise windows (bifurcations, direction changes).
-   Titans memory's internal weights learn to detect and respond to these patterns.
-   At inference, the memory **internally** uses surprise for state updates:
    -   High surprise → stronger memory writes
    -   Low surprise → gentle updates (already predicted)
-   Result: Memory naturally "remembers" bifurcations more than straight sections.

### Output Coordinate Reconstruction
Final global trajectory is reconstructed from Local Frame predictions:
$$P_{global} = P_{anchor} + R_{anchor} \cdot (P_{local} \times S)$$
- $S$: Normalization scale (`NORM_MAP_SCALE`).
- $P_{anchor}, R_{anchor}$: Pose of the window's first frame (or BIRD's feedback).