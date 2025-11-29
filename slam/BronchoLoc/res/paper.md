# BronchoLoc: A Deep Multi-Hypothesis Particle Filter for Bronchoscopic Localization

**Abstract**
This document presents a comprehensive technical definition of **BronchoLoc**, a deep learning-based system for localizing a bronchoscope within the human airway tree using only monocular video and a pre-operative CT map. Unlike traditional Simultaneous Localization and Mapping (SLAM) or regression-based approaches, BronchoLoc formulates localization as a **Multi-Hypothesis Tracking** problem solved by a unified, differentiable neural network. The system integrates a Spatio-Temporal Vision Transformer (ST-ViViT) for visual feature extraction, a Graph Attention Network (GAT) for topological map encoding, and a recurrent Long Short-Term Memory (LSTM) network for kinematic prediction. Crucially, it introduces a **Particle-Conditioned Fusion** mechanism and an in-network **Sequential Importance Resampling (SIR)** layer, allowing the model to maintain and evolve a multi-modal posterior distribution of the camera's pose. This document details the mathematical formulation of every component, including linear algebra operations, tensor transformations, and loss function derivations.

---

## 1. Introduction

### 1.1 Problem Formulation
Let the state of the bronchoscope at time $t$ be defined as $\mathbf{s}_t = [\mathbf{p}_t, \mathbf{v}_t]$, where $\mathbf{p}_t \in \mathbb{R}^3$ is the 3D position in the CT coordinate system and $\mathbf{v}_t \in \mathbb{R}^3$ is the velocity. The orientation is implicitly handled by the visual encoder's view-dependency but explicitly tracking position is the primary objective.

The system receives two inputs:
1.  **Visual Observation**: A video clip $\mathcal{V}_t = \{I_{t-T+1}, \dots, I_t\}$, where $I \in \mathbb{R}^{H \times W \times C}$ is a video frame.
2.  **Map Prior**: A topological graph $\mathcal{G} = (\mathcal{N}, \mathcal{E})$, where $\mathcal{N} = \{\mathbf{n}_1, \dots, \mathbf{n}_M\}$ are the 3D coordinates of airway centerlines, and $\mathcal{E}$ represents connectivity.

The goal is to estimate the posterior distribution $P(\mathbf{s}_t | \mathcal{V}_{1:t}, \mathcal{G})$. Due to the visual self-similarity of the bronchial tree (visual aliasing), this posterior is often multi-modal (non-Gaussian). Therefore, we approximate it using a set of $K$ weighted particles:
$$ P(\mathbf{s}_t | \mathcal{V}_{1:t}, \mathcal{G}) \approx \sum_{k=1}^K w_t^{(k)} \delta(\mathbf{s}_t - \mathbf{s}_t^{(k)}) $$

### 1.2 System Architecture Overview
The BronchoLoc architecture $\Phi$ is a recurrent function that updates the particle set:
$$ \{\mathbf{s}_t^{(k)}\}_{k=1}^K = \Phi(\mathcal{V}_t, \mathcal{G}, \{\mathbf{s}_{t-1}^{(k)}\}_{k=1}^K) $$

The function $\Phi$ is composed of four differentiable sub-modules:
1.  **Visual Encoder $E_V$**: $\mathbb{R}^{T \times H \times W \times C} \rightarrow \mathbb{R}^{T \times D}$
2.  **Map Encoder $E_M$**: $\mathcal{G} \rightarrow \mathbb{R}^{M \times D}$
3.  **Particle-Conditioned Fusion $F$**: Combines visual and map features relative to particle state.
4.  **Kinematic Updater $U$**: Updates particle states via LSTM.

---

## 2. Mathematical Formulation of Components

### 2.1 Visual Encoder: Spatio-Temporal ViViT
We utilize a Factorized Encoder variant of the Video Vision Transformer (ViViT).

#### 2.1.1 Patch Partitioning and Linear Embedding
The input video tensor $\mathbf{X} \in \mathbb{R}^{T \times H \times W \times C}$ is tokenized. We extract non-overlapping patches of size $P \times P$.
Let $N_H = H/P$ and $N_W = W/P$. The number of spatial tokens per frame is $N_S = N_H \times N_W$.
Each patch $\mathbf{x}_{t,i} \in \mathbb{R}^{P^2 C}$ is flattened and projected via a learnable linear matrix $\mathbf{W}_E \in \mathbb{R}^{(P^2 C) \times D}$:
$$ \mathbf{z}_{t,i}^{(0)} = \mathbf{W}_E \mathbf{x}_{t,i} + \mathbf{e}_{pos, i} + \mathbf{e}_{temp, t} $$
where $\mathbf{e}_{pos} \in \mathbb{R}^{N_S \times D}$ is the spatial positional embedding and $\mathbf{e}_{temp} \in \mathbb{R}^{T \times D}$ is the temporal embedding.

#### 2.1.2 Spatial Transformer Block
For each time step $t$, we process the spatial tokens $\mathbf{Z}_t \in \mathbb{R}^{N_S \times D}$ through $L_S$ transformer layers.
A single Multi-Head Self-Attention (MSA) layer is defined as follows. Let input be $\mathbf{X} \in \mathbb{R}^{N \times D}$. We compute Query, Key, and Value matrices:
$$ \mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V $$
where $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times D_{head}}$.
The attention weights are:
$$ \mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{D_{head}}}\right) \in \mathbb{R}^{N \times N} $$
The output of the head is $\mathbf{A}\mathbf{V}$. Multi-head attention concatenates $h$ heads and projects with $\mathbf{W}_O$.
$$ \text{MSA}(\mathbf{X}) = [\text{head}_1, \dots, \text{head}_h]\mathbf{W}_O $$
The Transformer block includes LayerNorm (LN) and a Multi-Layer Perceptron (MLP):
$$ \mathbf{Z}' = \text{MSA}(\text{LN}(\mathbf{Z})) + \mathbf{Z} $$
$$ \mathbf{Z}_{out} = \text{MLP}(\text{LN}(\mathbf{Z}')) + \mathbf{Z}' $$
We apply global average pooling across the spatial dimension to obtain a frame embedding:
$$ \mathbf{y}_t = \frac{1}{N_S} \sum_{i=1}^{N_S} \mathbf{z}_{t,i}^{(L_S)} $$
Resulting in a sequence of frame tokens $\mathbf{Y} \in \mathbb{R}^{T \times D}$.

#### 2.1.3 Temporal Transformer Block
To model motion dynamics, we apply $L_T$ transformer layers over the temporal sequence $\mathbf{Y}$.
$$ \mathbf{Z}_{vis} = \text{Transformer}_{Temp}(\mathbf{Y}) \in \mathbb{R}^{T \times D} $$
This tensor $\mathbf{Z}_{vis}$ encodes the visual features of the bronchoscope's motion over the window $T$.

### 2.2 Map Encoder: Graph Attention Network (GAT)
The map is a graph where nodes $\mathbf{n}_i \in \mathbb{R}^3$ are 3D coordinates.
We initialize node features $\mathbf{h}_i^{(0)}$ using a linear projection of their normalized 3D coordinates (and optionally node type, e.g., degree):
$$ \mathbf{h}_i^{(0)} = \mathbf{W}_{node} \mathbf{n}_i $$

We employ a Graph Attention Layer. For a node $i$ and its neighbors $\mathcal{N}(i)$:
$$ e_{ij} = \text{LeakyReLU}\left( \mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j] \right) $$
where $\mathbf{W} \in \mathbb{R}^{D \times D}$ is a weight matrix, $\mathbf{a} \in \mathbb{R}^{2D}$ is the attention vector, and $\|$ denotes concatenation.
The attention coefficients are normalized:
$$ \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})} $$
The updated node feature is:
$$ \mathbf{h}_i' = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}\mathbf{h}_j \right) $$
After $L_G$ layers, we obtain the map embeddings $\mathbf{Z}_{map} \in \mathbb{R}^{M \times D}$.

### 2.3 Particle-Conditioned Fusion
This module is critical for Multi-Hypothesis Tracking. It answers: *"Given particle $k$ is at $\mathbf{p}_k$, does the map at that location match the video?"*

#### 2.3.1 Relative Map Encoding
We cannot simply attend to the global map $\mathbf{Z}_{map}$ because the visual appearance depends on the *relative* position.
For each particle $k \in \{1 \dots K\}$ and each map node $m \in \{1 \dots M\}$:
$$ \Delta \mathbf{p}_{k,m} = \mathbf{n}_m - \mathbf{p}_k^{(t-1)} $$
We project this relative vector using a Multi-Layer Perceptron (MLP):
$$ \mathbf{e}_{rel}^{(k,m)} = \text{MLP}_{rel}(\Delta \mathbf{p}_{k,m}) \in \mathbb{R}^D $$
We augment the map features for particle $k$:
$$ \mathbf{H}_{map}^{(k)} = \mathbf{Z}_{map} + \mathbf{e}_{rel}^{(k)} \in \mathbb{R}^{M \times D} $$
*Note: In implementation, this is done efficiently via broadcasting without explicit loop expansion where possible, or by selecting only the $K_{NN}$ nearest nodes to reduce $M$.*

#### 2.3.2 Cross-Attention Fusion
We fuse the visual features $\mathbf{Z}_{vis}$ (Query) with the particle-specific map features $\mathbf{H}_{map}^{(k)}$ (Key/Value).
For particle $k$:
$$ \mathbf{Q}_k = \mathbf{Z}_{vis}\mathbf{W}_Q^F $$
$$ \mathbf{K}_k = \mathbf{H}_{map}^{(k)}\mathbf{W}_K^F $$
$$ \mathbf{V}_k = \mathbf{H}_{map}^{(k)}\mathbf{W}_V^F $$
$$ \text{Context}_k = \text{softmax}\left(\frac{\mathbf{Q}_k \mathbf{K}_k^\top}{\sqrt{D}}\right) \mathbf{V}_k \in \mathbb{R}^{T \times D} $$
This context vector encodes the map information relevant to the particle's current view.

### 2.4 Kinematic Updater: LSTM
We use an LSTM to predict the trajectory updates. The input to the LSTM at time step $\tau$ (within the window $T$) for particle $k$ is a concatenation of the visual token, the context, and the particle's own state embedding.

$$ \mathbf{x}_{lstm}^{(k, \tau)} = [\mathbf{z}_{vis}^{(\tau)} \| \text{Context}_k^{(\tau)} \| \text{MLP}_{state}(\mathbf{p}_k)] $$

The LSTM update equations are:
$$ \begin{aligned}
\mathbf{f}_\tau &= \sigma(\mathbf{W}_f \mathbf{x}_\tau + \mathbf{U}_f \mathbf{h}_{\tau-1} + \mathbf{b}_f) \\
\mathbf{i}_\tau &= \sigma(\mathbf{W}_i \mathbf{x}_\tau + \mathbf{U}_i \mathbf{h}_{\tau-1} + \mathbf{b}_i) \\
\mathbf{o}_\tau &= \sigma(\mathbf{W}_o \mathbf{x}_\tau + \mathbf{U}_o \mathbf{h}_{\tau-1} + \mathbf{b}_o) \\
\tilde{\mathbf{c}}_\tau &= \tanh(\mathbf{W}_c \mathbf{x}_\tau + \mathbf{U}_c \mathbf{h}_{\tau-1} + \mathbf{b}_c) \\
\mathbf{c}_\tau &= \mathbf{f}_\tau \odot \mathbf{c}_{\tau-1} + \mathbf{i}_\tau \odot \tilde{\mathbf{c}}_\tau \\
\mathbf{h}_\tau &= \mathbf{o}_\tau \odot \tanh(\mathbf{c}_\tau)
\end{aligned} $$

The hidden state $\mathbf{h}_\tau$ is projected to predict velocity and noise scale:
$$ \hat{\mathbf{v}}_\tau^{(k)} = \mathbf{W}_{vel} \mathbf{h}_\tau $$
$$ \boldsymbol{\sigma}_\tau^{(k)} = \text{softplus}(\mathbf{W}_{noise} \mathbf{h}_\tau) $$

The particle position is updated via Euler integration:
$$ \mathbf{p}_\tau^{(k)} = \mathbf{p}_{\tau-1}^{(k)} + \hat{\mathbf{v}}_\tau^{(k)} \cdot \Delta t + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{\sigma}_\tau^{(k)}) $$

---

## 3. Sequential Importance Resampling (SIR)

To function as a particle filter, the network must redistribute computational resources from low-probability hypotheses to high-probability ones. We implement a differentiable (or hard, during inference) resampling step.

### 3.1 Weight Calculation
We define the "likelihood" of a particle based on the alignment between its visual observation and the map at its predicted location.
For particle $k$, let $n^*$ be the nearest map node to $\mathbf{p}_k$.
The unnormalized log-weight is the dot product of the visual embedding and the map node embedding:
$$ s_k = \mathbf{z}_{vis}^{(T)} \cdot \mathbf{z}_{map}^{(n^*)} $$
The normalized weights are:
$$ w_k = \frac{\exp(s_k / \tau)}{\sum_{j=1}^K \exp(s_j / \tau)} $$
where $\tau$ is a temperature parameter.

### 3.2 Resampling Algorithm
We use **Multinomial Resampling**.
1.  Compute Cumulative Distribution Function (CDF): $C_k = \sum_{j=1}^k w_j$.
2.  Generate $K$ uniform random numbers $u_i \sim \mathcal{U}[0, 1]$.
3.  For each $u_i$, find index $j$ such that $C_{j-1} < u_i \le C_j$.
4.  Copy particle state: $\mathbf{s}_{new}^{(i)} \leftarrow \mathbf{s}_{old}^{(j)}$.
5.  Reset weights: $w_i \leftarrow 1/K$.

This step is non-differentiable. During training, we bypass explicit resampling and rely on the loss function to shape the gradients (Soft Resampling or just weighting the loss). During inference, we apply Hard Resampling every $N_{resample}$ frames.

---

## 4. Training Objective

The network is trained end-to-end using a composite loss function $\mathcal{L}$.

### 4.1 Best-of-K Trajectory Loss ($\mathcal{L}_{pose}$)
Since the posterior is multi-modal, penalizing the mean of particles against the ground truth is incorrect. We penalize the *best* particle.
Let $\mathbf{P}_{gt} \in \mathbb{R}^{T \times 3}$ be the ground truth trajectory and $\mathbf{P}_{pred}^{(k)} \in \mathbb{R}^{T \times 3}$ be the trajectory of particle $k$.
$$ \mathcal{L}_{pose} = \min_{k} \left( \frac{1}{T} \sum_{t=1}^T \| \mathbf{p}_{gt, t} - \mathbf{p}_{pred, t}^{(k)} \|_2^2 \right) $$
This encourages at least one mode of the distribution to track the true path.

### 4.2 Dense Retrieval Loss ($\mathcal{L}_{ret}$)
To enforce that the visual encoder learns semantically meaningful features aligned with the map, we use a frame-wise InfoNCE loss.
For frame $t$, let the ground truth position be $\mathbf{p}_{gt, t}$. The positive key is the map node $n^+$ closest to $\mathbf{p}_{gt, t}$. All other map nodes $n^-$ are negatives.
$$ \mathcal{L}_{ret} = - \sum_{t=1}^T \log \frac{\exp(\mathbf{z}_{vis, t} \cdot \mathbf{z}_{map, n^+} / \tau_{nce})}{\sum_{m=1}^M \exp(\mathbf{z}_{vis, t} \cdot \mathbf{z}_{map, m} / \tau_{nce})} $$
This creates a "Visual GPS" effect, ensuring that if the camera sees a specific bifurcation, the embedding matches that bifurcation's map node.

### 4.3 Geometry Violation Loss ($\mathcal{L}_{geo}$)
We utilize a pre-computed Signed Distance Field (SDF) $\Psi: \mathbb{R}^3 \rightarrow \mathbb{R}$.
$\Psi(\mathbf{x}) < 0$ inside airways, $\Psi(\mathbf{x}) > 0$ in tissue.
$$ \mathcal{L}_{geo} = \frac{1}{K \cdot T} \sum_{k=1}^K \sum_{t=1}^T \text{ReLU}(\Psi(\mathbf{p}_{pred, t}^{(k)})) $$
This penalizes particles that drift through airway walls. Since $\Psi$ is interpolated tri-linearly, it is differentiable.

### 4.4 Smoothness Loss ($\mathcal{L}_{smooth}$)
To enforce physical plausibility:
$$ \mathcal{L}_{smooth} = \frac{1}{K \cdot T} \sum_{k=1}^K \sum_{t=1}^T \| \mathbf{a}_{t}^{(k)} \|_2^2 = \sum \| (\mathbf{p}_{t} - 2\mathbf{p}_{t-1} + \mathbf{p}_{t-2}) \|^2 $$

### 4.5 Total Loss
$$ \mathcal{L}_{total} = \mathcal{L}_{pose} + \lambda_{geo}\mathcal{L}_{geo} + \lambda_{smooth}\mathcal{L}_{smooth} + \lambda_{ret}\mathcal{L}_{ret} $$

---

## 5. Implementation Details

### 5.1 Data Preprocessing
-   **Video**: Frames are resized to $128 \times 128$. Pixel values normalized to $[-1, 1]$.
-   **Trajectories**: Centerline coordinates are normalized to the unit cube $[-1, 1]^3$ using dataset-wide statistics:
    $$ \mathbf{p}_{norm} = \frac{\mathbf{p}_{raw} - \mathbf{c}_{dataset}}{s_{dataset}} $$
    where $\mathbf{c}_{dataset}$ is the global centroid and $s_{dataset}$ is the scaling factor.

### 5.2 Hyperparameters
-   **Particles ($K$)**: 32
-   **Sequence Length ($T$)**: 16 frames
-   **Embedding Dimension ($D$)**: 256
-   **ViViT Layers**: $L_S=4, L_T=2$
-   **GAT Layers**: $L_G=3$
-   **LSTM Hidden Size**: 512
-   **Optimizer**: AdamW, $\text{lr}=1e-4$, $\beta=(0.9, 0.999)$
-   **Batch Size**: 8 (Effective particles = $8 \times 32 = 256$)

### 5.3 Curriculum Learning
To avoid local minima (e.g., particles staying still to minimize $\mathcal{L}_{geo}$), we employ a curriculum:
-   **Phase 1 (Epochs 0-5)**: $\lambda_{geo} = 0$. Model learns visual-map correspondence.
-   **Phase 2 (Epochs 5+)**: $\lambda_{geo}$ linearly ramps from $0 \rightarrow 10.0$.

## 6. Conclusion
BronchoLoc provides a mathematically rigorous framework for deep probabilistic localization. By explicitly modeling the multi-modal nature of the posterior via particle filtering and conditioning deep feature fusion on particle states, it overcomes the limitations of single-hypothesis trackers. The integration of dense retrieval loss ensures strong supervisory signals even in the absence of distinct visual landmarks, making it a robust solution for bronchoscopic navigation.
