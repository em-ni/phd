# Full ANT pipeline explained from first principles
## Dataset
The inputs needed by the model are:
- A video of the bronchoscopy
- A trajectory associated with the video
- The lung centerline

The video is a sequence of Nf frames.
Each frame is a 2D image (WxH) of the bronchoscopy.
Each pixel is a color value C = (R, G, B).

The trajectory is a sequence of Np points in 3D space.
Each point is a vector (x, y, z).

The lung centerline is a sequence of Nc points in 3D space.
Each point is a vector (x, y, z).

These must be transformed into a window, which is the model input:
From a given frame, sample a window of size window_size, with a gap of frame_skip between each frame, basically reducing the fps of the video.
Resize each video frame to img_size.
Normalize each pixel value to [-1, 1].
The result is the video tensor of shape (window_size, img_size, img_size, 3).

Each window is associated with a set of canidate centerline points.
These points are selected as follows:
From the position of the first frame (either the starting position at the lung entrance or predicted at previous step) of the window, select all the points in the centerline that are within a radius of MAP_QUERY_RADIUS.
Use DBSCAN filter to remove points from disconnected airways (not reachable without exiting the ball)
Downsample using farthest point sampling.
The result is a list of 3D coordinates and their associated index in the centerline (N, 3)

These have to be transformed into local frame (relative to the first frame of the window):
The first frame position is a (T, 7) vector (x0, y0, z0, qx0, qy0, qz0, qw0) where T is the timestamp.
Hence each candidate point is transformed relative to the first frame position, by subtracting the first frame position (x0,y0,z0) to its coordinate and with the inverse rotation extracted from the first frame rotation (qx0, qy0, qz0, qw0).
Normalize each point to [-1, 1] dividing by NORM_MAP_SCALE.
Points are padded to DEFAULT_MAX_MAP_POINTS.
A mask is created for valid points (i.e. non padded points)
The result is the map points tensor of shape (window_size, DEFAULT_MAX_MAP_POINTS, 3) and the mask of shape (window_size, DEFAULT_MAX_MAP_POINTS)

If the data is loaded for training, to each frame is associated a target action, otherwise that's what the model has to predict.
The action is a vector associated to each frame of the window, built as follows for each frame:
Take the ground truth position of the frame (x, y, z, qx, qy, qz, qw).
Find the closest point in the centerline to the frame GT position.
Transform the point into local frame (same as above)
Normalize each point to [-1, 1] dividing by NORM_MAP_SCALE.
(for now the orientation is not used)
The result is the action tensor of shape (window_size, 3)

## Model
The input of the model are the video tensor (window_size, img_size, img_size, 3) and the map points tensor (window_size, DEFAULT_MAX_MAP_POINTS, 3) and the mask of shape (window_size, DEFAULT_MAX_MAP_POINTS)
The output of the model is the action tensor of shape (window_size, 3)

The video tensor goes to a Spatio-Temporal Vision Transformer (STViViT in ViViT-style).
This output a single feature vector of dimension embed_dim per frame, so for a single window it outputs a tensor of shape (window_size, embed_dim). 
(considering the batch size B, the output is the visual tokens tensor of shape (B, window_size, embed_dim))
TODO: add details about the STViViT

The map points tensor goes to the Map Encoder, which is a MultiLayer Perceptron (MLP).
It is composed of a sequence of Linear layers interleaved with GELU activations.
The input dimension is 3 (the 3D coordinates).
The hidden layers have sizes [64, 128].
The final layer projects to output_dim.
The MLP is applied point-wise to each map point, meaning no pooling is applied as we want to attend to individual points.
This outputs a single feature vector of dimension output_dim per point, so for a single window it outputs a tensor of shape (window_size, output_dim). 
(considering the batch size B, the output is the map features tensor of shape (B, window_size, output_dim)).
output_dim is equal to embed_dim.

The visual tokens are projected to the Query space via a Linear layer, resulting in a tensor of shape (B, window_size, embed_dim).
The map features are projected to the Key space via a Linear layer, resulting in a tensor of shape (B, window_size, DEFAULT_MAX_MAP_POINTS, embed_dim).

The attention scores are computed as the dot product between the Queries and the Keys.
The scores are scaled by the inverse square root of embed_dim.
The mask is applied to set the scores of padded points to negative infinity.
The scores are normalized with a softmax layer to obtain probabilities of shape (B, window_size, DEFAULT_MAX_MAP_POINTS).

The final prediction is computed as the weighted sum of the original map points (3D coordinates), using the attention probabilities as weights.
This effectively selects a position within the convex hull of the candidate centerline points.
The result is the action tensor of shape (window_size, 3).

## Training
The model is trained using
- MSE loss between the predicted action and the target action i.e. for each batch the loss is computed between the action tensor and the target action tensor  of each frame of the window, for each window in the batch (B, window_size, 3).
- Cross-entropy loss on the attention weights, this is to make the model sharper in the prediction, avoiding smoothing the probability distribution and spreading it across all points.

## Inference
The model is used to predict the action for each frame of the window, for each window in the batch (B, window_size, 3). The very first frame of the video is initialized as the starting position of the centerline, then each subsequent first frame of subsequent windows is initialized as the predicted position from the previous window.
The global trajectory is then built by concatenating the predicted positions of each window.
This can lead to a trajectory that is not smooth and drifts away from the real trajectory, as the model is not aware of previous windows.
For this reason we need the BIRD model.

# Full BIRD pipeline explained from first principles
# Model
The input of the model are the output window of the ant