# timing and complexity experiments
difference between mono_realtime and mono_video
ExtractOrb: mono_realtime ~ mono_video
Tracking: mono_realtime ~ 5-10 x mono_video

given that I kept the settings the same, the problem is the streaming speed

Average duration of one step of SLAM.TrackMonocular(resized_frame, tframe) by varying one parameter at a time:
nFeatures: 3000 -> 79.69 milliseconds
nFeatures: 2500 76.22 milliseconds and 48.83 milliseconds
nFeatures: 2000 -> 56.98 milliseconds

Orb extraction complexity (from gpt):
O(N⋅M⋅(1−S^(-2))/(1−S^(-2L)))

with:
N = nFeatures
M = width * height
S = scaleFactor
L = nLevels

from gpt:
To provide a more holistic and single O() notation that incorporates the key parameters of 
nFeatures, nLevels, scaleFactor, and the image resolution (described by width and height), 
we need to consider how each parameter influences the overall complexity of the ORB feature 
extraction process across multiple levels of image pyramids.

Parameters and their impact:
nFeatures (N): Total number of features to detect and compute across all levels, impacting 
complexity linearly for feature processing.

nLevels (L): The number of pyramid levels influences how many times the image is processed 
at decreasing resolutions. Each level has fewer pixels than the previous, reducing by a factor 
related to the scaleFactor.

scaleFactor (S): Determines the rate at which the resolution decreases per level. Each level 
is scaled down by this factor squared (for both width and height).

Resolution (Width × Height = M): Initial number of pixels to process.

Computing the Complexity:
Given these parameters:

At each level i, the image size is reduced exponentially based on the scaleFactor. The size 
at level i can be approximated as M⋅S^(-2i) (assuming a square reduction for simplification).
The cost of processing each level includes detecting features and computing their descriptors, 
which would scale with the number of features detected at that level and the size of the image 
at that level.

Approximate Formula:
The total number of pixels processed across all levels is a geometric series:

M + M⋅S^(-2) + M⋅S^(-4) + … + M⋅S^(-2(L−1))

Summing this up results in:

M⋅(1−S^(-2L))/(1−S^(-2))

Assuming that the distribution of features might not be even across levels, but for a rough 
estimate, we could assume that the features are distributed in some manner related to the 
number of pixels at each level. Let's simplify by assuming that features are processed uniformly 
across levels for a rough complexity estimate.

Combining into a Single O():
Given the parameters and their impact:

The complexity is primarily dominated by the number of features and the effective number of 
pixels processed across levels. Thus, a single O() notation that would approximately represent 
the complexity considering the exponential reduction of image size per level and the distribution 
of feature processing can be expressed as:

O(N⋅M⋅(1−S^(-2))/(1−S^(-2L)))

This notation tries to incorporate the interactions between the number of features, the levels 
of the pyramid, the scaling factor's impact on the size of images at each level, and the original 
resolution of the image. It captures the complexity driven by the initial resolution and the 
decrease in processing needed at each successive pyramid level, modified by the number of features 
N aimed to be processed overall.


# results on bronchoscopy dataset
in summary basically one of the video worked from start to end neither with lung.yaml nor new_lung.yaml
; bronchoscopy dataset
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/dynamic_seq_000.mp4                     
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/dynamic_seq_001.mp4                     
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/dynamic_seq_002.mp4                     
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/dynamic_seq_003.mp4                     
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/real_seq_000_part_0_dif_1.mp4           
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/real_seq_000_part_1_dif_1.mp4           
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/real_seq_001_part_0_dif_1.mp4           
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/real_seq_001_part_1_dif_1.mp4           
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/real_seq_001_part_2_dif_1.mp4           
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/real_seq_001_part_3_dif_1.mp4           
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/real_seq_001_part_4_dif_1.mp4           
; lung.yaml: X    new_lung.yaml: X
;       video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_000_part_0_dif_0.mp4         
; lung.yaml: V    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_000_part_1_dif_0.mp4         
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_000_part_2_dif_0.mp4         
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_001_part_0_dif_0.mp4         
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_001_part_1_dif_0.mp4         
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_001_part_2_dif_0.mp4         
; lung.yaml: X    new_lung.yaml: X
;       video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_001_part_3_dif_0.mp4         
; lung.yaml: V    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_001_part_4_dif_0.mp4         
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_002_part_0_dif_1.mp4         
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_002_part_1_dif_1.mp4         
; lung.yaml: V    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_002_part_2_dif_1.mp4         
; lung.yaml: X    new_lung.yaml: X
;       video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_003_part_0_dif_1.mp4         
; lung.yaml: V    new_lung.yaml: X
;       video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_003_part_1_dif_1.mp4         
; lung.yaml: V    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_004_part_0_dif_0.mp4         
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_004_part_1_dif_0.mp4         
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_004_part_2_dif_1.mp4         
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_004_part_3_dif_1.mp4         
; lung.yaml: X    new_lung.yaml: X
;       video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_005_part_0_dif_1.mp4         
; lung.yaml: V    new_lung.yaml: V
;       video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_005_part_1_dif_1.mp4         
; lung.yaml: V    new_lung.yaml: V
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_006_part_0_dif_1.mp4         
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_006_part_1_dif_1.mp4         
; lung.yaml: X    new_lung.yaml: X
; video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_006_part_2_dif_1.mp4         
; lung.yaml: X    new_lung.yaml: X
;       video = em/run/videos/datasets/bronchoscopy_dataset/dataset/stable_seq_006_part_3_dif_1.mp4         
; lung.yaml: V    new_lung.yaml: X
