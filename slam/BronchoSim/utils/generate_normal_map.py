import cv2
import numpy as np
import os

def generate_normal_map(input_path, strength=5.0):
    # Read image in grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image {input_path}")
        return

    # Invert if needed (mucosa veins might be darker or lighter, usually darker)
    # We want bumps.
    img = img.astype(np.float32) / 255.0

    # Calculate gradients
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # Construct normal map
    # N = normalize(-strength * gx, -strength * gy, 1)
    rows, cols = img.shape
    normal_map = np.zeros((rows, cols, 3), dtype=np.float32)
    
    # Z component is 1.0 (pointing up)
    # X and Y are gradients
    normal_map[..., 0] = -gx * strength
    normal_map[..., 1] = -gy * strength
    normal_map[..., 2] = 1.0

    # Normalize
    norm = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
    normal_map /= norm

    # Map to [0, 255]
    # [-1, 1] -> [0, 1] -> [0, 255]
    normal_map = ((normal_map + 1.0) / 2.0 * 255.0).astype(np.uint8)

    # Save (OpenCV uses BGR, so swap R and B for standard normal map RGB)
    # Standard Normal Map: R=X, G=Y, B=Z
    # OpenCV: B=Z, G=Y, R=X? No, OpenCV is BGR.
    # So we want output file to be RGB.
    # normal_map[..., 0] is X (Red)
    # normal_map[..., 1] is Y (Green)
    # normal_map[..., 2] is Z (Blue)
    # So in BGR memory: [Z, Y, X]
    output_bgr = np.zeros_like(normal_map)
    output_bgr[..., 0] = normal_map[..., 2] # B = Z
    output_bgr[..., 1] = normal_map[..., 1] # G = Y
    output_bgr[..., 2] = normal_map[..., 0] # R = X

    output_path = input_path.replace(".png", "_normal.png")
    cv2.imwrite(output_path, output_bgr)
    print(f"Normal map saved to {output_path}")

if __name__ == "__main__":
    texture_path = "data/textures/mucosa_diffuse.png"
    generate_normal_map(texture_path, strength=8.0)
