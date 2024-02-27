import cv2
import numpy as np
from utils import *

def main(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    edges = process_frame_for_edges(image)
    
    # Show the original image and the edge detection result side by side
    result = np.hstack((image, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
    
    # Display the resulting frame
    cv2.imshow('Edge detection using green background', result)
    
    # Save the result image
    cv2.imwrite('tests\\edge_detection_result.png', result)
    
    # Wait for the ESC key (27) to close the window
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    
    # Destroy all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide an image path.")
