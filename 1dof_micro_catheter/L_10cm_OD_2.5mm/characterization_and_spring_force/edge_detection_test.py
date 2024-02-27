import cv2
import numpy as np

def main(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the color range for the green background
    lower_green = np.array([40, 40, 40])  # Lower bound for green
    upper_green = np.array([80, 255, 255])  # Upper bound for green
    
    # Create a mask to filter out the green background
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # Invert the mask to get the objects as white on a black background
    mask_objects = cv2.bitwise_not(mask_green)
    
    # Apply a median blur to reduce noise in the mask
    mask_objects_blurred = cv2.medianBlur(mask_objects, 5)
    
    # Detect edges using Canny on the masked objects
    edges = cv2.Canny(mask_objects_blurred, 100, 200)
    
    # Show the original image and the edge detection result side by side
    result = np.hstack((image, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
    
    # Display the resulting frame
    cv2.imshow('Edge detection using green background', result)
    
    # Save the result image
    cv2.imwrite('edge_detection_result.png', result)
    
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
