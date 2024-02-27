import cv2
import numpy as np

def main(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define a very narrow color range for yellow
    lower_yellow = np.array([25, 170, 170])  # Very narrow range for yellow detection
    upper_yellow = np.array([32, 255, 240])
    
    # Define the color ranges for red
    lower_red1 = np.array([0, 50, 50])  # Low hue range for red
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])  # High hue range for red
    upper_red2 = np.array([180, 255, 255])
    
    # Define the color range for blue
    lower_blue = np.array([100, 50, 50])  # Blue hue range
    upper_blue = np.array([140, 255, 255])
    
    # Create masks for yellow, red, and blue colors
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Combine the red masks
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Bitwise-AND mask and original image to extract yellow, red, and blue colors
    yellow_detected = cv2.bitwise_and(image, image, mask=mask_yellow)
    red_detected = cv2.bitwise_and(image, image, mask=mask_red)
    blue_detected = cv2.bitwise_and(image, image, mask=mask_blue)
    
    # Show the original image, yellow detection, red detection, and blue detection side by side
    result_yellow = np.hstack((image, yellow_detected))
    result_red = np.hstack((image, red_detected))
    result_blue = np.hstack((image, blue_detected))
    
    # Display the resulting frames
    cv2.imshow('Yellow color detection', result_yellow)
    cv2.imshow('Red color detection', result_red)
    cv2.imshow('Blue color detection', result_blue)
    
    # Save the result images
    cv2.imwrite('yellow_detection_result.png', result_yellow)
    cv2.imwrite('red_detection_result.png', result_red)
    cv2.imwrite('blue_detection_result.png', result_blue)
    
    # Wait for the ESC key (27) to close the windows
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
