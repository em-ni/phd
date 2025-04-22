import cv2 as cv

""" # Read image and output it as matrix of pixels
img = cv.imread('Resources/Photos/cat.jpg')
#img = cv.imread('Resources/Photos/cat_large.jpg')

# Display image
cv.imshow('Cat', img)

# Wait for any key to be pressed
cv.waitKey(0) """

# Read video
# To use camera connected to computer, use 0 as argument (or 1,2,3...)
#capture = cv.VideoCapture('Resources/Videos/dog.mp4')
capture = cv.VideoCapture(0)

# Read video frame by frame
while True:
    isTrue, frame = capture.read()
    print(isTrue)
    print(frame)
    cv.imshow('Video', frame)

    # Wait for any key to be pressed
    if cv.waitKey(1) & 0xFF==ord('d'): # if the key pressed is 'd'
        break

# Release video
capture.release()
cv.destroyAllWindows()