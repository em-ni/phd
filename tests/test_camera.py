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

# get camera in linux
capture = cv.VideoCapture(0)

# Read video frame by frame
while True:
    isTrue, frame = capture.read()

    # Frame is to bright so we can make it darker
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Decrease the brightness of the image
    alpha = 0.5  # Value between 0 and 1, where 0 means black and 1 means white
    beta = 50   # Value to be added to each pixel after scaling
    frame2 = cv.convertScaleAbs(gray, alpha=alpha, beta=beta)

    cv.imshow('Video', frame)
    cv.imshow('Video2', frame2)

    # Wait for any key to be pressed
    if cv.waitKey(1) & 0xFF==ord('d'): # if the key pressed is 'd'
        break

# Release video
capture.release()
cv.destroyAllWindows()
