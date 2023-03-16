import cv2 as cv

""" img = cv.imread('Resources/Photos/cat_large2.jpg')
cv.imshow('Cat', img) """

def rescaleFrame(frame, scale=0.75):
    # Works for Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    # Works for Live Video only
    capture.set(3, width)
    capture.set(4, height)

# Read video
capture = cv.VideoCapture('Resources/Videos/dog.mp4')

# Read video frame by frame
while True:
    isTrue, frame = capture.read()
    
    frame_resized = rescaleFrame(frame, scale=0.2)
    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    # Wait for any key to be pressed
    if cv.waitKey(1) & 0xFF==ord('d'): # if the key pressed is 'd'
        break

# Release video
capture.release()
cv.destroyAllWindows()