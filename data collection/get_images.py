import cv2 as cv
import uuid
import os
import time
import subprocess

# Define labels
labels = ['cavity', 'tumor']

# Define number of images to be collected
number_imgs = 15

# Setup folder to save images
IMAGES_PATH = os.path.join('data', 'images', 'collectedimages')

if not os.path.exists(IMAGES_PATH):
    if os.name == 'posix':
        subprocess.run(['mkdir', IMAGES_PATH], shell=True)
    if os.name == 'nt':
        subprocess.run(['mkdir', IMAGES_PATH], shell=True)
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        subprocess.run(['mkdir', path], shell=True)

# Capture images from camera
for label in labels:
    cap = cv.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for img_num in range(number_imgs):
        print('Collecting image {}'.format(img_num))
        ret, frame = cap.read()
        img_name = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv.imwrite(img_name, frame)
        cv.imshow('frame', frame)
        time.sleep(2)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv.destroyAllWindows()