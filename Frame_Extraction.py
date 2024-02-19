import cv2 as cv
import numpy as np

# Extract images frames from Video
def extractFrames(pathIn, pathOut):
    count = 0
    vidcap = cv.VideoCapture(pathIn)
    success, image = vidcap.read()
    while success:
        # Add an alpha channel to the frame
        image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
        print('Read a new frame: ', success)
        
        # If pixel of frame is black make it transparent
        # Define a threshold below which the pixel is considered 'black'
        threshold = 10  # You can adjust this threshold as needed
        black_indices = np.all(image[:, :, :3] < threshold, axis=2)
        image[black_indices] = [0, 0, 0, 0]  # Set alpha to 0 (transparent)

        cv.imwrite(pathOut + "/frame%d.png" % count, image)  # save frame as PNG file
        count = count + 1

        # Read the next frame
        success, image = vidcap.read()

#Change the names according to the video file name and folder to save the extracted frames
extractFrames("balloons.mov", "balloons_frames")
