import numpy as np
import sys
import cv2

#############################################################
# This file has no purpose towards the goal of the project. #
# Just a bunch of filter experiments.                       #
#############################################################


def see_filter(ffilter):
    capture = cv2.VideoCapture(0)
    try:
        while True:
            # TODO: ADD MODEL
            ret, frame = capture.read()

            filtered = ffilter(frame)
            cv2.imshow('frame', filtered)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        pass

    capture.release()
    cv2.destroyAllWindows()


def laplacian(frame):
    return cv2.Laplacian(frame,cv2.CV_64F)

def gaussian(frame):
    return cv2.GaussianBlur(frame,(5,5),0)

def sobelX(frame):
    pass

