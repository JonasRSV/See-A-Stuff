import tensorflow as tf
import numpy as np
import model
import os
import sys
import cv2
import train
import filters
from random import shuffle


def see_people():

    capture = cv2.VideoCapture(0)
    try:
        while True:
            # TODO: ADD MODEL
            ret, frame = capture.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        pass

    capture.release()
    cv2.destroyAllWindows()


def see_sobel():
    capture = cv2.VideoCapture(0)
    try:
        while True:
            # TODO: ADD MODEL
            ret, frame = capture.read()

            laplacian = cv2.Laplacian(frame,cv2.CV_64F)
            cv2.imshow('frame', laplacian)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        pass

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    if "-train" in sys.argv:
        train.train_model()

    if "-record" in sys.argv:
        train.record_training_data()

    if "-see" in sys.argv:
        see_people()

    if "-f_lap" in sys.argv:
        filters.see_filter(filters.laplacian)

    if "-f_gau" in sys.argv:
        filters.see_filter(filters.gaussian)












