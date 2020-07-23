import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from align import AlignDlib


alignment = AlignDlib('models/landmarks.dat')

def align_image(img, bounding_box ,dimension):
    return alignment.align(dimension, img, bounding_box,
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)