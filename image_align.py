import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from align import AlignDlib


alignment = AlignDlib('models/landmarks.dat')

def align_image(img, dimension):
    return alignment.align(dimension, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)