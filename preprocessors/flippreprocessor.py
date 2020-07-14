# import the necessary packages
from cv2 import cv2
import numpy as np


class FlipPreprocessor:
    """
    Randomly flips the input image horizontally.

    Attributes
    ----------
    prob: float
        flipping probability.
        eg: if prob = 0.5, input image will be flipped 50% of the time
    """

    def __init__(self, prob):
        # initialize the instance variables
        self.prob = prob

    def preprocess(self, img):
        # draw a random variable from a uniform distribution
        p = np.random.uniform(size = (1,))

        # check if the image is to be flipped
        if p < self.prob:
            img = cv2.flip(img, 1)

        # return the processed image
        return img
