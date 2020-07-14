# import the necessary packages
import numpy as np


class MeanPreprocessor:
    """
    Processes the image by (optionally) normalizing the pixel values into [0, 1],
    subtracts the mean, and finally divides the values by the standard deviation

    Attributes
    ----------
    mean: float or list of floats
        either the overall dataset mean (int) or the per-channel mean (list of ints)
    std: float or list of floats
        either the overall dataset std (float) or the per-channel std (list of floats)
    normalize: bool
        determines if the input data is to be normalized to the range [0, 1]
    """

    def __init__(self, mean, std, normalize = True):
        # initialize the instance variables
        self.mean = mean
        self.std = std
        self.normalize = normalize

    def preprocess(self, img):
        # check if the image is to be normalized
        if self.normalize:
            img = img.astype(np.float32) / 255.0

        # return the processed image
        return (img - self.mean) / self.std
