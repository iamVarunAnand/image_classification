# import the necessary packages
import numpy as np


class ReflectionPadPreprocessor:
    """
    Pads the images by the specified amount using the reflection of the outer edges. 
    Equivalent to using 'REFLECT' mode in tf.pad
    (Support only for padding height and width by the same amount)

    Attributes
    ----------
    pad: int
        amount of padding to add to both height and width of the image
    """

    def __init__(self, pad):
        # initialize the instance variables
        self.pad = pad

    def preprocess(self, img):
        # zero pad the image
        img = np.pad(img, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)))

        # reflect pad the image
        for i, j in zip(range(self.pad), range(self.pad)):
            xstart = self.pad
            xend = img.shape[1] - self.pad - 1
            ystart = self.pad
            yend = img.shape[0] - self.pad - 1

            img[:, xstart - i - 1] = img[:, xstart + i + 1]
            img[:, xend + i + 1] = img[:, xend - i - 1]
            img[ystart - j - 1, :] = img[ystart + j + 1, :]
            img[yend + j + 1, :] = img[yend - j - 1, :]

        # return the processed image
        return img
