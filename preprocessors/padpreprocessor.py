# import the necessary packages
import tensorflow as tf


class PadPreprocessor:
    """
    Zero pads the input image by the specified amount
    (Support only for padding both height and width by the same amount)

    Attributes
    ----------
    pad: int
        amount of padding to add to both height and width of the image
    """

    def __init__(self, pad):
        # initialize the instance variables
        self.pad = pad

    def preprocess(self, img):
        # return the padded image
        return tf.pad(img, [[self.pad, self.pad], [self.pad, self.pad], [0, 0]], mode="CONSTANT")
