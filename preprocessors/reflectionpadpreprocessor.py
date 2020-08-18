# import the necessary packages
import tensorflow as tf


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
        # return the processed image
        return tf.pad(img, [[self.pad, self.pad], [self.pad, self.pad], [0, 0]], mode="REFLECT")
