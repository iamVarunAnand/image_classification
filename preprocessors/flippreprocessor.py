# import the necessary packages
import tensorflow as tf


class FlipPreprocessor:
    """
    Randomly flips the input image horizontally (with 50% chance).

    """

    def __init__(self):
        pass

    def preprocess(self, img):
        # return the processed image
        return tf.image.random_flip_left_right(img)
