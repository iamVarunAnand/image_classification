# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    """
    Converts the input image into an array format that is compatible with
    keras' requirements

    Attribute:
    ---------
    data_format: optional
        Image data format, can be either "channels_first" or "channels_last".
        Defaults to None, in which case the global setting tf.keras.backend.image_data_format() is used
    """

    def __init__(self, data_format = None):
        # initialize the instance variables
        self.data_format = data_format

    def preprocess(self, img):
        # return the processed image by applying the keras utility function
        # that correctly rearranges the dimensions of the image
        return img_to_array(img, data_format = self.data_format)
