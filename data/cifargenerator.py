# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
import numpy as np


class CifarGenerator:
    """
    Generator class responsible for supplying data to the model.

    Attributes
    ----------
    x: np ndarray
        array of images
    y: np ndarray
        array of class labels
    batch_size: int
        batch size to be used while generating batches
    preprocessors: (optional) list
        all the preprocessors to be applied to the input data. defaults to None
    aug: (optional) tf.keras.preprocessing.image.ImageDataGenerator object
        data augmentation to be applied to the input data. defaults to None
    """

    def __init__(self, x, y):
        # initialize the data
        self.x = x
        self.y = y

        # initialize the instance variables
        self.num_images = self.x.shape[0]

        # convert the labels from integers into vectors
        self.lb = LabelBinarizer()
        self.y = self.lb.fit_transform(self.y)

    def generator(self, passes=np.inf):
        # initialize a variable to keep a count on the epochs
        epochs = 0

        # loop through the dataset indefinitely
        while(epochs < passes):
            # loop through the dataset
            for idx in range(0, self.num_images):
                # yield the current data point
                yield self.x[idx], self.y[idx]

            # increment the epoch count
            epochs += 1


class CifarPreprocessor:
    def __init__(self, preprocessors):
        # initialize the instance variables
        self.preprocessors = preprocessors

    def preprocess(self, img, lbl):
        # loop through the preprocessors and preprocess the image
        for p in self.preprocessors:
            img = p.preprocess(img)

        # return the processed data
        return img, lbl
