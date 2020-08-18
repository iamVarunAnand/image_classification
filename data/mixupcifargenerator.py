# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
import numpy as np


class MixUpCifarGenerator:
    """
    Implements [mixup] (https://arxiv.org/abs/1710.09412) training method
    Implementation based on FastAI's slightly modified version, explained here:
    https://forums.fast.ai/t/mixup-data-augmentation/22764

    Attributes
    ----------
    x: np ndarray
        array of images
    y: np ndarray
        array of class labels
    batch_size: int
        batch size to be used while generating batches
    alpha: (optional) float
        beta distribution sampling parameter. defaults to 0.4 (as mentioned in the original paper)
    preprocessors: (optional) list
        all the preprocessors to be applied to the input data. defaults to None
    aug: (optional) tf.keras.preprocessing.image.ImageDataGenerator object
        data augmentation to be applied to the input data. defaults to None
    """

    def __init__(self, x, y, alpha=0.4):
        # initialize the cifar data
        self.x = x
        self.y = y

        # initialize the instance variables
        self.alpha = alpha
        self.num_images = self.x.shape[0]

        # convert the labels from integers into vectors
        self.lb = LabelBinarizer()
        self.y = self.lb.fit_transform(self.y)

    def generator(self, passes=np.inf):
        # initialize a variable to keep a count on the epochs
        epochs = 0

        # loop through the dataset indefinitely
        while(epochs < passes):
            # initialize the indices
            indices = list(range(self.num_images))
            np.random.shuffle(indices)

            # loop through the dataset
            for i in range(0, self.num_images):
                # extract the indices
                idx1 = indices[i]
                idx2 = i

                # grab the data batches
                x1, yb = self.x[idx1], self.y[idx1]
                x2 = self.x[idx2]

                # randomly sample the lambda value from beta distribution.
                lamb = np.random.beta(self.alpha + 1, self.alpha)

                # remove possible duplicates
                lamb = np.maximum(lamb, 1 - lamb)

                # reshape the parameter to a suitable shape
                xlamb = lamb.reshape((1, 1, 1))

                # perform the mixup
                xb = (xlamb * x1) + ((1 - xlamb) * x2)

                # yield the current data
                yield xb, yb

            # increment the epoch count
            epochs += 1
