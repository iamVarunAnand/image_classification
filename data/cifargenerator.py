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

    def __init__(self, x, y, batch_size, preprocessors = None, aug = None):
        # initialize the data
        self.x = x
        self.y = y

        # initialize the instance variables
        self.bs = batch_size
        self.preprocessors = preprocessors
        self.aug = aug

        # initialize additional variables
        self.num_images = self.x.shape[0]
        self.lb = LabelBinarizer()
        self.lb.fit(y)

    def generator(self, passes = np.inf):
        # initialize a variable to keep a count on the epochs
        epochs = 0

        # loop through the dataset indefinitely
        while(epochs < passes):
            # initialize the indices
            indices = list(range(self.num_images))
            np.random.shuffle(indices)

            # loop through the dataset in batches
            for i in range(0, self.num_images, self.bs):
                # extract the current indices
                cur_indices = sorted(indices[i: i + self.bs])

                # grab the current batch
                xb, yb = self.x[cur_indices], self.y[cur_indices]

                # if any preprocessors are supplied, apply them
                if self.preprocessors is not None:
                    # loop through the images
                    proc_x = []
                    for img in xb:
                        # loop through the preprocessors
                        for p in self.preprocessors:
                            img = p.preprocess(img)

                        proc_x.append(img)

                    # update the images
                    xb = np.array(proc_x)

                # one-hot encode the labels
                yb = self.lb.transform(yb)

                # if any augmenter is supplied, apply it
                if self.aug is not None:
                    xb, yb = next(self.aug.flow(xb, yb, batch_size = self.bs))

                # yield the current batch
                yield xb, yb

            # increment the epoch count
            epochs += 1
