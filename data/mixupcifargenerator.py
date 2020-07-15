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

    def __init__(self, x, y, batch_size, alpha = 0.4, preprocessors = None, aug = None):
        # initialize the cifar data
        self.x = x
        self.y = y

        # initialize the instance variables
        self.bs = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.alpha = alpha

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

                # initialize the other batch of indices
                if i + self.bs < self.num_images:
                    oth_indices = list(range(i, i + self.bs))
                else:
                    oth_indices = list(range(i, self.num_images))

                # grab the data batches
                x1, yb = self.x[cur_indices], self.y[cur_indices]
                x2 = self.x[oth_indices]

                # if any preprocessors are supplied, apply them
                if self.preprocessors is not None:
                    # loop through the images
                    proc_x1 = []
                    proc_x2 = []
                    for img1, img2 in zip(x1, x2):
                        # loop through the preprocessors
                        for p in self.preprocessors:
                            img1 = p.preprocess(img1)
                            img2 = p.preprocess(img2)

                        proc_x1.append(img1)
                        proc_x2.append(img2)

                    # update the images
                    x1 = np.array(proc_x1)
                    x2 = np.array(proc_x2)

                # randomly sample the lambda value from beta distribution.
                lamb = np.random.beta(self.alpha + 1, self.alpha, x1.shape[0])

                # remove possible duplicates
                lamb = np.maximum(lamb, 1 - lamb)

                # reshape the parameter to a suitable shape
                xlamb = lamb.reshape((-1, 1, 1, 1))

                # perform the mixup
                xb = (xlamb * x1) + ((1 - xlamb) * x2)

                # one-hot encode the labels
                yb = self.lb.transform(yb)

                # if any augmentation is supplied, apply it
                if self.aug is not None:
                    xb, yb = next(self.aug.flow(xb, yb, batch_size = self.bs))

                # yield the current batch
                yield xb, yb

            # increment the epoch count
            epochs += 1
