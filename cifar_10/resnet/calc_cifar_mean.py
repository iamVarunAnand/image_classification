# import the necessary packages
from keras.datasets import cifar10
import numpy as np

# load the dataset
((x_train, y_train), (x_test, y_test)) = cifar10.load_data()

# compute the per channel mean
dataset_mean = np.mean(x_train, axis = (0, 1, 2))
dataset_std = np.std(x_train, axis = (0, 1, 2))

print(dataset_mean, dataset_std, sep = "\n")
