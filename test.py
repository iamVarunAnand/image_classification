# import the necessary packages
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model

from image_classification.preprocessors import ImageToArrayPreprocessor
from image_classification.preprocessors import MeanPreprocessor
from image_classification.data import CifarGenerator
from image_classification.utils import config
from image_classification.layers import Mish
import numpy as np
import argparse

# construct an argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required = True, help = "path to model weights")
args = vars(ap.parse_args())

# load the dataset
(_, _), (x_test, y_test) = cifar10.load_data()

# intialize the preprocessors
mp = MeanPreprocessor(mean = config.STATS["mean"], std = config.STATS["std"], normalize = True)
iap = ImageToArrayPreprocessor()

# initialize the data generator
test_datagen = CifarGenerator(x_test, y_test, config.BS, preprocessors = [mp, iap]).generator(passes = 1)

# load the model
model = load_model(args["weights"], custom_objects = {"Mish": Mish})
# model = load_model(args["weights"])

# evaluate the model
test_steps = np.ceil(x_test.shape[0] / config.BS)
H = model.evaluate(x = test_datagen, batch_size = config.BS, steps = test_steps)

# print the results
print(f"[INFO] test set loss: {np.round(H[0], 4)}")
print(f"[INFO] test set acc: {np.round(H[1], 4)}")
