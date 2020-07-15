# import the necessary packages
from tensorflow.keras.datasets import cifar10
import numpy as np
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# define the output path to store the computed stats
JSON_PATH = "cifar10_stats.json"

# initialize the dataset
(x_train, _), (_, _) = cifar10.load_data()

# normalize the images into [0, 1]
x_train = x_train.astype("float") / 255.0

# compute the stats
mean = np.mean(x_train, axis = (0, 1, 2))
std = np.std(x_train, axis = (0, 1, 2))

# initialize the stats dict and write it to the output json file
stats = {"mean": list(np.round(mean, 4)), "std": list(np.round(std, 4))}
f = open(JSON_PATH, "w+")
json.dump(stats, f)
f.close()
