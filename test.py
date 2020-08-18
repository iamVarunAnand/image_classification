# import the necessary packages
from tensorflow.keras.models import load_model
from image_classification.data import DataDispatcher
from image_classification.utils import config
from image_classification.layers import Mish
import numpy as np
import argparse

# construct an argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to model weights")
args = vars(ap.parse_args())

# load the dataset
dd = DataDispatcher()
test_ds = dd.get_test_data()

# load the model
model = load_model(args["weights"], custom_objects={"Mish": Mish})
# model = load_model(args["weights"])

# evaluate the model
test_steps = np.ceil(dd.num_test_imgs / config.BS)
H = model.evaluate(x=test_ds, batch_size=config.BS, steps=test_steps)

# print the results
print(f"[INFO] test set loss: {np.round(H[0], 4)}")
print(f"[INFO] test set acc: {np.round(H[1], 4)}")
