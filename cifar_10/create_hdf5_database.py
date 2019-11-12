# import the necessary packages
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import preprocess_input
from passion.datasets import HDF5DatasetWriter
from imutils import paths
import numpy as np
import argparse
import cv2
import os

# construct the argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "path to dataset directory")
ap.add_argument("-o", "--output", required = True, help = "path to store database")
args = vars(ap.parse_args())

# define a list containing the CIFAR-10 label names
label_names = {"airplane" : 1, "automobile" : 2, "bird" : 3, "cat" : 4, "deer" : 5, "dog" : 6, "frog" : 7, "horse" : 8, "ship" : 9, "truck" : 10}

# grab the paths to the images in the dataset
image_paths = list(paths.list_images(args["dataset"]))

# grab the labels from the image paths
labels = [image_path.split(os.path.sep)[-2] for image_path in image_paths]

# split the image paths into train and validation sets
(train_paths, val_paths, train_labels, val_labels) = train_test_split(image_paths, labels, test_size = 0.1, random_state = 42, stratify = labels)

# build the paths to the databases
TRAIN_DB = os.path.sep.join([args["output"], "cifar_10_224_train.h5"])
VAL_DB = os.path.sep.join([args["output"], "cifar_10_224_val.h5"])

# initialize the dataset
dataset = [(train_paths, train_labels, TRAIN_DB), (val_paths, val_labels, VAL_DB)]

# loop through the dataset
for (image_paths, labels, db) in dataset:
    # print an info about the current database
    print("[INFO] building {}".format(db))

    # initialize the dataset writer
    dims = (len(image_paths), 224, 224, 3)
    db_writer = HDF5DatasetWriter(output_path = db, dims = dims)
    db_writer.open()

    # initialize the batch size and loop through the image paths
    for (i, (image_path, label)) in enumerate(zip(image_paths, labels)):
        # read the image from disk
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # preprocess the input image
        image = preprocess_input(image)

        # add the image to the database
        db_writer.add(image, label_names[label])

        # print and update
        if (i + 1) > 0 and (i + 1) % 1024 == 0:
            print("[INFO] processed {} / {}".format(i + 1, len(image_paths)))

    # close the database
    db_writer.close()
