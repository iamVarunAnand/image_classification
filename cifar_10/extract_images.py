# import the necessary packages
import numpy as np
import argparse
import pickle
import cv2
import os
# construct the argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "path to pickled dataset directory")
ap.add_argument("-o", "--output", required = True, help = "path to output directory")
ap.add_argument("-s", "--size", type = int, default = 32, help = "output image size")
args = vars(ap.parse_args())

# define a list containing the CIFAR-10 label names
label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def unpickle(file):
    try:
        # skip if the current file is the metadata file
        if file.split(os.path.sep)[-1] == "batches.meta":
            print("[INFO] skipping metadata file")
            return None
        # else unpickle the file into a dictionary and return it
        else:
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')

            return dict
    except:
        print("[INFO] not a binary file, skipping...")
        return None

def extract_images(dict, dataset_type = "train"):
    # build the path to the output directory
    OUTPUT_PATH = os.path.sep.join([args["output"], str(args["size"]), dataset_type])

    # if the output directory doesn't exist, create it
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # extract the required values from the dictionary
    filenames = dict[b"filenames"]
    images = dict[b"data"]
    labels = dict[b"labels"]

    for (i, (image, label, filename)) in enumerate(zip(images, labels, filenames)):
        # reshape the image feature vector into a 2D image
        image = image.reshape(3, 32, 32)
        image = image.swapaxes(0, 1)
        image = image.swapaxes(1, 2)

        # build the path to the output directory
        class_label = label_names[label]
        OUTPUT_IMAGE_DIR = os.path.sep.join([OUTPUT_PATH, class_label])

        # if the output image directory doesn't exist, create it
        if not os.path.exists(OUTPUT_IMAGE_DIR):
            os.makedirs(OUTPUT_IMAGE_DIR)

        # build the path to the output file
        OUTPUT_IMAGE_PATH = os.path.sep.join([OUTPUT_IMAGE_DIR, filename.decode("utf-8")])

        # if the output image size is not 32, reshape the image to the appropriate size
        if args["size"] is not 32:
            # resize the image
            image = cv2.resize(image, (args["size"], args["size"]))

        # save the image to disk
        cv2.imwrite(OUTPUT_IMAGE_PATH, image)

        # print an update
        print("[INFO] processed {} / {}".format(i + 1, len(labels)))


print(os.listdir(args["input"]))

for file in os.listdir(args["input"]):
    # build the path to the pickled file
    PATH = os.path.sep.join([args["input"], file])

    # unpickle the file
    dict = unpickle(PATH)

    if dict is not None:
        if file == "test_batch":
            extract_images(dict, dataset_type = "test")
        else:
            extract_images(dict)


# dataset_dict = unpickle(args["input"])
# print(dataset_dict.keys())
