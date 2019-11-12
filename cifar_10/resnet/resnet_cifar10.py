# import the necessary packages
from pyimagesearch.callbacks import CyclicLR
from pyimagesearch.callbacks import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers import Lambda
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.regularizers import l2
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys

# set the numpy random number generator seed
np.random.seed(7)

# initialize the tensorflow session to allow gpu memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)
set_session(session)

class RSModel:
    @staticmethod
    def build(height, width, depth, classes):
        # initialize the input shape
        if K.image_data_format() == "channels_last":
            input_shape = (height, width, depth)
        else:
            input_shape = (depth, height, width)

        # initialize the model
        model = Sequential()

        # reshape layer
        model.add(Lambda(lambda image : tf.image.resize(image, (224, 224)), input_shape = input_shape))

        # initialize the base model
        model.add(ResNet50(include_top = False, weights = "imagenet", input_shape = (224, 224, 3)))

        # softmax layer
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(32, kernel_initializer = "he_normal", kernel_regularizer = l2(0.001)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(32, kernel_initializer = "he_normal", kernel_regularizer = l2(0.001)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes, kernel_initializer = "he_normal", kernel_regularizer = l2(0.001)))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

# initialize global parameters
MIN_LR = 1e-5
MAX_LR = 1e-2
STEP_SIZE = 4
BATCH_SIZE = 128
EPOCHS = 24
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# load the dataset
((x_train, y_train), (x_test, y_test)) = cifar10.load_data()

# preprocess input images
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

# convert the labels from integers into vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# initialize the callbacks
checkpoint = ModelCheckpoint(filepath = "models/RS_32_32(2)_{epoch:03d}.h5", period = 5)

# initialize global parameters
MIN_LR = 1e-5
MAX_LR = 1e-2
STEP_SIZE = 4
BATCH_SIZE = 64
EPOCHS = 48
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# initialize the model
model = RSModel.build(32, 32, 3, 10)
opt = SGD(lr = MIN_LR, momentum = 0.9, nesterov = True)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

# initialize the cyclic lr callback
clr = CyclicLR(mode = "triangular2",
               base_lr = MIN_LR,
               max_lr = MAX_LR,
               step_size = STEP_SIZE * (x_train.shape[0] // BATCH_SIZE))

# initialize the model checkpointer
mc = ModelCheckpoint(filepath = "models/RS_32_32_{epoch:03d}.h5", period = 4)

# initialize the cyclic lr callback
clr = CyclicLR(mode = "triangular", base_lr = MIN_LR, max_lr = MAX_LR,
               step_size = STEP_SIZE * (x_train.shape[0] // BATCH_SIZE))

# initialize the training monitor
ID = np.random.randint(0, 100)
fig_path = "plots/{}.png".format(ID)
tm = TrainingMonitor(fig_path = fig_path)

# initialize the data augmentation generator
aug = ImageDataGenerator(rotation_range = 30, height_shift_range = 0.1, width_shift_range = 0.1,
                         shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = "nearest")

# initialize the model
model = RSModel.build(32, 32, 3, 10)
opt = SGD(lr = MIN_LR, momentum = 0.9, nesterov = True)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

# train the model
print("[INFO] training the model...")
H = model.fit_generator(aug.flow(x_train, y_train, batch_size = BATCH_SIZE),
                        validation_data = (x_test, y_test),
                        steps_per_epoch = np.ceil(x_train.shape[0] / BATCH_SIZE),
                        epochs = EPOCHS, verbose = 1, callbacks = [clr, mc, tm])

# evaluating the model
print("[INFO] evaluating the model...")
predictions = model.predict(x_test, batch_size = BATCH_SIZE)
print(classification_report(y_train.argmax(axis = 1), predictions.argmax(axis = 1), target_names=CLASSES))

# plot the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label = "loss")
plt.plot(N, H.history["acc"], label = "acc")
plt.plot(N, H.history["val_loss"], label = "val_loss")
plt.plot(N, H.history["val_acc"], label = "val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.savefig("plots/triangular_training_plot.png")

# plot the learning rate history
N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclic Learning Rate Schedule (trianglar)")
plt.xlabel("Iterations")
plt.ylabel("Learning Rate")
plt.savefig("plots/triangular_lr_plot.png")
