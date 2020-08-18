# import the necessary packages
from .cifargenerator import CifarGenerator, CifarPreprocessor
from .mixupcifargenerator import MixUpCifarGenerator
from ..preprocessors import *
from ..utils.config import *

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
import tensorflow as tf


class DataDispatcher:
    def __init__(self):
        # initialize the instance variables
        self.num_train_imgs = None
        self.num_val_imgs = None
        self.num_test_imgs = None

    def get_train_data(self):
        # load the dataset and obtain the validation split
        (x_train, y_train), (_, _) = cifar10.load_data()
        (x_train, x_val, y_train, y_val) = train_test_split(x_train, y_train, test_size=0.1,
                                                            random_state=42, stratify=y_train)

        # calculate the total number of images
        self.num_train_imgs = x_train.shape[0]
        self.num_val_imgs = x_val.shape[0]

        # initialize the appropriate data generators
        if USE_MIXUP:
            train_gen = MixUpCifarGenerator(x_train, y_train).generator
        else:
            train_gen = CifarGenerator(x_train, y_train).generator

        val_gen = CifarGenerator(x_train, y_train).generator

        # initialize the appropriate preprocessors
        if USE_REFLECTION_PAD:
            pp = ReflectionPadPreprocessor(4)
        else:
            pp = PadPreprocessor(4)

        fp = FlipPreprocessor()
        patchp = PatchPreprocessor(32, 32)
        mp = MeanPreprocessor(mean=STATS["mean"], std=STATS["std"])

        train_cpp = CifarPreprocessor([pp, fp, patchp, mp])
        val_cpp = CifarPreprocessor([mp])

        # initialize the datasets
        train_ds = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float32, tf.uint8),
                                                  output_shapes=(list(x_train[0].shape), [10, ]))
        val_ds = tf.data.Dataset.from_generator(val_gen, output_types=(tf.float32, tf.uint8),
                                                output_shapes=(list(x_train[0].shape), [10, ]))

        # set up the tf.data.Dataset objects
        train_ds = (
            train_ds
            .shuffle(1000)
            .map(train_cpp.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(BS)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        val_ds = (
            val_ds
            .map(val_cpp.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(BS)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        # return the datasets
        return train_ds, val_ds

    def get_test_data(self):
        # load the dataset
        (_, _), (x_test, y_test) = cifar10.load_data()

        # calculate the total number of images
        self.num_test_imgs = x_test.shape[0]

        # initialize the data generator
        test_gen = CifarGenerator(x_test, y_test).generator

        # initialize the preprocessors
        mp = MeanPreprocessor(mean=STATS["mean"], std=STATS["std"])
        test_cpp = CifarPreprocessor([mp])

        # initialize the tf dataset
        test_ds = tf.data.Dataset.from_generator(test_gen, output_types=(tf.float32, tf.uint8),
                                                 output_shapes=(list(x_test[0].shape), [10, ]))

        # set up the tf.data.Dataset object
        test_ds = (
            test_ds
            .map(test_cpp.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(BS)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        # return the dataset
        return test_ds
