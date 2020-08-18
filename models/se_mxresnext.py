import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import the necessary packages
from image_classification.layers import Mish

from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import multiply
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np


class SEMXResNeXt:
    # static variable that stores the base width of the model
    base_width = None

    @staticmethod
    def squeeze_excite_block(tensor, ratio=16, name="se_block"):
        init = tensor
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = init.shape[channel_axis]
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D(name=name + "_gap")(init)
        se = Reshape(se_shape, name=name + "_reshape")(se)
        se = Dense(filters // ratio, kernel_initializer='he_normal', use_bias=False, name=name + "_squeeze")(se)
        se = Activation("relu", name=name + "_squeeze_relu")(se)
        se = Dense(filters, kernel_initializer='he_normal', use_bias=False, name=name + "_excite")(se)
        se = Activation("sigmoid", name=name + "_excite_sigmoid")(se)

        if K.image_data_format() == 'channels_first':
            se = Permute((3, 1, 2))(se)

        x = multiply([init, se], name=name + "_scale")
        return x

    @staticmethod
    def residual_module(data, K, stride, chan_dim, groups, width_per_group, red=False, reg=1e-4, bn_eps=2e-5,
                        bn_mom=0.9, name="res_block"):
        # shortcut branch
        shortcut = data

        # compute the width
        width = int((K / float(SEMXResNeXt.base_width) * width_per_group)) * groups

        # first bottleneck block - 1x1
        bn1 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom, name=name + "_bn1")(data)
        act1 = Mish(name=name + "_mish1")(bn1)
        conv1 = Conv2D(width, (1, 1), use_bias=False, kernel_regularizer=l2(reg),
                       kernel_initializer="he_normal", name=name + "_conv1")(act1)

        # conv block - 3x3
        bn2 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom, name=name + "_bn2")(conv1)
        act2 = Mish(name=name + "_mish2")(bn2)
        conv2 = Conv2D(width, (3, 3), strides=stride, groups=groups, padding="same", use_bias=False,
                       kernel_initializer="he_normal", kernel_regularizer=l2(reg), name=name + "_conv2")(act2)

        # second bottleneck block - 1x1
        bn3 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom, name=name + "_bn3")(conv2)
        act3 = Mish(name=name + "mish3")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(
            reg), kernel_initializer="he_normal", name=name + "_conv3")(act3)

        # se module
        conv3 = SEMXResNeXt.squeeze_excite_block(conv3, name=name + "_se_block")

        # if dimensions are to be reduced, apply a conv layer to the shortcut
        if red:
            shortcut = AveragePooling2D(pool_size=(2, 2), strides=stride,
                                        padding="same", name=name + "_avg_pool")(act1)
            shortcut = Conv2D(K, (1, 1), strides=(1, 1), use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=l2(reg), name=name + "_red")(shortcut)
            shortcut = BatchNormalization(name=name + "_red_bn")(shortcut)

        # add the shortcut and final conv
        x = add([conv3, shortcut], name=name + "_add")

        # return the addition as the output of the residual block
        return x

    @staticmethod
    def build(height, width, depth, classes, stages, filters, groups, width_per_group, stem_type="imagenet",
              reg=1e-4, bn_eps=2e-5, bn_mom=0.9):
        # set the input shape
        if K.image_data_format() == "channels_last":
            input_shape = (height, width, depth)
            chan_dim = -1
        else:
            input_shape = (depth, height, width)
            chan_dim = 1

        # set the base width
        SEMXResNeXt.base_width = filters[1]

        # initialize a counter to keep count of the total number of layers in the model
        n_layers = 0

        # input block
        inputs = Input(shape=input_shape)

        # stem
        if stem_type == "imagenet":
            x = Conv2D(filters[0], (3, 3), strides=(2, 2), use_bias=False, padding="same",
                       kernel_initializer="he_normal", kernel_regularizer=l2(reg), name="stem_conv1")(inputs)
            x = Conv2D(filters[0], (3, 3), strides=(1, 1), use_bias=False, padding="same",
                       kernel_initializer="he_normal", kernel_regularizer=l2(reg), name="stem_conv2")(x)
            x = Conv2D(filters[0], (3, 3), strides=(1, 1), use_bias=False, padding="same",
                       kernel_initializer="he_normal", kernel_regularizer=l2(reg), name="stem_conv3")(x)
            x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="stem_max_pool")(x)
        elif stem_type == "cifar":
            x = Conv2D(filters[0], (3, 3), use_bias=False, padding="same", kernel_initializer="he_normal",
                       kernel_regularizer=l2(reg), name="stem_conv")(inputs)

        # increment the number of layers
        n_layers += 1

        # loop through the stages
        for i in range(0, len(stages)):
            # set the stride value
            stride = (1, 1) if i == 0 else (2, 2)

            name = f"stage{i + 1}_res_block1"
            x = SEMXResNeXt.residual_module(x, filters[i + 1], stride, chan_dim, groups,
                                            width_per_group, reg=reg, red=True, bn_eps=bn_eps, bn_mom=bn_mom, name=name)

            # loop through the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a residual module
                name = f"stage{i + 1}_res_block{j + 2}"
                x = SEMXResNeXt.residual_module(x, filters[i + 1], (1, 1), chan_dim, groups, width_per_group, reg=reg,
                                                bn_eps=bn_eps, bn_mom=bn_mom, name=name)

            # increment the number of layers
            n_layers += (3 * stages[i])

        # BN -> MISH -> POOL
        x = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom, name="final_bn")(x)
        x = Mish(name="final_mish")(x)
        x1 = GlobalAveragePooling2D(name="global_avg_pooling")(x)
        x2 = GlobalMaxPooling2D(name="global_max_pooling")(x)
        x = concatenate([x1, x2], axis=-1, name="concatenate")

        # softmax classifier
        sc = Dense(classes, kernel_initializer="he_normal", kernel_regularizer=l2(reg), name="classifier")(x)
        sc = Activation("softmax", name="softmax")(sc)

        # increment the number of layers
        n_layers += 1

        print(f"[INFO] {__class__.__name__}{n_layers}_{groups}x{width_per_group}d built successfully!")

        # return the constructed network architecture
        return Model(inputs=inputs, outputs=sc, name=f"{__class__.__name__}{n_layers}_{groups}x{width_per_group}d")


def SEMXResNeXt29_16x2d(height=32, width=32, depth=3, classes=10):
    return SEMXResNeXt.build(height, width, depth, classes,
                             stages=[3, 3, 3],
                             filters=[64, 64, 128, 256],
                             groups=16,
                             width_per_group=2,
                             stem_type="cifar")


def SEMXResNeXt47_16x2d(height=32, width=32, depth=3, classes=10):
    return SEMXResNeXt.build(height, width, depth, classes,
                             stages=[5, 5, 5],
                             filters=[64, 64, 128, 256],
                             groups=16,
                             width_per_group=2,
                             stem_type="cifar")


def SEMXResNeXt65_16x2d(height=32, width=32, depth=3, classes=10):
    return SEMXResNeXt.build(height, width, depth, classes,
                             stages=[7, 7, 7],
                             filters=[64, 64, 128, 256],
                             groups=16,
                             width_per_group=2,
                             stem_type="cifar")


def SEMXResNeXt83_16x2d(height=32, width=32, depth=3, classes=10):
    return SEMXResNeXt.build(height, width, depth, classes,
                             stages=[9, 9, 9],
                             filters=[64, 64, 128, 256],
                             groups=16,
                             width_per_group=2,
                             stem_type="cifar")
