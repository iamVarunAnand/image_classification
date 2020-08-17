# import the necessary packages
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


class XResNeXt:
    """
    Implements a modified version of ResNeXt (https://arxiv.org/abs/1611.05431)
    according to (https://arxiv.org/abs/1812.01187) with some inputs from
    FastAI's implementation of CNN models

    Methods
    -------
    residual_module
        builds a single residual block based on input parameters
    build
        utilizes 'residual_module' to build entire model architecture based on input parameters
    """

    """
    Builds residual block

    Parameters
    ----------
    data: tf.keras.layer object
        reference to previous layer in the model architecture
    K: int
        number of filters to be used in the residual block
    stride: tuple (int, int)
        stride to be used by the conv layer in the block
    chan_dim: int
        BatchNormalization channel dimension (1 for NCHW, -1 for NHWC)
    groups: int
        cardinality of the model
    red: (optional) bool
        controls whether the spatial dimensions are to be reduced, defaults to False
    reg: (optional) float
        l2 regularization to be applied to every weighted layer, defaults to 1e-4 (acc to original ResNet paper)
    bn_eps: (optional) float
        Epsilon value for all BN layers, defaults to 2e-5 (acc to original ResNet paper)
    bn_mom: (optional) float
        Momentum value for all BN layers, defaults to 0.9 (acc to original ResNet paper)
    name: (optional) str
        name to be appended to all layers in the block, defaults to "res_block"
    """

    # static variable that stores the base width of the model
    base_width = None

    @staticmethod
    def residual_module(data, K, stride, chan_dim, groups, width_per_group, red=False, reg=1e-4, bn_eps=2e-5,
                        bn_mom=0.9, name="res_block"):
        # shortcut branch
        shortcut = data

        # compute the width
        width = int((K / float(XResNeXt.base_width) * width_per_group)) * groups

        # first bottleneck block - 1x1
        bn1 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom, name=name + "_bn1")(data)
        act1 = Activation("relu", name=name + "_relu1")(bn1)
        conv1 = Conv2D(width, (1, 1), use_bias=False, kernel_regularizer=l2(reg),
                       kernel_initializer="he_normal", name=name + "_conv1")(act1)

        # conv block - 3x3
        bn2 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom, name=name + "_bn2")(conv1)
        act2 = Activation("relu", name=name + "_relu2")(bn2)
        conv2 = Conv2D(width, (3, 3), strides=stride, groups=groups, padding="same", use_bias=False,
                       kernel_initializer="he_normal", kernel_regularizer=l2(reg), name=name + "_conv2")(act2)

        # second bottleneck block - 1x1
        bn3 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom, name=name + "_bn3")(conv2)
        act3 = Activation("relu", name=name + "_relu3")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(
            reg), kernel_initializer="he_normal", name=name + "_conv3")(act3)

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

    """
    Builds the ResNet model architecture

    Parameters
    ----------
    height: int
        height of the input images
    width: int
        width of the input images
    depth: int
        number of channels in the input images
    classes: int
        number of unique classes in the dataset
    stages: list of ints
        each element in the list specifies the number of residual blocks to be applied at that stage
    filters: list of ints
        each element in the list specifies the number of filters to be applied by conv layers in that stage
    groups: int
        cardinality of the model
    stem_type: (optional) str
        can be one of 'imagenet' or 'cifar'. defaults to 'imagenet'
    reg: (optional) float
        l2 regularization to be applied to every weighted layer, defaults to 1e-4 (acc to original ResNet paper)
    bn_eps: (optional) float
        Epsilon value for all BN layers, defaults to 2e-5 (acc to original ResNet paper)
    bn_mom: (optional) float
        Momentum value for all BN layers, defaults to 0.9 (acc to original ResNet paper)
    """

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
        XResNeXt.base_width = filters[1]

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
            x = XResNeXt.residual_module(x, filters[i + 1], stride, chan_dim, groups,
                                         width_per_group, reg=reg, red=True, bn_eps=bn_eps, bn_mom=bn_mom, name=name)

            # loop through the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a residual module
                name = f"stage{i + 1}_res_block{j + 2}"
                x = XResNeXt.residual_module(x, filters[i + 1], (1, 1), chan_dim, groups, width_per_group, reg=reg,
                                             bn_eps=bn_eps, bn_mom=bn_mom, name=name)

            # increment the number of layers
            n_layers += (3 * stages[i])

        # BN => RELU -> POOL
        x = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_mom, name="final_bn")(x)
        x = Activation("relu", name="final_relu")(x)
        x1 = GlobalAveragePooling2D(name="global_avg_pooling")(x)
        x2 = GlobalMaxPooling2D(name="global_max_pooling")(x)
        x = concatenate([x1, x2], axis=-1, name="concatenate")

        # softmax classifier
        sc = Dense(classes, kernel_initializer="he_normal", kernel_regularizer=l2(reg), name="classifier")(x)
        sc = Activation("softmax", name="softmax")(sc)

        # increment the number of layers
        n_layers += 1

        print(f"[INFO] {__class__.__name__}{n_layers} built successfully!")

        # return the constructed network architecture
        return Model(inputs=inputs, outputs=sc, name=f"{__class__.__name__}{n_layers}")


def XResNeXt29_16x2d(height=32, width=32, depth=3, classes=10):
    return XResNeXt.build(height, width, depth, classes,
                          stages=[3, 3, 3],
                          filters=[16, 64, 128, 256],
                          groups=16,
                          width_per_group=2,
                          stem_type="cifar")
