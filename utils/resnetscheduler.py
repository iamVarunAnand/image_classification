"""
Utility function that implements the learning rate schedule followed in the original ResNet paper
(https://arxiv.org/abs/1512.03385).

To be passed to tf.keras.callbacks.LearningRateScheduler during training
"""

# import the necessary packages
from .config import *


def resnet_lr_scheduler(self, epoch):
    init_lr = INIT_LR

    if epoch < 1:
        lr = init_lr / 10
    elif epoch < 90:
        lr = init_lr
    elif epoch < 135:
        lr = init_lr / 10
    else:
        lr = init_lr / 100

    return lr
