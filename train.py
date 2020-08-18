# add the current directory to PYTHONPATH
from pathlib import Path
import sys
import os
path = Path(os.getcwd())
sys.path.append(str(path.parent))

# import the necessary packages
from sklearn.model_selection import train_test_split
from image_classification.callbacks import TrainingMonitor, CosineScheduler
from image_classification.data import DataDispatcher
from image_classification.utils.dispatcher import MODELS
from image_classification.utils import resnet_lr_scheduler
from image_classification.utils import config
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import numpy as np


# initialize the training data
dd = DataDispatcher()
train_ds, val_ds = dd.get_train_data()

# compute some additional training constants
steps_per_epoch = np.ceil(dd.num_train_imgs / config.BS)
validation_steps = np.ceil(dd.num_val_imgs / config.BS)

# initialize the callbacks
if config.USE_COSINE:
    lrs = CosineScheduler(config.INIT_LR, steps_per_epoch, config.EPOCHS, warmup=5)
else:
    lrs = LearningRateScheduler(resnet_lr_scheduler)

tm = TrainingMonitor(config.TM_FIG_PATH, json_path=config.TM_JSON_PATH, start_at=config.START_EPOCH)
mc = ModelCheckpoint(f"weights/{config.MODEL_NAME}" + "_{epoch:03d}.h5")
callbacks = [tm, mc, lrs]

# initialize the model
model = MODELS[config.MODEL_NAME]

# initialize the loss fn
if config.USE_LBL_SMOOTH:
    loss = CategoricalCrossentropy(label_smoothing=0.1)
else:
    loss = CategoricalCrossentropy()

# initialize the optimizer and compile the model
opt = SGD(lr=config.INIT_LR, momentum=0.9)
model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

# train the model
model.fit(x=train_ds, epochs=config.EPOCHS, steps_per_epoch=steps_per_epoch,
          validation_data=val_ds, validation_steps=validation_steps,
          callbacks=callbacks, initial_epoch=config.START_EPOCH)
