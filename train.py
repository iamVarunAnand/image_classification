# add the current directory to PYTHONPATH
import sys
import os
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import the necessary packages
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD

from image_classification.callbacks import TrainingMonitor, CosineScheduler
from image_classification.data import CifarGenerator, MixUpCifarGenerator
from image_classification.preprocessors import ImageToArrayPreprocessor
from image_classification.preprocessors import ReflectionPadPreprocessor
from image_classification.preprocessors import PatchPreprocessor
from image_classification.preprocessors import FlipPreprocessor
from image_classification.preprocessors import MeanPreprocessor
from image_classification.preprocessors import PadPreprocessor
from image_classification.utils.dispatcher import MODELS
from image_classification.utils import resnet_lr_scheduler
from image_classification.utils import config

import numpy as np

# load the dataset and obtain the validation split
(x_train, y_train), (_, _) = cifar10.load_data()
(x_train, x_val, y_train, y_val) = train_test_split(x_train, y_train, test_size = 0.1,
                                                    random_state = 42, stratify = y_train)

# initialize the preprocessors
if config.USE_REFLECTION_PAD:
    pp = ReflectionPadPreprocessor(4)
else:
    pp = PadPreprocessor(4)

fp = FlipPreprocessor(0.5)
patchp = PatchPreprocessor(32, 32)
mp = MeanPreprocessor(mean = config.STATS["mean"], std = config.STATS["std"], normalize = True)
iap = ImageToArrayPreprocessor()

# initialize the data generators
if config.USE_MIXUP:
    train_datagen = MixUpCifarGenerator(x_train, y_train, config.BS,
                                        preprocessors = [pp, fp, patchp, mp, iap]).generator()
else:
    train_datagen = CifarGenerator(x_train, y_train, config.BS, preprocessors = [pp, fp, patchp, mp, iap]).generator()

val_datagen = CifarGenerator(x_val, y_val, config.BS, preprocessors = [mp, iap]).generator()

# compute some additional training constants
steps_per_epoch = np.ceil(x_train.shape[0] / config.BS)
validation_steps = np.ceil(x_val.shape[0] / config.BS)

# initialize the callbacks
if config.USE_COSINE:
    lrs = CosineScheduler(config.INIT_LR, steps_per_epoch, config.EPOCHS, warmup = 5)
else:
    lrs = LearningRateScheduler(resnet_lr_scheduler)

tm = TrainingMonitor(config.TM_FIG_PATH, json_path = config.TM_JSON_PATH, start_at = config.START_EPOCH)
mc = ModelCheckpoint(f"weights/{config.MODEL_NAME}" + "_{epoch:03d}.h5")
callbacks = [tm, mc, lrs]

# initialize the model
model = MODELS[config.MODEL_NAME]

# initialize the loss fn
if config.USE_LBL_SMOOTH:
    loss = CategoricalCrossentropy(label_smoothing = 0.1)
else:
    loss = CategoricalCrossentropy()

# initialize the optimizer and compile the model
opt = SGD(lr = config.INIT_LR, momentum = 0.9)
model.compile(loss = loss, optimizer = opt, metrics = ["accuracy"])

# train the model
model.fit(x = train_datagen, epochs = config.EPOCHS, steps_per_epoch = steps_per_epoch,
          validation_data = val_datagen, validation_steps = validation_steps,
          callbacks = callbacks, initial_epoch = config.START_EPOCH)
