# import the necessary packages
from .dispatcher import MODELS

# model configs
model_name = "xresnet20"
MODEL = MODELS[model_name]

# training configs
EPOCHS = 180
BS = 128
