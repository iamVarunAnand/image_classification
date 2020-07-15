# import the necessary packages
import json

# dataset configs
USE_MIXUP = False
USE_REFLECTION_PAD = False
STATS = json.load(open("cifar10_stats.json", "r"))

# model configs
MODEL_NAME = "xresnet20"

# callback configs
TM_FIG_PATH = f"output/{MODEL_NAME}.png"
TM_JSON_PATH = f"output/{MODEL_NAME}.json"

# training configs
EPOCHS = 180
START_EPOCH = 0
BS = 128
INIT_LR = 1e-1
USE_LBL_SMOOTH = False
USE_COSINE = False
