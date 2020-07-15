# import the necessary packages
from ..models import *

# initialize a dict mapping model names to the mdoels
MODELS = {
    # IMAGENET MODELS
    # "xresnet18": XResNet18(height = 224, width = 224, depth = 10, classes = 1000),
    # "xresnet34": XResNet34(height = 224, width = 224, depth = 10, classes = 1000),
    # "xresnet50": XResNet50(height = 224, width = 224, depth = 10, classes = 1000),
    # "xresnet101": XResNet101(height = 224, width = 224, depth = 10, classes = 1000),
    # "xresnet152": XResNet152(height = 224, width = 224, depth = 10, classes = 1000),

    # CIFAR MODELS
    "xresnet20": XResNet20(height = 32, width = 32, depth = 3, classes = 10),
    # "xresnet32": XResNet32(height = 32, width = 32, depth = 3, classes = 10),
    # "xresnet44": XResNet44(height = 32, width = 32, depth = 3, classes = 10),
    # "xresnet56": XResNet56(height = 32, width = 32, depth = 3, classes = 10)
}
