# import the necessary packages
from ..models import *

# initialize a dict mapping model names to the mdoels
MODELS = {
    "xresnet20": XResNet20(height=32, width=32, depth=3, classes=10),
    # "xresnet32": XResNet32(height = 32, width = 32, depth = 3, classes = 10),
    # "xresnet44": XResNet44(height = 32, width = 32, depth = 3, classes = 10),
    # "xresnet56": XResNet56(height = 32, width = 32, depth = 3, classes = 10),
    # "mxresnet20": MXResNet20(height = 32, width = 32, depth = 3, classes = 10),
    # "mxresnet32": MXResNet32(height = 32, width = 32, depth = 3, classes = 10),
    # "mxresnet44": MXResNet44(height = 32, width = 32, depth = 3, classes = 10),
    # "mxresnet56": MXResNet56(height = 32, width = 32, depth = 3, classes = 10)
    # "se-mxresnet20": SEMXResNet20(height = 32, width = 32, depth = 3, classes = 10),
    # "se-mxresnet32": SEMXResNet32(height = 32, width = 32, depth = 3, classes = 10),
    # "se-mxresnet44": SEMXResNet44(height = 32, width = 32, depth = 3, classes = 10),
    # "se-mxresnet56": SEMXResNet56(height = 32, width = 32, depth = 3, classes = 10)
}
