# import the necessary packages
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K


class LinearWarmupScheduler(Callback):
    """
    Implements a learning rate schedule with a linear warmup from 0 to max_lr as specified
    by the callback parameters

    Attributes
    ----------
    max_lr: float
        maximum learning rate to be used while training the model
    steps_per_epoch: int
        number of iterations in one epoch. typically set to int(np.ceil(num_imgs / batch_size))
    tot_epochs: int
        total number of training epochs
    warmup: (optional) int
        number of epochs to warmup for, defaults to 5
    """

    def __init__(self, max_lr, steps_per_epoch, tot_epochs, warmup = 5):
        # check if total epochs is atleast = warmup
        if tot_epochs < warmup:
            msg = "[ERROR] total epochs cannot be less than warmup period"
            raise ValueError(msg)

        # parent class constructor
        super(LinearWarmupScheduler, self).__init__()

        # initialize the instance variables
        self.max_lr = max_lr
        self.warm_steps = steps_per_epoch * warmup
        self.reg_steps = steps_per_epoch * (tot_epochs - warmup)
        self.steps_per_epoch = steps_per_epoch
        self.history = {"lrs": []}

    def on_epoch_begin(self, epoch, logs = None):
        # set the iterations to the number of batches seen (to facilitate stop and start training)
        self.iterations = self.steps_per_epoch * epoch

    def on_batch_begin(self, batch, logs = None):
        # increment the number of iterations
        self.iterations += 1

        # calculate the learning rate
        if self.iterations <= self.warm_steps:
            lr = (self.iterations / self.warm_steps) * self.max_lr
        else:
            lr = self.max_lr

        # update the learning rate
        K.set_value(self.model.optimizer.lr, lr)

        # add the current learning rate to the history dictionary
        self.history["lrs"].append(lr)
