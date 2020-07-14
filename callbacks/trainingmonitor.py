# import the necessary packages
from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
    """
    Utility callback that is used to generate the loss and accuracy plot at the end of each epoch

    Attributes
    ----------
    fig_path: str
        path to store the generated plot
    json_path: str, (optional)
        path to store a log of the training metrics, defaults to None
    start_at: int, (optional)
        epoch to start plotting from (useful for stop-start training), defaults to 0
    """

    def __init__(self, fig_path, start_at = 0, json_path = None):
        # store the output path for the figure, the path to the JSON serialized file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.fig_path = fig_path
        self.json_path = json_path
        self.start_at = start_at

    def on_train_begin(self, logs = {}):
        # initialize the history dictionary
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())

                # check to see if a starting epoch was supplied
                if self.start_at > 0:
                    # loop over the entries in the history log and trim any entries that are past the starting epoch
                    for key in self.H.keys():
                        self.H[key] = self.H[key][:self.start_at]

    def on_epoch_end(self, epoch, logs = {}):
        # loop over the logs and update the loss, accuracy etc, for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        # check to see if the training history should be serialized to file
        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H))
            f.close()

        # ensure atleast two epochs have passed before plotting
        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.figure()
            plt.style.use("ggplot")
            plt.plot(N, self.H["loss"], label = "train_loss")
            plt.plot(N, self.H["val_loss"], label = "val_loss")
            plt.plot(N, self.H["accuracy"], label = "acc")
            plt.plot(N, self.H["val_accuracy"], label = "val_acc")
            plt.title("Training Loss [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

            # save the figure
            plt.savefig(self.fig_path)
            plt.close()
