# Image Classification using ResNets
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

![GitHub Issues](https://img.shields.io/github/issues/iamVarunAnand/image_classification)
![GitHub stars](https://img.shields.io/github/stars/iamVarunAnand/image_classification)
![GitHub forks](https://img.shields.io/github/forks/iamVarunAnand/image_classification)
[![GitHub license](https://img.shields.io/github/license/iamVarunAnand/image_classification.svg)](https://github.com/iamVarunAnand/image_classification/blob/master/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PR-welcome-brightgreen)](http://makeapullrequest.com)

This project aimed to replicate and improve ResNet SOTA results on CIFAR10. I achieved a **6.90%** error rate using a **20-layer** ResNet with **0.27M** parameters, improving upon the original ResNet20's 8.75% error rate and matching ResNet56's 6.97% (presented [here](https://arxiv.org/abs/1512.03385)). Replacing ResNet blocks with ResNeXt further reduced the error rate to 5.32%.

|                    MODEL                     | TEST ERR | TEST ACC  |
| :------------------------------------------: | :------: | :-------: |
| [ResNet20](https://arxiv.org/abs/1512.03385) |   8.75   |   91.25   |
|                  XResNet20                   |   8.18   |   91.82   |
|                  MXResNet20                  |   7.93   |   92.07   |
|                SE-MXResNet20                 |   7.81   |   92.19   |
|                + cosine decay                |   7.53   |   92.47   |
|              + label smoothing               |   7.49   |   92.51   |
|                   + mixup                    |   7.03   |   92.97   |
|             + reflection padding             | **6.90** | **93.10** |
- **Model Architecture updates**
	1. All the updates mentioned in the [Bag of Tricks](https://arxiv.org/abs/1812.01187) paper - *XResNet*
	2. [Mish](https://arxiv.org/abs/1908.08681) activation instead of ReLU - *MXResNet*
	3. [Squeeze-Excite (SE)](https://arxiv.org/abs/1709.01507) blocks wherever possible - *SE-MXResNet*
  
- **Updates to the Training procedure**
	1. [MixUp](https://arxiv.org/abs/1710.09412) training.
	2. [Cosine-decayed](https://arxiv.org/abs/1608.03983) learning rate schedule.
	3. Adding Label Smoothing.
	4. Using reflection padding instead of zero padding for the input images.

### Results for other models
|     MODEL     | PARAMS | TEST ERR | TEST ACC |
| :-----------: | :----: | :------: | :------: |
| SE-MXResNet20 | 0.27M  |   6.90   |  93.10   |
| SE-MXResNet32 | 0.47M  |   6.20   |  93.80   |
| SE-MXResNet44 | 0.67M  |   6.12   |  93.88   |
| SE-MXResNet56 | 0.86M  |   5.64   |  94.36   |

### Update:
I have updated the repository with ResNeXt based models to assess their influence in improving the performance. For this purpose, I have modified the original ResNeXt models presented [here](https://arxiv.org/abs/1611.05431), such that they have roughly the same complexities as their ResNet counterparts.
1. **Addition of bottlenecks:** Since ResNeXt models make use of bottleneck residual blocks, I have increased the width of the ResNet models by 4x, which accounts for the reduction the feature maps undergo while entering a bottleneck block. The ResNeXt models therefore have [64, 64, 128, 256] filters as opposed to [16, 16, 32, 64] in ResNet.
2. **Determining cardinality:** To do so, I have referred to the process followed in the original paper, which is demonstrated in the following table. I finally settled on using a cardinality of 16, which translates to a bottleneck width of 2. In other words, the bottleneck conv layer is implemented as a grouped convolution consisting of 16 groups, each having 2 feature maps.

| Cardinality - C | Bottleneck width - d | Group Conv width |
| :-------------: | :------------------: | :--------------: |
|        1        |          16          |        16        |
|        2        |          10          |        20        |
|        4        |          6           |        24        |
|     **16**      |        **2**         |      **32**      |

Following this process, I developed a model called *XResNeXt29_16x2d*. This model has 0.32M parameters, comparable to XResNet20. The extra 9 layers are a result of using the bottleneck blocks, which consist of 3 conv layers as opposed to the basic block's 2. The performance of this model is shown below.
|        MODEL         | PARAMS | TEST ERR | TEST ACC  |
| :------------------: | :----: | :------: | :-------: |
|      XResNet29       | 0.31M  |   7.62   |   92.38   |
|   XResNeXt29_16x2d   | 0.32M  |   6.70   |   93.30   |
| SE-MXResNeXt29_16x2d | 0.36M  | **5.32** | **94.68** |

**After adding all the updates, this 29 layer model outperforms the 56 layer SE-MXResNet, while using less than half the number of parameters.**

## Replicating the results
- To replicate the results obtained above, first, clone this repository to your local machine and install all the necessary packages. Optionally, prior to running these commands, you can create a virtual environment by following the steps listed at [this](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/) link
```
$ git clone https://github.com/iamVarunAnand/image_classification.git
$ cd image_classification
$ pip install -r requirements.txt
```
- All training related configurations are specified in a separate config file, located in *utils/config.py.* All the available options are listed below:
```python
# dataset configs
USE_MIXUP = False # determines whether to use mixup training
USE_REFLECTION_PAD = False # determines if reflection pad is to be used for the input images, instead of zero pad

# model configs
MODEL_NAME = "xresnet20" # model to be used for training

# training configs
EPOCHS = 180 # number of training epochs
START_EPOCH = 0 # epoch to start training at (useful for stop-start training)
BS = 128 # batch size to be used while training
INIT_LR = 1e-1 # starting learning rate. (original ResNet paper recommends setting this to 1e-1)
USE_LBL_SMOOTH = False # determines if label smoothing is used while training
USE_COSINE = False # determines if the learning rate is to be scheduled using the cosine decay policy.
```
[**NOTE**] For the complete list of supported models, refer to the *dispatcher.py* file in the *utils* folder. This file consists a dictionary, mapping model names to the corresponding *tf.keras.Model* object.

- After setting all the necessary parameters in the configuration file, to train the model, run the following command ***from the base directory of the project***.
```
$ python train.py
```
### Note on callbacks
During training, calls to the following callbacks are made either at the end of every batch or every epoch, dependent on the particular callback.
- *LearningRateScheduler*: Schedules the learning rate as per the policy specified in the config file.
- *ModelCheckpoint*: Serializes model weights to disk after every epoch. By default, the model weights are stored in the *weights* folder.
- *TrainingMonitor*: This callback is responsible for plotting the loss and accuracies at the end of every epoch and saving the plot to disk. By default, the plots (and optionally a json file containing the model metrics) are saved to the *output* directory.
