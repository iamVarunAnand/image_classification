# Image Classification using ResNets
The goal of this project is to firstly replicate the ResNet SOTA results on CIFAR10, and use several recently published *updates* to push this state of the art as high as possible. Using such updates, I was able to achieve an error rate of **6.90%** on the CIFAR10 test set, using a **20-layer** ResNet that consists of a mere **0.27M parameters**. For comparison, the original ResNet20 presented in [1], had an error rate of **8.75%**. 
|MODEL|TEST ERR|TEST ACC
| :-: | :-: | :-:
|[ResNet20](https://arxiv.org/abs/1512.03385)|8.75|91.25
|XResNet20|8.18|91.82
|MXResNet20|7.93|92.07
|SE-MXResNet20|7.81|92.19
|+ cosine decay|7.53|92.47
|+ label smoothing|7.49|92.51
|+ mixup|7.03|92.97
|+ reflection padding|**6.90**|**93.10**
- **Model Architecture updates**
	1. All the updates mentioned in the [Bag of Tricks](https://arxiv.org/abs/1812.01187) paper - *XResNet*
	2. [Mish](https://arxiv.org/abs/1908.08681) activation instead of ReLU - *MXResNet*
	3. [Squeeze-Excite (SE)](https://arxiv.org/abs/1709.01507) blocks wherever possible - *SE-MXResNet*
  
- **Updates to the Training procedure**
	1. [MixUp](https://arxiv.org/abs/1710.09412) training.
	2. [Cosine-decayed](https://arxiv.org/abs/1608.03983) learning rate schedule.
	3. Adding Label Smoothing.
	4. Using reflection padding instead of zero padding for the input images.
