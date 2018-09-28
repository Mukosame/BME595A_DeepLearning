# Homework 5

_Neural Networks - CNN in PyTorch_

Xiaoyu Xiang(xiang43@purdue.edu)
--------------------
## Platform and Packages
- Python 3.6.5 
- PyTorch 0.4.0, torchvision
- Numpy, matplotlib, OpenCV 3

## Dataset: MNIST
To load the dataset, we use the PyTorch MNIST dataloader.

For the MNIST dataset, we separate it into 2 parts: training set(60,000 images), and and test set(10,000 images). During the training process, we expect to see the training loss and validation loss decrease over epochs.

Then we decide the batch size. Batch size decides how many samples would be sent to train. Let the whole sample number of training set be N, and batch size c, so the whole training set is divided into floor(N/c) parts.

We only shuffle the training set's data. For validation set, since the shuffle wouldn't inlfuence the output, we skip the shuffle process.

After one epoch, we will calculate the training error, which should be the average over all training examples(or the average over all batches).

## Dataset: CIFAR100
To load the dataset, we use the PyTorch CIFAR100 dataloader. To show the prediction result, we define a list of class names. The other settings for loading this dataset is are similar to MNIST dataset.

## Code Implementation

### img2num.py

```python
from img2num import Img2Num
```

This class can turn binary hand-written number's image into number. It applies PyTorch's nn package to build the neural network, and realize the optimization step.

We set the following parameters for training this neural network:

- Epoch Number: 20
- Loss Function: MSE
- Learning Rate: 0.1
- Batch Size: 16
- Optimizer: SGD
- Image Normalization: True
- Trained on: single GPU (Nvidia Quadro M6000)

To init the network: ```net = Img2Num()```

Traing this network: ```net.train()```

Get number output from one input image: ```net.(image)```

### img2obj.py

```python
from img2num import Img2Obj
```
This class can turn input image into its type name (the type must belongs to the 100 classes of CIFAR100).

We set the following parameters for training this neural network:

- Epoch Number: 30
- Loss Function: Cross Entropy Loss
- Learning Rate: 1(epoch 0~9), 0.1(epoch 10~14), 0.01(epoch 15~19), 0.001(epoch 20~24), 0.0001(epoch 25~29)
- Batch Size: 128
- Optimizer: Adadelta
- Image Normalization: True
- Trained on: single GPU (Nvidia Quadro M6000)

To init the network: ```net = Img2Obj()```

Traing this network: ```net.train()```

Get number output from one input image: ```net(image)```

## Test Result

Run ```python test.py```, and we get the following results:

### MNIST
####Training and Validation Loss

The training and validation loss over epochs of img2num(MNIST) are shown below:

![My loss](https://github.com/Mukosame/BME595A_DeepLearning/blob/master/wk5/mnist_loss.jpg "MNIST's loss")

The same loss of HW4's fully connected network is shown below:

![hw4 loss](https://github.com/Mukosame/BME595A_DeepLearning/blob/master/wk4/ptnn_loss.jpg "HW4's loss")

#### Test Accuracy

Test accuracy can also reflect the performance of network, which is calculated by: ```accuracy = correct_num / total_num```

The test accuracy over epochs of img2num(MNIST) are shown below:

![My acc](https://github.com/Mukosame/BME595A_DeepLearning/blob/master/wk5/mnist_acc.jpg "MNIST's Accuracy")

While the test accuracy of HW4 is shown below:

![My acc](https://github.com/Mukosame/BME595A_DeepLearning/blob/master/wk4/ptnn_acc.jpg "HW4's Accuracy")

#### Training Time

I plot the training time for HW4 and HW5 on each epoch below. First is HW5's result.

![My time](https://github.com/Mukosame/BME595A_DeepLearning/blob/master/wk5/mnist_time.jpg "MNIST's Training Time")

The next plot is fully connected network in HW4:

![Torch time](https://github.com/Mukosame/BME595A_DeepLearning/blob/master/wk4/ptnn_time.jpg "HW4 fully connected network's Training Time")

Since this homework is trained on GPU, it's hard to make a conclusion that this week's network is slower/faster than last week's.

#### Overall Comparision

To better compare the performance of our network and PyTorch's network, I made the following table that inludes our interested metrics:

| Metric        | HW5           | HW4  |
| ------------- |:-------------:| -----:|
| Lowest Training Loss      | 0.0015687 | 0.0163000 |
| Lowest Validation Loss      | 0.001995 | 0.0154754 |
| Highest Test Accuracy      |   0.9916   |  0.9094  |
| Avarage Training Time / s |    12.8721   |  14.6034   |

So we can make a safe conclusion based on the table above: LeNet5 outperforms the fully connected network on this task.

###CIFAR100
####Training and Validation Loss

The training and validation loss over epochs of img2obj(CIFAR100) are shown below:

![My loss](https://github.com/Mukosame/BME595A_DeepLearning/blob/master/wk5/cifar10_loss.jpg "CIFAR100's loss")

#### Test Accuracy

Test accuracy can also reflect the performance of network, which is calculated by: ```accuracy = correct_num / total_num```

The test accuracy over epochs of img2obj(CIFAR100) are shown below:

![My acc](https://github.com/Mukosame/BME595A_DeepLearning/blob/master/wk5/cifar10_acc.jpg "CIFAR100's Accuracy")

#### Training Time
The training time decreases when we set the batch size very large. The training time over each epoch is plotted below:

![My time](https://github.com/Mukosame/BME595A_DeepLearning/blob/master/wk5/cifar10_time.jpg "CIFAR100's Training Time")

After training, we select the model with highest accuracy for visualization:

- At which epoch: 29
- Training Loss: 2.1595
- Validation Loss: 3.0493
- Test Accuracy: 28.95%

When given an image(encoded in RGB and turned into numpy array), the output of ```net.view(np_img)``` is shown as below:

![My time](https://github.com/Mukosame/BME595A_DeepLearning/blob/master/wk5/view_result.jpg "function view's output")

Since I don't have a webcamera, I cannot show the result related to it. Generally, the idea is to get the image from webcam and send it to ```net.view```. If the image is loaded correctly, then the result should be the same as above.

## Summary
- CIFAR100 is harder to train than MNIST
- For optimizer: We tried SGD, Adadelta todo. According to [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html#adadelta), Adadelta has the fastest converging speed. And my result did show so. But the problem is, the accuracy just stays ~22% after around 5 epochs. No matter how I tried, like decrease the learning rate when the training loss is almost stable, etc, it just doesn't help. So I tried SGD later and divided the learning rate by 10 every 10 epochs. It takes longer time to achieve the similar training loss as Adadelta, but the performance is still not bad. And since all optimizers seem to stay stable after 20 epochs, it doesn't matter so much to chhose which optimizer if our total epoch number is around 20.
- The loss is still decreasing slowly, which proves the network is truly learning. But due to the limit of time, we stop at epoch 30. It should get better results if give more epochs.
- For batch size: We tried 16, 32, 64, 132, 256 and 128. It turns out that the best performance for SGD occurs when batch size is around 128.
- The Nvidia GPU's default fan speed setting is 20%, while the working power setting is adaptive, which would cause the GPU's temperature very, very high when training for long time, and it may cause suddenly shut down. So we should take some methods, such as increasing fan speed or limit GPU power to make sure it won't overheat.

