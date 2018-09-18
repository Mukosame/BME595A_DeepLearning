# Homework 4

_Neural Networks - Back-propagation pass_

Xiaoyu Xiang(xiang43@purdue.edu)
--------------------
## Platform and Packages
- Python 3.6.5 
- PyTorch 0.4.0, torchvision
- Numpy

## Dataset: MNIST
To load the dataset, we use the PyTorch MNIST dataloader.

For the MNIST dataset, we separate it into 3 parts: training set(70%), validation set(15%), and test set(15%). During the training process, we expect to see the training loss and validation loss decrease over epochs.

Then we decide the batch size. Batch size decides how many samples would be sent to train. Let the whole sample number of training set be N, and batch size c, so the whole training set is divided into floor(N/c) parts.

We only shuffle the training set's data. For validation set, since the shuffle wouldn't inlfuence the output, we skip the shuffle process.

After one epoch, we will calculate the training error, which should be the average over all training examples(or the average over all batches).

## Code Implementation
### neural_network.py
In this script, we define a class NeuralNetwork, to use this class:

```python
from neural_network import NeuralNetwork
```

It takes 1 <list>input, that defines the size of input for each layer. 

```NeuralNetwork.getlayer(layer_index: int)``` can assign the weights for each layer. If not assigned, each layer will take random numbers(mean = 0, std = 1/sqrt(layer size)) as weights.

```NeuralNetwork.forward(input: FloatTensor)``` can do the forward pass. For each layer, the output ```y``` has the following relationship with the input ```x```:


```y = Wx + b```

Where the b is the bias node. 

```NeuralNetwork.backward(target: FloatTensor, loss: string)``` can take 1D or 2D FloatTensor as input, which can compute the gradient of target matrix for back-propagation pass.

```NeuralNetwork.updateParams(eta: float)``` is used for update parameters with the calculated backward values and given rate eta.

The loss function used in this update strategy is Mean Square Error Loss, or L2 Loss. Which can be expressed in the formula below.

![MSE loss](https://cdn-images-1.medium.com/max/1600/1*mlXnpXGdhMefPybSQtRmDA.png "MSE Loss's Formula")

![MSE loss](https://cdn-images-1.medium.com/max/1600/1*EqTaoCB1NmJnsRYEezSACA.png "Plot of MSE Loss(Y) v.s. Predictions(X)")

Also, the loss function can be changed and even defined by ourselves. That's what we can set through the parameter loss.

If we set it ```None```, it will apply the default loss MSE.

By setting it as ```CE```, cross-entropy loss is applied:

![Cross-Entropy loss](https://cdn-images-1.medium.com/max/1600/1*gNuP7PN6sC42vAYWvoAMMA.png "Cross Entropy Loss's Formula")

Where p is the ground truth, an q is the network output(or, prediction result).

### myimg2num.py
This class is used to turn images in mnist (handwritten numbers) into numbers. 

To achieve this, we build a 4-layer neural network. The input of this network is 28x28x1 images, and the output is the probability of 10 classes.

To speed up training, we use the NeuralNetwork API in batch mode. Let c be the batch size, so the network output would be 10 x c. The batch size only influence the backward propagation: instead of calculate the gradient one by one, we calculate the average gradient over all samples in this batch.



### nnimg2num.py
This class achieves the same goal as ```myimg2num.py```, but instead of using our homemade neural network, we apply PyTorch's nn package to build the neural network, and realize the optimization step.


## Test Result
Run ```'python test.py'```
