# Homework 2

_Neural Networks - Feedforward Pass_

Xiaoyu Xiang(xiang43@purdue.edu)
--------------------
## Platform and Packages
- Python 3.6, PyTorch 0.4.1
- Numpy

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
### logic_gates.py

This script defines 4 classes: ```AND```, ```OR```, ```NOT```, ```XOR```.

All the classes take bool input and give bool outputs.

For ```AND```, ```OR```, ```NOT```, they are generated by 1-layer neural network. By transferring the input bool to int, we can apply the linear formula to calculate the output. Where the weights for each logic gates are updated by backward propagation strategy conducted by ```logicGates.train()```.

## Test Result
Run ```'python test.py'```

For each logic gate, the test results are listed in the tables below.

### AND

| Input 1        | Input2          | Output  |
| ------------- |:-------------:| -----:|
| True      | True | True |
| True      | False      |   False |
| False | True    |    False |
| False | False | False|

### OR

| Input 1        | Input2          | Output  |
| ------------- |:-------------:| -----:|
| True      | True | True |
| True      | False      |   True |
| False | True    |    True |
| False | False | False|

### NOT

| Input | Output |
| ------------- |:-------------:|
| True | False|
| False | True|

### XOR

| Input 1        | Input2          | Output  |
| ------------- |:-------------:| -----:|
| True      | True | False |
| True      | False      |   True |
| False | True    |    True |
| False | False | False|

Although the results are samw with HW02, the theta used in each gates are different.