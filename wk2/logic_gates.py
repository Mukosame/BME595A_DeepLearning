import numpy as np
import torch
from neural_network import NeuralNetwork

class AND():
    def __init__(self):
        self.and_nn = NeuralNetwork([2,1])
        self.layer0 = self.and_nn.getLayer(0)
        self.layer0.fill_(0)
        # update weight
        self.layer0  += torch.DoubleTensor([[-10],[7],[7]])
        
    def forward(self):
        return self.and_nn.forward(torch.DoubleTensor([[self.a], [self.b]]))
        
    def __call__(self, a: bool, b: bool):
        self.a = a
        self.b = b
        output = self.forward()
        return bool(np.around(output.numpy()))

class OR():
    def __init__(self):
        self.ornn = NeuralNetwork([2,1])
        self.layer0 = self.ornn.getLayer(0)
        self.layer0.fill_(0)
        self.layer0 += torch.DoubleTensor([[-1],[7],[7]])

    def forward(self):
        return self.ornn.forward(torch.DoubleTensor([[self.a], [self.b]]))

    def __call__(self, a:bool, b:bool):
        self.a = a
        self.b = b
        output = self.forward()
        return bool(np.around(output.numpy()))

class NOT():
    def __init__(self):
        self.notnn = NeuralNetwork([1,1])
        self.layer0 = self.notnn.getLayer(0)
        self.layer0.fill_(0)
        self.layer0 += torch.DoubleTensor([[5],[-7]])

    def forward(self):
        return self.notnn.forward(torch.DoubleTensor([[self.a]]))

    def __call__(self, a:bool):
        self.a = a
        output = self.forward()
        return bool(np.around(output.numpy()))

class XOR():
    def __init__(self):
        self.xornn = NeuralNetwork([2,2,1])
        self.layer10 = self.xornn.getLayer(0)
        self.layer10.fill_(0)
        self.layer10 += torch.DoubleTensor([[-5,-5],[6,-6],[-6,6]])
        self.layer20 = self.xornn.getLayer(1)
        self.layer20.fill_(0)
        self.layer20 += torch.DoubleTensor([[-5],[7],[7]])

    def forward(self):
        return self.xornn.forward(torch.DoubleTensor([[self.a], [self.b]]))

    def __call__(self, a:bool, b:bool):
        self.a = a
        self.b = b
        output = self.forward()
        return bool(np.around(output.numpy()))
