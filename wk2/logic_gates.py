import numpy as np
import torch
from neural_network import NeuralNetwork

class AND(boolean):
    def __init__(self):
        self.and_nn = NeuralNetwork([2,1])
        self.layer0 = self.and_nn.getLayer(0)
        self.layer0.fill_(0)
        # update weight
        self.layer0  = self.layer0 + torch.DoubleTensor([-1],[0.7],[0.7])
        
    def forward(self):
        a_tensor = torch.DoubleTensor(self.a)
        b_tensor = torch.DoubleTensor(self.b)
        return self.and_nn.forward(a_tensor, b_tensor)
        
    def __call__(self, a: bool, b: bool):
        self.a = a
        self.b = b
        output = self.forward()
        return bool(output)

class OR(boolean):
    def __init__(self):
        self.ornn = NeuralNetwork([2,1])
        self.layer0 = self.ornn.getLayer(0)
        self.layer0.fill_(0)
        self.layer0 += torch.DoubleTensor([1],[7],[7])

    def forward(self):
        a_tensor = torch.DoubleTensor(self.a)
        b_tensor = torch.DoubleTensor(self.b)
        return self.ornn.forward(a_tensor, b_tensor)

    def __call__(self, a:bool, b:bool):
        self.a = a
        self.b = b
        output = self.forward()
        return bool(output)

class NOT(boolean):
    def __init__(self):
        self.notnn = NeuralNetwork([1,1])
        self.layer0 = self.notnn.getLayer(0)
        self.layer0.fill_(0)
        self.layer0 += torch.DoubleTensor([5],[-7])

    def forward(self):
        a_tensor = torch.DoubleTensor(self.a)
        return self.notnn.forward(a_tensor)

    def __call__(self, a:bool):
        self.a = a
        output = self.forward()
        return bool(output)

class XOR(boolean):
    def __init__(self):
        self.xornn = NeuralNetwork([2,1])
        self.layer10 = self.xornn.getLayer(0)
        self.layer10.fill_(0)
        self.layer10 += torch.DoubleTensor([-5,-5],[6,-6],[-6,6])
        self.layer20 = self.xornn.getLayer(1)
        self.layer20.fill_(0)
        self.layer20 += torch.DoubleTensor([-5],[6],[6])

    def forward(self):
        a_tensor = torch.DoubleTensor(self.a)
        b_tensor = torch.DoubleTensor(self.b)
        return self.xornn1.forward(a_tensor, b_tensor)

    def __call__(self, a:bool, b:bool):
        self.a = a
        self.b = b
        output = self.forward()
        return bool(output)
