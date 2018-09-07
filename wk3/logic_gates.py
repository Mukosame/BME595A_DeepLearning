import numpy as np
import torch
from neural_network import NeuralNetwork

class AND():
    def __init__(self):
        self.and_nn = NeuralNetwork([2,1])
        self.layer0 = self.and_nn.getLayer(0)
        self.layer0.fill_(0)
        # update weight
        #self.layer0  = torch.FloatTensor([[-10],[7],[7]])
        
    def forward(self):
        return self.and_nn.forward(torch.FloatTensor([[self.a], [self.b]]))
        
    def __call__(self, a: bool, b: bool):
        self.a = a
        self.b = b
        outp = self.forward()
        self.output = bool(np.around(outp.numpy()))
        return self.output

    def train(self):
        data = [[True, True], [True, False], [False, True], [False, False]]
        label = [True, False, False, False]
        eta = 0.1 #learning rate
        for epoch in range(2): #loop over the dataset a few times
            running_loss = 0.0
            for i in range(4):
                self.and_nn.forward(torch.FloatTensor(data[i]))        
                self.and_nn.backward(label[i], None)
                self.and_nn.updateParams(eta)
                running_loss += self.and_nn.loss
                print(running_loss)
        print(self.layer0)

class OR():
    def __init__(self):
        self.ornn = NeuralNetwork([2,1])
        self.layer0 = self.ornn.getLayer(0)
        self.layer0.fill_(0)
        #self.layer0 += torch.FloatTensor([[-1],[7],[7]])

    def forward(self):
        return self.ornn.forward(torch.FloatTensor([[self.a], [self.b]]))

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
        #self.layer0 += torch.FloatTensor([[5],[-7]])

    def forward(self):
        return self.notnn.forward(torch.FloatTensor([[self.a]]))

    def __call__(self, a:bool):
        self.a = a
        output = self.forward()
        return bool(np.around(output.numpy()))

class XOR():
    def __init__(self):
        self.xornn = NeuralNetwork([2,2,1])
        self.layer10 = self.xornn.getLayer(0)
        self.layer10.fill_(0)
        #self.layer10 += torch.FloatTensor([[-5,-5],[6,-6],[-6,6]])
        self.layer20 = self.xornn.getLayer(1)
        self.layer20.fill_(0)
        #self.layer20 += torch.FloatTensor([[-5],[7],[7]])

    def forward(self):
        return self.xornn.forward(torch.FloatTensor([[self.a], [self.b]]))

    def __call__(self, a:bool, b:bool):
        self.a = a
        self.b = b
        output = self.forward()
        return bool(np.around(output.numpy()))
