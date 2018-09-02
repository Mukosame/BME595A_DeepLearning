import numpy as np
import torch

def sigmoid(a):
    return (1/(1+torch.exp(-a)))

class NeuralNetwork(object):
    def __init__(self, layers:list):
        # in: size of input layer
        # hi: size of hidden layers i
        # out: size of output laye
        #self.device = torch.device("cuda:1,2" if torch.cuda.is_available() else "cpu")
        self.layers = layers
        self.input = self.layers[0]
        self.output = self.layers[len(self.layers) - 1]
        self.theta = {}
        for i in range(len(self.layers) - 1):
            self.theta[str(i)] = torch.from_numpy(np.random.normal(0, 1/np.sqrt(self.layers[i]), (self.layers[i]+1, self.layers[i+1])))
            # input size: layer[i] + one bias node
            # output size: layer[i+1]
       
    def getLayer(self, layer:int):
        return self.theta[str(layer)]
    
    def forward(self, input):
        # input is 2D/3D DoubleTensor
        # output is DoubleTensor
        (height, width) = input.size()
        output = input 
        bias = torch.ones((1,width),dtype = torch.double)
        for i in range(len(self.layers) - 1):
            input_c = torch.cat((bias, output), 0)
            temp = torch.mm(torch.t(self.theta[str(i)]),input_c)
            output = sigmoid(temp)

        return output       
