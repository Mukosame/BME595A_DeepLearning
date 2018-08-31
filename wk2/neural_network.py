import numpy as np
import torch

class NeuralNetwork(object):
    def __init__(self, in, h1, h2, out):
        # in: size of input layer
        # hi: size of hidden layers i
        # out: size of output laye
        self.device = torch.device("cuda:1,2" if torch.cuda.is_available() else "cpu")
        
    def getLayer(self, layer):
        # set weights of neural network
        return 0(layer)
    
    def forward(self, input):
    # output is DoubleTensor
        return output
    
    def __call__(self):
        self.forward()
        
