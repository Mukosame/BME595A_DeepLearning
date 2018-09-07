import numpy as np
import torch

class NeuralNetwork(object):
    def __init__(self, layers:list):
        # in: size of input layer
        # hi: size of hidden layers i
        # out: size of output laye
        #self.device = torch.device("cuda:1,2" if torch.cuda.is_available() else "cpu")
        self.layers = layers
        self.input_size = self.layers[0]
        self.output_size = self.layers[len(self.layers) - 1] #output size
        self.theta = {} 
        self.dE_theta = {} #should have the similar structure as theta
        self.middle = {}
        for i in range(len(self.layers) - 1):
            self.theta[str(i)] = torch.from_numpy(np.random.normal(0, 1/np.sqrt(self.layers[i]), (self.layers[i]+1, self.layers[i+1])))
            self.dE_theta[str(i)] = 0 * self.theta[str(i)]
            self.middle[str(i)] = torch.zeros((self.layers[i], 1), dtype = torch.float)
            # input size: layer[i] + one bias node
            # output size: layer[i+1]
        self.loss = 0
        

    def getLayer(self, layer:int):
        return self.theta[str(layer)]
    
    def forward(self, input):
        def sigmoid(a):     
            return (1/(1+torch.exp(-a)))
        # input is 2D/3D FloatTensor
        # output is FloatTensor
        self.input = input
        (height, width) = input.size()
        self.middle[str(0)] = input 
        bias = torch.ones((1,width),dtype = torch.float)
        for i in range(len(self.layers) - 1):
            input_c = torch.cat((bias, self.middle[i]), 0)
            self.middle[i+1] = torch.mm(torch.t(self.theta[str(i)]),input_c)
        output = sigmoid(self.middle[len(self.layers)])
        self.prediction = output
        return self.prediction    
    
    def backward(self, target, lossType):
        # computer average gradient across seen samples, doesn't return anything
        if lossType == 'CE':
            # Use CE loss
            self.loss = - target / self.prediction # Loss = np.sum(-target * np.log(self.output))
        else:
            # Use default MSE loss
            self.loss = 2*(self.prediction - target)/self.output_size # MSE's gradient
        self.loss = self.loss * np.dot(self.prediction, (1-self.prediction)) #sigmoid
        temp = 1#torch.ones((self.output_size, 1), dtype = torch.float)
        for i in reversed(range(len(self.layers) - 1)):
            self.dE_theta[str(i)] = self.middle[str(i)] * temp
            temp = self.theta[str(i)] * temp

    def updateParams(self, eta):
        self.theta -= eta * self.dE_theta * self.loss