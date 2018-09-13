import numpy as np
import torch

class NeuralNetwork(object):
    def __init__(self, layers):
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
            self.theta[str(i)] = torch.from_numpy(np.random.normal(0, 1/np.sqrt(self.layers[i]), (self.layers[i]+1, self.layers[i+1]))).type(torch.FloatTensor)
            #self.dE_theta[str(i)] = 0 * self.theta[str(i)]
            #self.middle[str(i)] = torch.zeros((self.layers[i], 1), dtype = torch.float)
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
        bias = torch.ones((1,width),dtype = torch.float)
        self.middle[str(0)] = input
        #print(self.middle[str(0)])
        for i in range(len(self.layers) - 1):
            self.middle[str(i)] = torch.cat((bias, self.middle[str(i)]), 0)
            self.middle[str(i+1)] = torch.mm(torch.t(self.theta[str(i)]), self.middle[str(i)])
        self.prediction = sigmoid(self.middle[str(len(self.layers)-1)])
        #print(self.prediction)
        return self.prediction    
    
    def backward(self, target, lossType):
        # computer average gradient across seen samples, doesn't return anything
        if lossType == 'CE':
            # Use CE loss
            self.loss = torch.log(torch.sum(torch.exp(self.prediction))) - torch.sum(self.prediction.t()*target)
            loss_grad = self.prediction - target#- target + torch.exp(self.prediction) / torch.sum(torch.exp(self.prediction))
        else:
            # Use default MSE loss
            self.loss = torch.mean((self.prediction - target)**2)
            loss_grad = 2*(self.prediction - target)/self.output_size # MSE's gradient
            #print(self.loss.size())
            #print(self.prediction)
        loss_grad = self.prediction * (1-self.prediction) * loss_grad #sigmoid
        temp = torch.ones((self.output_size, 1), dtype = torch.float)#torch.FloatTensor([1])
        for i in reversed(range(len(self.layers) - 1)):
            #print(self.middle[str(i)], temp)
            if i != (len(self.layers)-2):
                index = torch.LongTensor([1, 2])
                temp = torch.index_select(temp, 0, index)
                #print(temp)
            self.dE_theta[str(i)] = loss_grad * torch.mm(self.middle[str(i)],  torch.t(temp))
            temp = self.theta[str(i)].mm(temp) # update for next round

    def updateParams(self, eta):
        for i in range(len(self.layers)-1):
            #print(self.theta[str(i)], eta, self.dE_theta[str(i)], self.loss)
            self.theta[str(i)] -= eta * self.dE_theta[str(i)]
