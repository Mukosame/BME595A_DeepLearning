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
        # in batch mode: input size = [batch_size, input_length]
        self.input = torch.t(input) 
        height = self.input.size(0)
        width = self.input.size(1)
        #print(self.input.size())
        bias = torch.ones((1,width), dtype = torch.float)
        self.middle[str(0)] = self.input
        #print(self.middle[str(0)])
        for i in range(len(self.layers) - 1):
            self.middle[str(i)] = torch.cat((bias, self.middle[str(i)]), 0)
            self.middle[str(i+1)] = sigmoid(torch.mm(torch.t(self.theta[str(i)]), self.middle[str(i)]))
        self.prediction = self.middle[str(len(self.layers)-1)]
        #print(self.prediction)
        self.prediction = torch.t(self.prediction)
        return self.prediction    
    
    def backward(self, target, lossType):
        target = torch.t(target)
        prediction = torch.t(self.prediction)
        width = target.size(1)
        # computer average gradient across seen samples, doesn't return anything
        if lossType == 'CE':
            # Use CE loss
            self.loss = torch.log(torch.sum(torch.exp(prediction))) - torch.sum(prediction.t()*target)
            loss_grad = prediction - target#- target + torch.exp(self.prediction) / torch.sum(torch.exp(self.prediction))
        else:
            # Use default MSE loss
            self.loss = torch.mean((prediction - target)**2)
            loss_grad = 2*(prediction - target)/self.output_size # MSE's gradient
            #print(self.loss.size())
            #print(prediction)
        temp = prediction * (1-prediction) * loss_grad #sigmoid
        #print(loss_grad.size())
        #temp = torch.ones((self.output_size, width), dtype = torch.float)#torch.FloatTensor([1])
        for i in reversed(range(len(self.layers) - 1)):
            #print(self.middle[str(i)], temp)
            if i != (len(self.layers)-2):
                # ignore the first bias 
                #print(temp.size())
                temp = temp.narrow(0, 1, temp.size(0)-1)
            #print(self.middle[str(i)].size(), torch.t(temp).size())
            self.dE_theta[str(i)] = torch.mm(self.middle[str(i)],  torch.t(temp))
            temp = self.theta[str(i)].mm(temp) * self.middle[str(i)] * (1-self.middle[str(i)]) # update for next round

    def updateParams(self, eta):
        for i in range(len(self.layers)-1):
            #print(self.theta[str(i)], eta, self.dE_theta[str(i)], self.loss)
            self.theta[str(i)] -= eta * self.dE_theta[str(i)]
