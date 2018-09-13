import numpy as np
import torch
from neural_network import NeuralNetwork

class AND():
    def __init__(self):
        self.and_nn = NeuralNetwork([2,1])
        self.layer0 = self.and_nn.getLayer(0)
        #self.layer0.fill_(0)
        #self.layer0  += torch.FloatTensor([[-10],[7],[7]])
        
    def forward(self):
        return self.and_nn.forward(torch.FloatTensor([[self.a], [self.b]]))
        
    def __call__(self, a: bool, b: bool):
        self.a = a
        self.b = b
        outp = self.forward()
        self.output = bool(np.around(outp.numpy()))
        return self.output

    def train(self):
        train_log = open('and_train_log.txt', 'w')
        data = torch.FloatTensor([[[True], [True]], [[True], [False]], [[False], [True]], [[False], [False]]])
        label = torch.FloatTensor([True, False, False, False])
        eta = 0.1 #learning rate
        for epoch in range(100): #loop over the dataset a few times: max iteration times
            running_loss = 0.0
            #epoch_loss = 100
            index = torch.randperm(4)
            rand_data = torch.index_select(data, 0, index)
            rand_label = torch.index_select(label,0,index)
            for i in range(4):
                self.and_nn.forward(rand_data[i])        
                self.and_nn.backward(rand_label[i], None)
                #print(rand_data[i], rand_label[i])
                self.and_nn.updateParams(eta)
                running_loss += self.and_nn.loss
            epoch_loss = float(running_loss)/4
            #print(epoch_loss)
            train_log.write(str(epoch_loss)+'\n')
            if (epoch_loss < 0.01):
                break
        print(self.layer0)

class OR():
    def __init__(self):
        self.ornn = NeuralNetwork([2,1])
        self.layer0 = self.ornn.getLayer(0)
        #self.layer0.fill_(0)
        #self.layer0 += torch.FloatTensor([[-1],[7],[7]])

    def forward(self):
        return self.ornn.forward(torch.FloatTensor([[self.a], [self.b]]))

    def __call__(self, a:bool, b:bool):
        self.a = a
        self.b = b
        output = self.forward()
        return bool(np.around(output.numpy()))

    def train(self):
        train_log = open('or_train_log.txt', 'w')
        data = torch.FloatTensor([[[True], [True]], [[True], [False]], [[False], [True]], [[False], [False]]])
        label = torch.FloatTensor([True, True, True, False])
        eta = 0.1 #learning rate
        for epoch in range(100): #loop over the dataset a few times: max iteration times
            running_loss = 0.0
            #epoch_loss = 100
            index = torch.randperm(4)
            rand_data = torch.index_select(data, 0, index)
            rand_label = torch.index_select(label,0,index)
            for i in range(4):
                self.ornn.forward(rand_data[i])        
                self.ornn.backward(rand_label[i], None)
                #print(rand_data[i], rand_label[i])
                self.ornn.updateParams(eta)
                running_loss += self.ornn.loss
            epoch_loss = float(running_loss)/4
            #print(epoch_loss)
            train_log.write(str(epoch_loss)+'\n')
            if (epoch_loss < 0.01):
                break
        print(self.layer0)

class NOT():
    def __init__(self):
        self.notnn = NeuralNetwork([1,1])
        self.layer0 = self.notnn.getLayer(0)
        #self.layer0.fill_(0)
        #self.layer0 += torch.FloatTensor([[5],[-7]])

    def forward(self):
        return self.notnn.forward(torch.FloatTensor([[self.a]]))

    def __call__(self, a:bool):
        self.a = a
        output = self.forward()
        return bool(np.around(output.numpy()))

    def train(self):
        train_log = open('not_train_log.txt', 'w')
        data = torch.FloatTensor([[[True]], [[False]]])
        label = torch.FloatTensor([False, True])
        eta = 0.1 #learning rate
        for epoch in range(100): #loop over the dataset a few times: max iteration times
            running_loss = 0.0
            #epoch_loss = 100
            index = torch.randperm(2)
            rand_data = torch.index_select(data, 0, index)
            rand_label = torch.index_select(label,0,index)
            for i in range(2):
                self.notnn.forward(rand_data[i])        
                self.notnn.backward(rand_label[i], None)
                #print(rand_data[i], rand_label[i])
                self.notnn.updateParams(eta)
                running_loss += self.notnn.loss
            epoch_loss = float(running_loss)/4
            #print(epoch_loss)
            train_log.write(str(epoch_loss)+'\n')
            if (epoch_loss < 0.01):
                break
        print(self.layer0)

class XOR():
    def __init__(self):
        self.xornn = NeuralNetwork([2,2,1])
        self.layer10 = self.xornn.getLayer(0)
        #self.layer10.fill_(0)
        #self.layer10 += torch.FloatTensor([[-0.1, -0.1],[0.6,-0.6],[-0.6,0.6]])
        self.layer20 = self.xornn.getLayer(1)
        #self.layer20.fill_(0)
        #self.layer20 += torch.FloatTensor([[-1],[7],[7]])

    def forward(self):
        return self.xornn.forward(torch.FloatTensor([[self.a], [self.b]]))

    def __call__(self, a:bool, b:bool):
        self.a = a
        self.b = b
        output = self.forward()
        return bool(np.around(output.numpy()))

    def train(self):
        train_log = open('xor_train_log.txt', 'w')
        data = torch.FloatTensor([[[True], [True]], [[True], [False]], [[False], [True]], [[False], [False]]])
        label = torch.FloatTensor([False, True, True, False])
        eta = 3 #learning rate
        for epoch in range(200): #loop over the dataset a few times: max iteration times
            running_loss = 0.0
            #epoch_loss = 100
            index = torch.randperm(4)
            rand_data = torch.index_select(data, 0, index)
            rand_label = torch.index_select(label,0,index)
            for i in range(4):
                self.xornn.forward(rand_data[i])        
                self.xornn.backward(rand_label[i], None)
                #print(rand_data[i], rand_label[i])
                self.xornn.updateParams(eta)
                running_loss += self.xornn.loss
            epoch_loss = float(running_loss)/4
            #print(epoch_loss)
            train_log.write(str(epoch_loss)+'\n')
            if (epoch_loss < 0.01):
                break
        print(self.layer10)
        print(self.layer20)