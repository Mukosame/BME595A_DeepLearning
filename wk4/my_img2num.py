# perform forward and back-prop using torch.nn
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as F
from neural_network import NeuralNetwork

class NnImg2Num(object):
    def private __init__(self):
        # input: 28 x 28 = 784 vector
        self.batch = 16
        self.nni2n = NeuralNetwork([(self.batch, 784)])

    def forward(self, img):
    # input: [28 x 28 ByteTensor] img
        img_re = torch.reshape(torch.ByteTensor(img), (-1,))
        # already 28x28 for mnist, reshape to 1 x 784 vector
        return self.nni2n(img_re)

    def train(self):
        train_log = open('my_train_log.txt', 'w')
        data = 
        label = 
        learning_rate = 0.1
        max_iteration = 1000
        batch = self.batch
        data_num = 
        for epoch in range(max_iteration):
            running_loss = 0.0
            index = randperm(data_num)
            rand_data = torch.index_select(data, 0, index)
            rand_label = torch.index_select(label, 0, index)
            split = 
            for i in range(split):
                self.nni2n.forward(rand_data[i])
                self.nni2n.backward(rand_label[i], None)
                self.nni2n.updateParams(learning_rate)
                running_loss += self.nni2n.loss
            epoch_loss = float(running_loss) / split
            train_log.write(str(epoch_loss)+'\n')
            if epoch_loss < 0.01:
                print('The training ends at ' + str(epoch) + ' epochs. \n')
                break