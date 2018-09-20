# perform forward and back-prop using torch.nn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

class NnImg2Num(object):
    def __init__(self):
        # input: 28 x 28 = 784 vector
        self.batch = 16
        # 2 hidden layers: 512 and 64
        # output layer: 10
        #self.mynn = NeuralNetwork([784,512,64,10])
        self.mynn = nn.Sequential(
            nn.Linear(784, 512), nn.Sigmoid(),
            nn.Linear(512, 64), nn.Sigmoid(),
            nn.Linear(64, 10), nn.Sigmoid()
        )
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.MNIST(root='./data', train = True, download = True, transform = self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch, shuffle = True, num_workers = 2)
        self.testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size = self.batch, shuffle = False, num_workers = 2)
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        self.device = torch.device("cpu")#torch.device("cuda:1,2" if torch.cuda.is_available() else "cpu")


    def forward(self, img):
    # input: [28 x 28 ByteTensor] img
        img_re = torch.reshape(torch.ByteTensor(img), (-1,))
        # already 28x28 for mnist, reshape to 1 x 784 vector
        output = self.mynn.forward(img_re)
        _, predict_label = torch.max(output.data,1)
        return predict_label

    def train(self):
        train_log = open('./log/nn_train_log.txt', 'w')
        learning_rate = 0.1
        max_iteration = 66
        batch = self.batch # batch size
        #data_num = 60,000
        #val_num = 10,000
        epoch_loss = np.zeros(max_iteration)
        vali_loss = epoch_loss
        acc = epoch_loss
        optimizer = torch.optim.SGD(self.mynn.parameters(), lr = learning_rate)
        criterion = nn.MSELoss() # returns to the mean loss of a batch

        def onehot(abc):
            onehot_label = torch.zeros(self.batch, 10)
            for i in range(batch):
                onehot_label[i][abc[i]] = 1 
            return onehot_label

        for epoch in range(max_iteration):
            running_loss = 0.0        
            # Train and calculate training loss
            for i, data in enumerate(self.trainloader, 0):   
                # each split of batch: i is the batch's index         
                inputs, raw_labels = data            
                labels = onehot(raw_labels)            
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients        
                optimizer.zero_grad()
                this_output = self.mynn.forward(inputs)
                this_loss = criterion(this_output, labels)
                # after each batch, update weights
                this_loss.backward()
                optimizer.step()
                running_loss += this_loss.item() #/ batch
            epoch_loss[epoch] = float(running_loss) / (i+1)

            # Test loss and accuracy
            val_loss = 0.0
            ## Test the network with test data
            correct = 0
            total = 0
            for i, data in enumerate(self.testloader, 0):
                inputs, raw_labels = data
                inputs, raw_labels = inputs.to(self.device), raw_labels.to(self.device)
                labels = onehot(raw_labels) 
                this_output = self.mynn.forward(inputs)
                _, prediction = torch.max(this_output.data, 1)
                #c = (prediction == raw_labels).squeeze()
                total += raw_labels.size(0)
                correct += (prediction == raw_labels).sum().item()
                loss = criterion(this_output, labels)
                val_loss += loss.item()
            vali_loss[epoch] = float(val_loss) / (i+1)
            acc[epoch] = float(correct) / total

            train_log.write(str(epoch_loss)+'\t'+str(vali_loss)+'\n')

            # end the epoch when loss is small enough
            if epoch_loss[epoch] < 0.01:
                print('The training ends at ' + str(epoch) + ' epochs. \n')
                break

        # plot loss vs epoch
        x = range(epoch+1)
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure()
        plt.plot(x, epoch_loss[0:epoch], 'bo-', label='Training Loss')
        plt.plot(x,vali_loss[0:epoch], 'ro-', label = 'Validation Loss')   
        plt.xlabel('Epoch')    
        plt.ylabel('Loss')
        fig.savefig('ptnn_loss.jpg', dpi = fig.dpi)

        # plot accuracy vs epoch
        plt.style.use('seaborn-whitegrid')
        fig2 = plt.figure()
        plt.plot(x, acc[0:epoch], 'bo-', label='Test Accuracy')
        plt.xlabel('Epoch')    
        plt.ylabel('Test Accuracy')
        fig.savefig('ptnn_acc.jpg', dpi = fig2.dpi)
