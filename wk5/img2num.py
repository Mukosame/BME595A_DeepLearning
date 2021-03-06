# perform forward and back-prop using torch.nn
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding = 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding = 0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, img):
        self.img_size = 28*28
        # input: [28 x 28 ByteTensor] img
        x = img#torch.view(torch.ByteTensor(img), (-1,self.img_size))
        # already 28x28 for mnist, reshape to 1 x 784 vector
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Img2Num(object):
    def __init__(self):
        # input: 28 x 28 = 784 vector
        self.img_size = 784
        self.batch = 16
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # output layer: 10
        self.net = LeNet5().to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.MNIST(root='./data', train = True, download = True, transform = self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch, shuffle = True, num_workers = 2)
        self.testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size = self.batch, shuffle = False, num_workers = 2)
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


    def forward(self, img):
    # input: [28 x 28 ByteTensor] img
        #x = torch.view(torch.ByteTensor(img), (-1,self.img_size))
        # already 28x28 for mnist, reshape to 1 x 784 vector
        output = self.net(x)
        _, predict_label = torch.max(output.data,1)
        return predict_label

    def train(self):
        train_log = open('./log/mnist_train_log.txt', 'w')
        learning_rate = 0.1
        max_iteration = 20
        batch = self.batch # batch size
        #data_num = 60,000
        #val_num = 10,000
        epoch_loss = np.zeros(max_iteration)
        vali_loss = np.zeros(max_iteration)
        acc = np.zeros(max_iteration)
        train_time = list()
        optimizer = torch.optim.SGD(self.net.parameters(), lr = learning_rate)
        criterion = nn.MSELoss() # returns to the mean loss of a batch

        def onehot(abc):
            onehot_label = torch.zeros(self.batch, 10)
            for i in range(batch):
                onehot_label[i][abc[i]] = 1 
            return onehot_label.to(self.device)

        for epoch in range(max_iteration):
            start_time = time.time()
            running_loss = 0.0        
            # Train and calculate training loss
            for i, data in enumerate(self.trainloader, 0):   
                # each split of batch: i is the batch's index         
                inputs, raw_labels = data            
                labels = onehot(raw_labels)            
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients        
                optimizer.zero_grad()
                this_output = self.net(inputs)
                #print(this_output.type(), labels.type())
                this_loss = criterion(this_output, labels)
                # after each batch, update weights
                this_loss.backward()
                optimizer.step()
                running_loss += this_loss.item() #/ batch
            epoch_loss[epoch] = float(running_loss) / (i+1)
            time_spend = time.time() - start_time
            train_time.append(time_spend)
            # Test loss and accuracy
            val_loss = 0.0
            ## Test the network with test data
            correct = 0
            total = 0
            for i, data in enumerate(self.testloader, 0):
                inputs, raw_labels = data
                inputs, raw_labels = inputs.to(self.device), raw_labels.to(self.device)                 
                labels = onehot(raw_labels) 
                #inputs, labels = inputs.to(self.device), labels.to(self.device)
                this_output = self.net(inputs)
                _, prediction = torch.max(this_output.data, 1)
                #print(prediction.type())
                #c = (prediction == raw_labels).squeeze()
                total += raw_labels.size(0)
                correct += (prediction == raw_labels).sum().item()
                loss = criterion(this_output, labels)
                val_loss += loss.item()
            vali_loss[epoch] = float(val_loss) / (i+1)
            acc[epoch] = float(correct) / total

            train_log.write(str(time_spend) + '\t' + str(epoch_loss[epoch])+'\t'+str(vali_loss[epoch])+ '\t' + str(acc[epoch]) + '\n')
            print(' Epoch {}: Training time={:.2f}s Training Loss={:.4f} Val Loss={:.4f} Acc={:.4f} \n'.format(epoch, time_spend, epoch_loss[epoch], vali_loss[epoch], acc[epoch]))

            # end the epoch when loss is small enough
            #if epoch_loss[epoch] < 0.01:
            #    print('The training ends at ' + str(epoch) + ' epochs. \n')
            #    break

        print('Average training time = '+ str(np.mean(train_time)) + '\n')
        # plot loss vs epoch
        x = np.arange(1,epoch+2)
        fig4 = plt.figure(4)
        plt.style.use('seaborn-whitegrid')
        plt.plot(x, epoch_loss[0:epoch+1], 'bo-', label='Training Loss')
        plt.plot(x,vali_loss[0:epoch+1], 'ro-', label = 'Validation Loss')   
        plt.xlabel('Epoch')    
        plt.ylabel('Loss')
        plt.legend()
        fig4.savefig('mnist_loss.jpg', dpi = fig4.dpi)

        # plot accuracy vs epoch
        fig5 = plt.figure(5)
        plt.style.use('seaborn-whitegrid')
        plt.plot(x, acc[0:epoch+1], 'bo-', label='Test Accuracy')
        plt.xlabel('Epoch')    
        plt.ylabel('Test Accuracy')
        plt.legend()
        fig5.savefig('mnist_acc.jpg', dpi = fig5.dpi)

        # plot train time vs epoch
        fig6 = plt.figure(6)
        plt.style.use('seaborn-whitegrid')
        plt.plot(x, train_time, 'bo-', label='Training Time')
        plt.xlabel('Epoch')    
        plt.ylabel('Training Time')
        plt.legend()
        fig6.savefig('mnist_time.jpg', dpi = fig6.dpi)# plot loss vs epoch
