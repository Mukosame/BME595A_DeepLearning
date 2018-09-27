# perform forward and back-prop using torch.nn
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

class LeNet5(nn.module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding = 0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding = 0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, img):
        self.img_size = 32*32
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

class Img2Obj(object):
    def __init__(self):
        # input: [32 x 32] image
        self.img_size = 32*32
        self.batch = 16
        self.device = torch.device("cuda:1,2" if torch.cuda.is_available() else "cpu")

        self.mynn = LeNet5().to(self.device)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch, shuffle = True, num_workers = 2)
        testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size = self.batch, shuffle = False, num_workers = 2)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def view(self, img):
        def imshow(img):
            img = img / 2 + 0.5 #unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

        imshow(torchvision.utils.make_grid(img))
        img = img.to(device)
        outputs = net(img)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' % classes[predicted]) 

    def cam (self):
        # load image from webcam and classify it
        img = cv2.
        view(img)
        return self.forward(img)

    def forward(self, img):
    # input: [32 x 32 ByteTensor] img
        = self.mynn
        return predict_label

    def train(self):
        train_log = open('./log/cifar10_train_log.txt', 'w')
        learning_rate = 0.1
        max_iteration = 20
        batch = self.batch # batch size
        #data_num = 60,000
        #val_num = 10,000
        epoch_loss = np.zeros(max_iteration)
        vali_loss = np.zeros(max_iteration)
        acc = np.zeros(max_iteration)
        train_time = list()
        optimizer = torch.optim.SGD(self.mynn.parameters(), lr = learning_rate)
        criterion = nn.CrossEntropyLoss() # returns to the mean loss of a batch

        def onehot(abc):
            onehot_label = torch.zeros(self.batch, 10)
            for i in range(batch):
                onehot_label[i][abc[i]] = 1 
            return onehot_label

        for epoch in range(max_iteration):
            start_time = time.time()
            running_loss = 0.0        
            # Train and calculate training loss
            for i, data in enumerate(self.trainloader, 0):   
                # each split of batch: i is the batch's index         
                inputs, labels = data            
                #labels = onehot(raw_labels)            
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients        
                optimizer.zero_grad()
                this_output = self.mynn(inputs)
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
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                #labels = onehot(raw_labels) 
                this_output = self.mynn.forward(inputs.view(batch, self.img_size))
                _, prediction = torch.max(this_output.data, 1)
                #c = (prediction == raw_labels).squeeze()
                total += labels.size(0)
                correct += (prediction == labels).sum().item()
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
        fig1 = plt.figure(1)
        plt.style.use('seaborn-whitegrid')
        plt.plot(x, epoch_loss[0:epoch+1], 'bo-', label='Training Loss')
        plt.plot(x,vali_loss[0:epoch+1], 'ro-', label = 'Validation Loss')   
        plt.xlabel('Epoch')    
        plt.ylabel('Loss')
        plt.legend()
        fig1.savefig('cifar10_loss.jpg', dpi = fig1.dpi)

        # plot accuracy vs epoch
        fig2 = plt.figure(2)
        plt.style.use('seaborn-whitegrid')
        plt.plot(x, acc[0:epoch+1], 'bo-', label='Test Accuracy')
        plt.xlabel('Epoch')    
        plt.ylabel('Test Accuracy')
        plt.legend()
        fig2.savefig('cifar10_acc.jpg', dpi = fig2.dpi)

        # plot train time vs epoch
        fig3 = plt.figure(3)
        plt.style.use('seaborn-whitegrid')
        plt.plot(x, train_time, 'bo-', label='Training Time')
        plt.xlabel('Epoch')    
        plt.ylabel('Training Time')
        plt.legend()
        fig3.savefig('cifar10_time.jpg', dpi = fig3.dpi)# plot loss vs epoch