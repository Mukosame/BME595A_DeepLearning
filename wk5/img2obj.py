# perform forward and back-prop using torch.nn
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding = 0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding = 0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

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

class Img2Obj:
    def __init__(self):
        #super(Img2Obj, self).__init__()
        # input: [32 x 32] image
        self.img_size = 32*32
        self.batch = 128
        self.device = torch.device("cuda:1,2" if torch.cuda.is_available() else "cpu")

        self.mynn = LeNet5().to(self.device)
        self.net_name = 'cifar100_lenet5'
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.CIFAR100(root='./data', train = True, download = True, transform = self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch, shuffle = True, num_workers = 2)
        self.testset = torchvision.datasets.CIFAR100(root = './data', train = False, download = True, transform = self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size = self.batch, shuffle = False, num_workers = 2)
        self.classes = ('apples', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
                        'bottles', 'bowls', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'cans', 'castle',
                        'caterpillar',
                        'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'kangaroo', 'couch',
                        'crocodile', 'cups', 'crab', 'dinosaur', 'elephant', 'dolphin', 'flatfish', 'forest', 'girl',
                        'fox', 'hamster', 'house', 'computer keyboard', 'lamp', 'lawn-mower', 'leopard', 'lion',
                        'lizard',
                        'lobster', 'man', 'maple', 'motorcycle', 'mountain', 'mouse', 'mushrooms', 'oak', 'oranges',
                        'orchids', 'otter', 'palm', 'pears', 'pickup truck', 'pine', 'plain', 'plates', 'poppies',
                        'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'roses', 'sea', 'seal',
                        'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                        'sunflowers', 'sweet peppers', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                        'train', 'trout', 'tulips', 'turtle', 'wardrobe', 'whale', 'willow', 'wolf', 'woman', 'worm'
                        )
        self.load_flag = False

    def view(self, img):
        if self.load_flag == False:        
            self.mynn.load_state_dict(torch.load('model/cifar100_lenet5_29.pth'))
            self.load_flag = True

        def imshow(npimg, text):
            #img = img / 2 + 0.5 #unnormalize
            #npimg = img.numpy()
            #toshow = np.transpose(npimg, (1, 2, 0))
            img = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)
            cv2.namedWindow(text, cv2.WINDOW_NORMAL)        
            cv2.resizeWindow(text, 256, 256)          
            cv2.imshow(text,img)  
            cv2.waitKey(0)  
            cv2.destroyAllWindows()  

        imgy = cv2.resize(img, (32, 32))
        imgy = self.transform(imgy)
        imgy = imgy.to(self.device)
        #print(imgy)
        imgy = torch.unsqueeze(imgy, 0)
        predicted = self.forward(imgy)
        imshow(img, predicted)

    def cam(self, idx = 0):
        if self.load_flag == False:        
            self.mynn.load_state_dict(torch.load('model/cifar100_lenet5_29.pth'))        
            self.load_flag = True
        # load image from webcam and classify it
        cam_img = cv2.VideoCapture(idx)
        while True:
            yes, frame = cam_img.read()
            if yes:
                #img = cv2.resize(frame, (32, 32))
                #img = self.transform(img)
                self.view(frame)
            else:
                print('\n Cannot reading from webcam')
                break

            if (cv2.waitKey(1)):
                break

        cam_img.release()
        cv2.destroyAllWindows()        

    def forward(self, img):
    # input: [32 x 32 ByteTensor] img
        outputs = self.mynn.forward(img)
        _, predicted = torch.max(outputs, 1)
        return self.classes[predicted]

    def train(self):
        train_log = open('./log/cifar10_train_log.txt', 'w')
        learning_rate = 1
        max_iteration = 30
        #batch = self.batch # batch size

        epoch_loss = np.zeros(max_iteration)
        vali_loss = np.zeros(max_iteration)
        acc = np.zeros(max_iteration)
        train_time = list()
        optimizer = torch.optim.Adadelta(self.mynn.parameters(), lr = learning_rate, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()#nn.MSELoss() # returns to the mean loss of a batch

        '''
        def onehot(abc):
            onehot_label = torch.zeros(self.batch, 10)
            for i in range(batch):
                onehot_label[i][abc[i]] = 1 
            return onehot_label
        '''
        def save_model(model,filename):
            state = model.state_dict()
            for key in state: state[key] = state[key].clone().cpu()    
            torch.save(state, filename)

        # self.mynn.load_state_dict(torch.load('cifar100_lenet5_0.pth'))
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
                this_output = self.mynn(inputs)
                _, prediction = torch.max(this_output.data, 1)
                #c = (prediction == raw_labels).squeeze()
                total += labels.size(0)
                correct += (prediction == labels).sum().item()
                loss = criterion(this_output, labels)
                val_loss += loss.item()
            vali_loss[epoch] = float(val_loss) / (i+1)
            acc[epoch] = float(correct) / total

            save_model(self.mynn, '{}_{}.pth'.format(self.net_name, epoch))

            train_log.write(str(time_spend) + '\t' + str(epoch_loss[epoch])+'\t'+str(vali_loss[epoch])+ '\t' + str(acc[epoch]) + str(learning_rate) + '\n')
            print(' Epoch{}: Time={:.2f}s TLoss={:.4f} VaLoss={:.4f} Acc={:.4f} lr={}\n'.format(epoch, time_spend, epoch_loss[epoch], vali_loss[epoch], acc[epoch], learning_rate))
            if epoch in [10, 15, 20, 25]:
                 learning_rate *= 0.1
            '''
            if epoch > 0:
                #print(str(epoch_loss[epoch-1]-epoch_loss[epoch]))
                if (epoch_loss[epoch-1]-epoch_loss[epoch]) < 0.001*epoch_loss[epoch-1]:
                    learning_rate *= 0.1
            # end the epoch when loss is small enough
            #if epoch_loss[epoch] < 0.01:
            #    print('The training ends at ' + str(epoch) + ' epochs. \n')
            #    break
            '''
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
