from __future__ import print_function
import os,sys,cv2,random,datetime,time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser(description='PyTorch AlexNet')
parser.add_argument('--data', default='/data/', type=str)
parser.add_argument('--save', default='/model/', type=str)
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
lr = 0.01

def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()

def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')

def save_model(model,filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()    
    torch.save(state, args.save+filename)

class AlexNet(nn.module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),  # performs ReLU operation on the conv layer ouput in place
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 200)
        )

    def forward(self, input):
        """
        Defines the forward computation performed at every call by defined AlexNet network
        """
        out = self.features(input)
        out = out.view(out.size(0), -1)  # linearized the output of the module 'features'
        out = self.classifier(out)
        out = F.softmax(out)  # apply softmax activation function on the output of the module 'classifier'
        return out

def train(epoch,args):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    batch = 128#batch size
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = todo torchvision.datasets.CIFAR100(root='./data', train = True, download = True, transform = transform)
    trainloader = todo torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle = True, num_workers = 2)
    for i, data in enumerate(trainloader, 0):
        inputs, targets = data
        if use_cuda: inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        #inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        outputs = outputs[0] # 0=cos_theta 1=phi_theta
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        printoneline(dt(),'Te=%d Loss=%.4f | AccT=%.4f%% (%d/%d) %.4f'
            % (epoch,train_loss/(batch_idx+1), 100.0*correct/total, correct, total, 
            loss.item()))
        batch_idx += 1
    print('')
    return float(train_loss)/(i+1)

def validate(epoch, args):
    testset = todo
    testloader = todo
    val_loss = 0.0
    ## Test the network with test data
    correct = 0
    total = 0
    for i, data in enumerate(testloader, 0):
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
    return = float(val_loss) / (i+1)

net = AlexNet()
net.load_state_dict(torch.load('model/todo'))
net.cuda()
criterion = nn.CrossEntropyLoss()

max_iteration = 20
print('start: time={}'.format(dt()))
epoch_loss = np.zeros(max_iteration)
for epoch in range(0, max_iteration):
    if epoch in [0,10,15,18]:
        if epoch!=0: args.lr *= 0.1
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    epoch_loss[epoch] = train(epoch,args)
    vali_loss[epoch] = validate(epoch, args)
    save_model(net, '{}_{}.pth'.format('AlexNet',epoch))
print('finish: time={}\n'.format(dt()))

# plot loss vs epoch
x = np.arange(1,epoch+2)
fig1 = plt.figure(1)
plt.style.use('seaborn-whitegrid')
plt.plot(x, epoch_loss[0:epoch+1], 'bo-', label='Training Loss') 
plt.plot(x,vali_loss[0:epoch+1], 'ro-', label = 'Validation Loss')    
plt.xlabel('Epoch')    
plt.ylabel('Loss')
plt.legend()
fig1.savefig('train_loss.jpg', dpi = fig1.dpi)