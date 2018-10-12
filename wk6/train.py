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
import torchvision.models as models
import torchvision.datasets as datasets

parser = argparse.ArgumentParser(description='PyTorch AlexNet')
parser.add_argument('--data', default='data/', type=str)
parser.add_argument('--save', default='model/', type=str)
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001

def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()

def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')

def build_class_dict():
    protocol = args.data + 'words.txt'
    class_dict = {}
    with open(protocol) as f:
        id_label_lines = f.readlines()
    for line in id_label_lines:
        l = line.replace('\n','').split('\t')
        #print(l)
        word_label = l[1].split(',')
        #print(word_label[0].rstrip())
        class_dict[l[0]] = word_label[0].rstrip()
    f.close()
    return class_dict

def save_model(state,filename):  
    torch.save(state, args.save+filename)

class AlexNet(nn.Module):
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
        out = F.softmax(out, dim=None)  # apply softmax activation function on the output of the module 'classifier'
        return out

def train(epoch, trainset, args):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    batch = 128#batch size
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=5)
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
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        printoneline(dt(),'Te=%d Loss=%.4f | AccT=%.4f%% (%d/%d) %.4f'
            % (epoch,train_loss/(batch_idx+1), 100.0*float(correct)/total, correct, total, 
            loss.item()))
        batch_idx += 1
    print('')
    return float(train_loss)/(i+1)
'''
def validate(epoch, args):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    testset = datasets.ImageFolder(os.path.join(args.data, 'test'), transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transform]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=5)
    val_loss = 0.0
    ## Test the network with test data
    correct = 0
    total = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        #labels = onehot(raw_labels) 
        this_output = net(inputs)
        _, prediction = torch.max(this_output.data, 1)
        #c = (prediction == raw_labels).squeeze()
        total += labels.size(0)
        correct += (prediction == labels).sum().item()
        loss = criterion(this_output, labels)
        val_loss += loss.item()
    return  float(val_loss) / (i+1)
'''
net = AlexNet()
alex_pretrained = models.alexnet(pretrained=True)
# copy weights
for i,j in zip(net.modules(), alex_pretrained.modules()): 
    if not list(i.children()):
        if len(i.state_dict()) > 0:
            if i.weight.size() == j.weight.size():
                i.weight.data = j.weight.data
                i.bias.data = j.bias.data
#net.load_state_dict(torch.load(alex_pretrained))
for param in net.parameters():
    param.requires_grad = False
for param in net.classifier[6].parameters():
    param.requires_grad = True
    #only update classifier
net.to(device)
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
trainset = datasets.ImageFolder(os.path.join(args.data, 'train'), transform=transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transform]))

max_iteration = 20
class_dict = build_class_dict()
num2class_dict = trainset.classes

print('start: time={}'.format(dt()))
epoch_loss = np.zeros(max_iteration)
vali_loss = np.zeros(max_iteration)
for epoch in range(0, max_iteration):
    if epoch in [0,11,15]:
        if epoch!=0: learning_rate *= 0.1
        optimizer = optim.Adam(net.classifier[6].parameters(), lr=learning_rate)

    epoch_loss[epoch] = train(epoch, trainset, args)
    #vali_loss[epoch] = validate(epoch, args)
    net_tosave =  net.state_dict()
    for key in net_tosave: net_tosave[key] = net_tosave[key].clone().cpu()
    save_model({'model': net_tosave, 'num2class': num2class_dict, 'class_dict': class_dict}, '{}_{}.pth'.format('AlexNet',epoch))
print('finish: time={}\n'.format(dt()))

# plot loss vs epoch
x = np.arange(1,epoch+2)
fig1 = plt.figure(1)
plt.style.use('seaborn-whitegrid')
plt.plot(x, epoch_loss[0:epoch+1], 'bo-', label='Training Loss') 
#plt.plot(x,vali_loss[0:epoch+1], 'ro-', label = 'Validation Loss')    
plt.xlabel('Epoch')    
plt.ylabel('Loss')
plt.legend()
fig1.savefig('train_loss.jpg', dpi = fig1.dpi)