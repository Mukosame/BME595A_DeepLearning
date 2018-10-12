import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse, os, sys
#import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
import pyscreenshot

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
        out = F.softmax(out)  # apply softmax activation function on the output of the module 'classifier'
        return out

parser = argparse.ArgumentParser(description='PyTorch AlexNet')
parser.add_argument('--model', default='model/', type=str)
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



def extract_class_name(wtf, class_dict):
    class_name = {}
    #print(wtf)
    if wtf in class_dict:
        class_name = class_dict[wtf]
    return class_name


class TestNet:
    def __init__(self):
        self.net = AlexNet()
        # load saved model
        filepath = os.path.join(args.model, 'AlexNet_19.pth')
        if os.path.isfile(filepath):
            saved = torch.load(filepath)
            self.net.load_state_dict(saved['model'])
            self.num2class = saved['num2class']
            #print(self.num2class)
            self.class_dict = saved['class_dict']
            #print(self.class_dict)
        else:
            print('No such model \n')
            sys.exit(0)

    def forward(self, img):
        input = torch.unsqueeze(img.type(torch.FloatTensor), 0)
        outputs = self.net(input)
        _, prediction = torch.max(outputs, 1)
        #print(prediction.data[0])
        return extract_class_name(self.num2class[prediction.data[0]], self.class_dict)

    def view(self, img):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        preprocess = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224), transform])
        def imshow(cvimg, text):
            #img = img / 2 + 0.5 #unnormalize
            #npimg = img.numpy()
            #toshow = np.transpose(npimg, (1, 2, 0))
            #cvimg = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)
            cv2.namedWindow(text, cv2.WINDOW_NORMAL)        
            #cv2.resizeWindow(text, 512, 512)
            cv2.putText(img, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)          
            cv2.imshow(text,cvimg)  
            #cv2.waitKey(1)
            #im = pyscreenshot.grab()
            #im.save('view_result.jpg')
            #cv2.destroyAllWindows()  
        tensorimg = preprocess(img)
        prediction = self.forward(tensorimg)
        print(prediction+'\n')
        imshow(img, prediction)

    def cam(self, idx = 0):
        
        obj = cv2.VideoCapture(idx)
        obj.set(3, 512)
        obj.set(4, 512)
        while True:
            yes, frame = obj.read()
            if yes:
                self.view(frame)
            else:
                print('Cannot reading from webcam \n')
                break
            if (cv2.waitKey(1)):
                break

        obj.release()
        cv2.destroyAllWindows()




test_net = TestNet()
#sample_img = cv2.imread('neko.JPEG')
#test_net.view(sample_img)
test_net.cam()