import torch
import numpy as np
import argparse, os, sys
#import matplotlib.pyplot as plt
import cv2
from train import AlexNet
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch AlexNet')
parser.add_argument('--model', default='model/', type=str)
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def build_class_dict():
    protocol = '/data/tiny-imagenet-200/words.txt'
    class_dict = {}
    with open(protocol) as f:
        id_label_lines = f.readline()
    for line in id_label_lines:
        l = line.replace('\n','').split('\t')
        word_label = l[1].split(',')
        class_dict[l[0]] = word_label[0].rstrip()
    f.close()
    return class_dict

def extract_class_name(wtf, class_dict):
    class_name = {}
    for i in wtf:
        for j, k in class_dict.items():
            if i == j:
                class_name[j] = k
                continue
    return class_name


class TestNet:
    def __init__(self):
        self.net = AlexNet()
        # load saved model
        filepath = os.path.join(args.model, 'AlexNet_19.pth')
        if os.path.isfile(filepath):
            self.net.load_state_dict(torch.load(filepath))
        else:
            print('No such model \n')
            sys.exit(0)

    def forward(self, img):
        input = torch.unsqueeze(img.type(torch.FloatTensor), 0)
        outputs = self.net(input)
        _, prediction = torch.max(outputs, 1)
        return extract_class_name(prediction.data[0], class_dict)

    def view(self, img):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        preprocess = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256), transforms.CenterCrop(224), transform])
        def imshow(cvimg, text):
            #img = img / 2 + 0.5 #unnormalize
            #npimg = img.numpy()
            #toshow = np.transpose(npimg, (1, 2, 0))
            #cvimg = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR)
            cv2.namedWindow(text, cv2.WINDOW_NORMAL)        
            cv2.resizeWindow(text, 512, 512)
            cv2.putText(img, text, (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 100), 5, cv2.LINE_AA)          
            cv2.imshow(text,cvimg)  
            #cv2.waitKey(1)  
            #cv2.destroyAllWindows()  
        tensorimg = preprocess(img)
        prediction = self.forward(tensorimg)
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



class_dict = build_class_dict()
test_net = TestNet()
sample_img = cv2.imread('neko.jpg')
test_net.view(sample_img)
#test_net.cam()