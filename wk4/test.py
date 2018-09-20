import torchvision
import torch
import torchvision.transforms as transforms
from my_img2num import MyImg2Num
from nn_img2num import NnImg2Num

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

## TRAIN ON MY NETWORK
mynet = MyImg2Num()
#mynet.to(device)
mynet.train()

# TRAIN ON TORCH NETWORK

net = NnImg2Num()
#net.to(device)
net.train()

