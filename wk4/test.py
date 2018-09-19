import torchvision
import torch
import torchvision.transforms as transforms
import MyImg2Num
import NnImg2Num

device = torch.device("cuda:1,2" if torch.cuda.is_available() else "cpu")
print(device)

## TRAIN ON MY NETWORK
mynet = MyImg2Num()
MyImg2Num.to(device)
mynet.train()

# TRAIN ON TORCH NETWORK
'''
net = NnImg2Num()
'''