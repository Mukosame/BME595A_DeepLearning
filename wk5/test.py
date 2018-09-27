from img2num import Img2Num
from img2obj import Img2Obj
from PIL import Image
import numpy as np
import torch
'''
net1 = Img2Num()
net1.train()

net2 = Img2Obj()
net2.train()
'''
# visualization of img2obj
net2 = Img2Obj()
read_img1 = Image.open('neko.jpg')
image_1 = np.asarray(read_img1)#.to(device)
net2.view(image_1)
