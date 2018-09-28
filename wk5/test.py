from img2num import Img2Num
from img2obj import Img2Obj
import cv2
import torch
import numpy as np
from PIL import Image
'''
net1 = Img2Num()
net1.train()

net2 = Img2Obj()
net2.train()
'''
# visualization of img2obj
net2 = Img2Obj()
image1 = cv2.imread('neko.jpg')
read_img1 = Image.open('neko.jpg')
image_1 = np.asarray(read_img1)#.to(device)
net2.view(image_1)
