from conv import Conv2D
import matplotlib.pyplot as plt
from PIL import Image
import time
import torch
import numpy as np
from torchvision import transforms, utils

def save_timg(tensor, filename):
    npimg = tensor.numpy()
    im = Image.fromarray(npimg.astype('uint8'))
    im.save(filename)

device = torch.device("cuda:1,2" if torch.cuda.is_available() else "cpu")    
read_img1 = Image.open('1280.jpg')
read_img2 = Image.open('1920.jpg')
input_img1 = torch.from_numpy(np.asarray(read_img1)).to(device)
input_img2 = torch.from_numpy(np.asarray(read_img2)).to(device)
# Part A
# init task 1
'''
conv2d = Conv2D(3, 1, 3, 1, 'known')
op_a1, out_1 = conv2d.forward(input_img1)
save_timg(out_1[:,:,0], 'a1.jpg')
op_a2, out_2 = conv2d.forward(input_img2)
save_timg(out_2[:,:,0], 'a2.jpg')
# task 2
conv2d = Conv2D(3, 2, 5, 1, 'known')
op_a3, out_3 = conv2d.forward(input_img1)
save_timg(out_3[:,:,0], 'a3.jpg')
save_timg(out_3[:,:,1], 'a4.jpg')
op_a4, out_4 = conv2d.forward(input_img2)
save_timg(out_4[:,:,0], 'a5.jpg')
save_timg(out_4[:,:,1], 'a6.jpg')
# task 3
conv2d = Conv2D(3, 3, 3, 2, 'known')
op_a5, out_5 = conv2d.forward(input_img1)
save_timg(out_5[:,:,0], 'a7.jpg')
save_timg(out_5[:,:,1], 'a8.jpg')
save_timg(out_5[:,:,2], 'a9.jpg')
op_a6, out_6 = conv2d.forward(input_img2)
save_timg(out_6[:,:,0], 'a10.jpg')
save_timg(out_6[:,:,1], 'a11.jpg')
save_timg(out_6[:,:,2], 'a12.jpg')

# Part B
# init
time_i = [0.0] * 11
op_i = [0] * 11
#x = [i for i in range(11)]

for i in range(11): # output channel: 2^i
    start_time = time.time()
    conv2d = Conv2D(3, 2**i, 3, 1, 'rand')
    op_i[i], out_img = conv2d.forward(input_img1)
    time_i[i] = time.time() - start_time

# plot the time taken for the performance as a function of i
fig = plt.figure()
plt.style.use('seaborn-whitegrid')
plt.plot(x, time_i, 'bo')
plt.xlabel('out channel 2^i')
plt.ylabel('time/s')
fig.savefig('o_chanel_time.png', dpi = fig.dpi)
'''
# Part C
size_list = [3, 5, 7, 9, 11]
time_k = [0.0] * 5
op_k = [0] * 5
i = 0
'''
for kernel_size in size_list:
    #start_time = time.time()
    conv2d = Conv2D(3, 2, kernel_size, 1, 'rand')
    op_k[i], out_img = conv2d.forward(input_img1)
    print(op_k[i])
    i = i + 1
    #time_k[kernel_size] = time.time() - start_time
'''
op_k = [64232280, 162623648, 303818424, 487247232, 712343000]
# plot the time taken for the performance as a function of kenel size list
fig = plt.figure()
plt.style.use('seaborn-whitegrid')
plt.plot(size_list, op_k, 'bo')
plt.xlabel('kernel size')
plt.ylabel('operations')
fig.savefig('kernel_size_time.png', dpi = fig.dpi)
'''
'''
