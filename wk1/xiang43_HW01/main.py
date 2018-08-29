import Conv2D 
import matplotlib.pyplot as plt
from skimage import io, transform
import time

input_img1 = io.imread('1280.jpg')
input_img2 = io.imread('1920.jpg')

# Part A
# init task 1
conv2d = Conv2D(3, 1, 3, 1, 'known')
op_a, out_1 = conv2d.forward(input_img1)
out_1.save("out_1.jpg")
op_a, out_1 = conv2d.forward(input_img2)
out_2.save("out_2.jpg")

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

# Part C
size_list = [3, 5, 7, 9, 11]
time_k = [0.0] * 5
op_k = [0] * 5
for kernel_size in range(size_list):
    start_time = time.time()
    conv2d = Conv2D(3, 2, kernel_size, 1, 'rand')
    op_k[kernel_size], out_img = conv2d.forward(input_img1)
    time_k[kernel_size] = time.time() - start_time
# plot the time taken for the performance as a function of kenel size list
fig = plt.figure()
plt.style.use('seaborn-whitegrid')
plt.plot(size_list, time_k, 'bo')
plt.xlabel('kernel size')
plt.ylabel('time/s')
fig.savefig('kernel_size_time.png', dpi = fig.dpi)