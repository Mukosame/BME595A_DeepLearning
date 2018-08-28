import Conv2D 
import matplotlib.pyplot as plt

input_img1 = Image('1280.jpg')
input_img2 = Image('1920.jpg')

# Part A
# init task 1
conv2d = Conv2D(3, 1, 3, 1, 'known')
op_a, out_1 = conv2d.forward(input_img1)
out_1.save("out_1.jpg")
op_a, out_1 = conv2d.forward(input_img2)
out_2.save("out_2.jpg")

# Part B
# init
time_i = [0] * 11
x = [i for i in range(11)]

for i in range(x):
    conv2d = Conv2D(3, 2**i, 3, 1, 'rand')
    time_i[i], out_img = conv2d.forward(input_img1)

# plot the time taken for the performance as a function of i
plt.style.use('seaborn-whitegrid')
plt.plot(x, time_i, 'bo')
plt.xlabel('out channel 2^i')
plt.ylabel('time')

# Part C
size_list = [3, 5, 7, 9, 11]
time_k = [0] * 5
for kernel_size in range(size_list):
    conv2d = Conv2D(3, 2, kernel_size, 1, 'rand')
    time_k[kernel_size], out_img = conv2d.forward(input_img1)
# plot the time taken for the performance as a function of kenel size list
plt.style.use('seaborn-whitegrid')
plt.plot(size_list, time_k, 'bo')
plt.xlabel('kernel size')
plt.ylabel('time')