import numpy as np 
import torch
import torch.nn.functional as F

class Conv2D(object):
    def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        self.K1 = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        self.K2 = torch.tensor([[-1, 0, 1], [-1, 0, 1,], [-1, 0, 1]])
        self.K3 = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self.K4 = torch.tensor([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        self.K5 = torch.tensor([[-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1]])
        self.kernel = torch.zeros(kernel_size, kernel_size, 3, o_channel).float()

        if mode == 'known':
            if o_channel == 1:
                self.kernel = self.K1.expand(kernel_size,kernel_size,3)
                #print (self.kernel)
            if o_channel == 2:
                self,kernel[:,:,:,0] = self.K4.expand(kernel_size,kernel_size,3)#torch.stack(self.K4, 3)
                self.kernel[:,:,:,1] = self.K5.expand(kernel_size,kernel_size,3)#torch.stack(self.K5, 3)
            if o_channel == 3:
                self.kernel[:,:,:,0] = self.K1.expand(kernel_size,kernel_size,3)#torch.stack(self.K1, 3)
                self.kernel[:,:,:,1] = self.K2.expand(kernel_size,kernel_size,3)#torch.stack(self.K2, 3)
                self.kernel[:,:,:,2] = self.K3.expand(kernel_size,kernel_size,3)#torch.stack(self.K3, 3)
                
        if mode == 'rand':
            self.kernel = torch.empty(kernel_size, kernel_size, 3, o_channel).self.kernel.uniform(-1, 1)#torch.randn(kernel_size, kernel_size, 3, o_channel)

    def forward(self, input):
        if type(input) == "numpy.ndarray":
            #input = input.transpose((2, 0, 1))
            input = torch.from_numpy(input)
        [d1,d2,d3] = input.size() 
        #print(d1,d2,d3)
        od1 = (d1 - self.kernel_size)/self.stride + 1
        od2 = (d2 - self.kernel_size)/self.stride + 1
        output = torch.empty(od1, od2, self.o_channel, dtype=torch.float)
        #output0 = torch.empty(od1, od2, dtype=torch.float)
        #output1 = output0
        #output2 = output0

        x = 0 # number of operations
        '''
        if self.o_channel == 1:
            for i in range(od1):
                for j in range(od2):
                    output[i, j] = np.sum(input(i*self.stride:i*self.stride+self.kernel_size, :)*self.kernel)
                    x = x + 1
        
        if self.o_channel == 2:
            for i in range(od1):
                for j in range(od2):
                    output0[i, j] = np.sum(input(i*self.stride:i*self.stride+self.kernel_size, :)*self.kernel)
                    output1[i, j] = np.sum(input(i*self.stride:i*self.stride+self.kernel_size, :)*self.kernel])
                    x = x + 2
            output[:,:,0] = output0
            output[:,:,1] = output1
'''
        for cn in range(self.o_channel): # for each output channel
            for i in range(od1):
                for j in range(od2):
                    temp = input[i*self.stride : i*self.stride+self.kernel_size, j*self.stride : j*self.stride+self.kernel_size, :].float()
                    #print(temp.size())
                    if self.o_channel == 1:
                        output[i, j, cn] = (temp*self.kernel.float()).sum()
                        #print(temp*self.kernel.float())
                    else:
                        output[i, j, cn] = torch.sum(temp*self.kernel[:,:,:,cn])
                    x = 3*(x + self.kernel_size **2 + self.kernel_size)
        # normalize output
            upper = torch.max(output[:,:,cn])
            lower = torch.min(output[:,:,cn])
            output[:,:,cn] = torch.round (255 * (output[:,:,cn] - lower) / (upper - lower))

        return x, output
