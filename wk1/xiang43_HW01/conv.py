import numpy as np 
import torch
import torch.nn.functional as F

device = torch.device("cuda:1,2" if torch.cuda.is_available() else "cpu")
class Conv2D(object):
    def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        self.K1 = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).to(device)
        self.K2 = torch.tensor([[-1, 0, 1], [-1, 0, 1,], [-1, 0, 1]]).to(device)
        self.K3 = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).to(device)
        self.K4 = torch.tensor([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]).to(device)
        self.K5 = torch.tensor([[-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1]]).to(device)
        self.kernel = torch.zeros(o_channel, 3, kernel_size, kernel_size).float().to(device)

        if mode == 'known':
            if o_channel == 1:
                self.kernel = self.K1.expand(3,kernel_size,kernel_size)
                #print (self.kernel)
            if o_channel == 2:
                self.kernel[0,:,:,:] = self.K4.expand(3, kernel_size,kernel_size)#torch.stack(self.K4, 3)
                self.kernel[1,:,:,:] = self.K5.expand(3,kernel_size,kernel_size)#torch.stack(self.K5, 3)
            if o_channel == 3:
                self.kernel[0,:,:,:] = self.K1.expand(3,kernel_size,kernel_size)#torch.stack(self.K1, 3)
                self.kernel[1,:,:,:] = self.K2.expand(3,kernel_size,kernel_size)#torch.stack(self.K2, 3)
                self.kernel[2,:,:,:] = self.K3.expand(3,kernel_size,kernel_size)#torch.stack(self.K3, 3)
                
        if mode == 'rand':
            self.kernel =torch.randn(o_channel, 3, kernel_size, kernel_size).to(device) # torch.empty(o_channel, 3, kernel_size, kernel_size).uniform(-1, 1)#

    def forward(self, input):
        if type(input) == "numpy.ndarray":
            input = torch.from_numpy(input)
        [d1,d2,d3] = input.size() 
        #print(d1,d2,d3)
        od1 = (d1 - self.kernel_size)/self.stride + 1
        od2 = (d2 - self.kernel_size)/self.stride + 1
        output = torch.empty(od1, od2, self.o_channel, dtype=torch.float).to(device)
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
                    else:
                        temk = self.kernel[cn,:,:,:].transpose(0,2).transpose(0,1)
                        #print(temk.size())
                        output[i, j, cn] = (temp*temk.float()).sum()
                    x = x + 3*(self.kernel_size **2 + self.kernel_size - 1) + 2
        # normalize output
            upper = torch.max(output[:,:,cn])
            lower = torch.min(output[:,:,cn])
            output[:,:,cn] = torch.round (255 * (output[:,:,cn] - lower) / (upper - lower))

        return x, output
