import numpy as np 
import torch
import torch.nn.functional as F

class Conv2D(object):
    def __init__(self, in_channel, o_channel, kenrel_size, stride, mode):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kenrel_size = kenrel_size
        self.stride = stride
        self.mode = mode
        self.K1 = torch.tensor([-1, -1, -1; 0, 0, 0; 1, 1, 1])
        self.K2 = torch.tensor([-1, 0, 1; -1, 0, 1; -1, 0, 1])
        self.K3 = torch.tensor([1, 1, 1; 1, 1, 1; 1, 1, 1])
        self.K4 = torch.tensor([-1, -1, -1, -1, -1; -1, -1, -1, -1, -1; 0, 0, 0, 0, 0; 1, 1, 1, 1, 1; 1, 1, 1, 1, 1])
        self.K5 = torch.tensor([-1, -1, 0, 1, 1; -1, -1, 0, 1, 1; -1, -1, 0, 1, 1; -1, -1, 0, 1, 1; -1, -1, 0, 1, 1])
        self.kernel = torch.zeros(kenrel_size, kenrel_size, 3, o_channel)

        if mode == 'known':
            if o_channel == 1:
                self.kernel = torch.stack(self.K1, 3)
            if o_channel == 2:
                self,kernel(:,:,:,0) = torch.stack(self.K4, 3)
                self.kernel(:,:,:,1) = torch.stack(self.K5, 3)
            if o_channel == 3:
                self.kernel(:,:,:,0) = torch.stack(self.K1, 3)
                self.kernel(:,:,:,1) = torch.stack(self.K2, 3)
                self.kernel(:,:,:,2) = torch.stack(self.K3, 3)
                
        if mode == 'rand':
            self.kernel = torch.randn(kenrel_size, kenrel_size, 3, o_channel)

    def forward(self, input):
        if type(input) is a numpy.ndarray:
            #input = input.transpose((2, 0, 1))
            input = torch.from_numpy(input)
        [d1,d2,d3] = torch.Size(input) 
        od1 = (d1 - self.kenrel_size)/self.stride + 1
        od2 = (d2 - self.kenrel_size)/self.stride + 1
        output = torch.empty(od1, od2, self.o_channel, dtype=torch.float)
        #output0 = torch.empty(od1, od2, dtype=torch.float)
        #output1 = output0
        #output2 = output0

        x = 0 # number of operations
        '''
        if self.o_channel == 1:
            for i in range(od1):
                for j in range(od2):
                    output[i, j] = np.sum(input(i*self.stride:i*self.stride+self.kenrel_size, :)*self.kernel)
                    x = x + 1
        
        if self.o_channel == 2:
            for i in range(od1):
                for j in range(od2):
                    output0[i, j] = np.sum(input(i*self.stride:i*self.stride+self.kenrel_size, :)*self.kernel)
                    output1[i, j] = np.sum(input(i*self.stride:i*self.stride+self.kenrel_size, :)*self.kernel])
                    x = x + 2
            output[:,:,0] = output0
            output[:,:,1] = output1
'''
        for cn in range(self.o_channel): # for each output channel
            for i in range(od1):
                for j in range(od2):
                    output[i, j, cn] = np.sum(input(i*self.stride:i*self.stride+self.kenrel_size-1, :)*self.kernel[:,:,:,cn])
                    x = 3*(x + self.kenrel_size **2 + self.kenrel_size)
        # normalize output
            upper = torch.max(output[:,:,cn])
            lower = torch.min(output[:,:,cn])
            output[:,:,cn] = torch.round (255 * (output[:,:,cn] - lower) / (upper - lower))

        return x, output