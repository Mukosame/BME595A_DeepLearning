import numpy as np 

class Conv2D(object):
    def __init__(self, in_channel, o_channel, kenrel_size, stride, mode):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kenrel_size = kenrel_size
        self.stride = stride
        self.mode = mode
        self.K1 = [-1, -1, -1; 0, 0, 0; 1, 1, 1]
        self.K2 = [-1, 0, 1; -1, 0, 1; -1, 0, 1]
        self.K3 = [1, 1, 1; 1, 1, 1; 1, 1, 1]
        self.K4 = [-1, -1, -1, -1, -1; -1, -1, -1, -1, -1; 0, 0, 0, 0, 0; 1, 1, 1, 1, 1; 1, 1, 1, 1, 1]
        self.K5 = [-1, -1, 0, 1, 1; -1, -1, 0, 1, 1; -1, -1, 0, 1, 1; -1, -1, 0, 1, 1; -1, -1, 0, 1, 1]

        if mode == 'known':
            if o_channel == 1:
                
            if o_channel == 2:
                
            if o_channel == 3:
                
        if mode == 'rand':
            self.kernel = 

    def forward(self, input):
        
        return x, output
