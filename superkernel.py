import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util

def Indicator(x):
    """
    x must have the data type floating point
    """
    x = Variable(torch.tensor(x), requires_grad=True)
    return (torch.FloatTensor([x >= 0]) - torch.sigmoid(x)).detach() + torch.sigmoid(x)

class DepthwiseConv2DMasked(nn.Module):
    def __init__(self, 
           kernel_size,
           stride,
           dilation,
           padding, 
           depthwise_initializer, 
           use_bias,               
           runtimes=None,
           dropout_rate=None,
           **kwargs):
    
        super(DepthwiseConv2DMasked, self).__init__(
            kernel_size=kernel_size, 
            stride=stride,
            dilation=dilation,
            padding=padding,
            depthwise_initializer=depthwise_initializer, 
            use_bias=use_bias,
            **kwargs)

        self.runtimes = runtimes
        self.dropout_rate = dropout_rate

        if kernel_size[0] != 5: # normal Depthwise type
            self.custom = False
        else:
            self.custom = True 
            if self.runtimes is not None:
                self.R50c = torch.FloatTensor([self.runtimes[2]]) # 50% of the 5x5
                self.R100c = torch.FloatTensor([self.runtimes[3]]) # 100% of the 5x5
                self.R5x5 = torch.FloatTensor([self.runtimes[3]]) # 5x5 for 100%
                self.R3x3 = torch.FloatTensor([self.runtimes[1]]) # 3x3 for 100%
            else:
                self.R50c = torch.FloatTensor([0.0])
                self.R100c = torch.FloatTensor([0.0])
                self.R5x5 = torch.FloatTensor([0.0])
                self.R3x3 = torch.FloatTensor([0.0])
                
    def build(self, input_shape):
        """
        input format:  [batch, input_channels, input_height, input_width]
        weight format: [input_channel, 1, kernel_height, kernel_width]
        """
        input_channels = input_shape[1]
        
        # Depthwise layer
        self.model = util.Conv2D(                
                nin=input_channels,
                nout=1,
                layer_type="depthwise",
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                padding=self.padding,
                use_bias=self.use_bias)
        
        self.model.weight = nn.Parameter(data=torch.Tensor(input_channels, 1, self.kernel_size, self.kernel_size), requires_grad=True)
        # TODO: Not intialize the weights yet 
        depthwise_initializer(self.model.weight)
        
        if self.use_bias:
            self.model.bias = nn.Parameter(data=torch.Tensor(input_channels), requires_grad=True)
            depthwise_initializer(self.model.bias)
        
        runtime = 0.0 
        if not self.custom:
            self.depthwise_kernel_masked = self.model.weight
            self.runtime_reg = runtime
            
        else:
            # the original implementation is channels_last
            # my implementation is channel first
            assert len(input_shape) == 4
            kernel_shape = self.model.weight.size()
            assert kernel_shape[1] == 1 # I don't think we handle depth mult
            assert kernel_shape[2] == 5 # you cannot mask out if it is 3x3 already! 
            assert kernel_shape[3] == 5 # you cannot mask out if it is 3x3 already! 

            # Thresholds
            self.t5x5 = nn.Parameter(data=torch.zeros(1, ), requires_grad=True)
            self.t50c = nn.Parameter(data=torch.zeros(1, ), requires_grad=True)
            self.t100c = nn.Parameter(data=torch.zeros(1, ), requires_grad=True)

            # create masks based on kernel_shape
            center_3x3 = np.zeros(kernel_shape)
            center_3x3[:,:,1:4,1:4] = 1.0 # center 3x3
            self.mask3x3 = torch.from_numpy(center_3x3, dtype=self.t5x5.dtype)

            center_5x5 = np.ones(kernel_shape) - center_3x3 # 5x5 - center 3x3
            self.mask5x5 = torch.from_numpy(center_5x5, dtype=self.t5x5.dtype)

            num_channels = int(kernel_shape[0])
            c50 = int(round(1.0*num_channels/2.0)) #  50 %
            c100 = int(round(2.0*num_channels/2.0)) # 100 %

            mask_50c = np.zeros(kernel_shape)
            mask_50c[0:c50,:,:,:] = 1.0 # from 0% to 50% channels
            self.mask50c = torch.from_numpy(mask_50c, dtype=self.t5x5.dtype)

            mask_100c = np.zeros(kernel_shape)
            mask_100c[c50:c100,:,:,:] = 1.0 # from 50% to 100% channels
            self.mask100c = torch.from_numpy(mask_100c, dtype=self.t5x5.dtype)

            #--> make indicator results "accessible" as separate vars
            kernel_3x3 = self.model.weight * self.mask3x3
            kernel_5x5 = self.model.weight * self.mask5x5
            self.norm5x5 = torch.norm(kernel_5x5, p=None)

            x5x5 = self.norm5x5 - self.t5x5
            if self.dropout_rate is not None: # zero-out with drop_prob_ 
                self.d5x5 = nn.Dropout(Indicator(x5x5), self.dropout_rate)
            else:
                self.d5x5 = Indicator(x5x5)


            depthwise_kernel_masked_outside = kernel_3x3 + kernel_5x5 * self.d5x5 

            kernel_50c = depthwise_kernel_masked_outside * self.mask50c
            kernel_100c = depthwise_kernel_masked_outside * self.mask100c
            self.norm50c = torch.norm(kernel_50c, p=None)
            self.norm100c = torch.norm(kernel_100c, p=None)
            
            
            x100c = self.norm100c - self.t100c
            if self.dropout_rate is not None: # noise to add
                self.d100c = nn.Dropout(Indicator(x100c), self.dropout_rate)
            else:
                self.d100c = Indicator(x100c) 

            if self.stride[0] == 1 and len(self.runtimes) == 5:
                x50c = self.norm50c - self.t50c
                if self.dropout_rate is not None: # noise to add
                    self.d50c = nn.Dropout(Indicator(x50c), self.dropout_rate)
                else:
                    self.d50c = Indicator(x50c) 
            else: # you cannot drop all layers!
                self.d50c = 1.0

            self.depthwise_kernel_masked = self.d50c * (kernel_50c + self.d100c * kernel_100c)

            # runtime term
            if self.runtimes is not None:
                ratio = self.R3x3 / self.R5x5
                runtime_channels = self.d50c * (self.R50c + self.d100c * (self.R100c-self.R50c)) 
                runtime = runtime_channels * ratio + runtime_channels * (1-ratio) * self.d5x5

            self.runtime_reg = runtime
            
            
    def forward(self, inputs, total_runtime):
        
        self.build(inputs.size())
        self.model.weight = self.depthwise_kernel_masked
        
        outputs = self.model(inputs)
        total_runtime = total_runtime + self.runtime_reg

        return outputs, total_runtime 
