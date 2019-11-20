import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_same_padding(kernel_size, stride=1, dilation=1):
    assert isinstance(stride, int) or isinstance(stride, tuple) 
    if isinstance(stride, int):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    else:
        paddding0 = ((stride[0] - 1) + dilation * (kernel_size - 1)) // 2
        paddding1 = ((stride[1] - 1) + dilation * (kernel_size - 1)) // 2
        padding = (paddding0, paddding1) 
    return padding

class Conv2D(nn.Module):
    def __init__(self, nin, nout, layer_type, kernel_size, stride=1, dilation=1, padding="same", use_bias=False):
        super(Conv2D, self).__init__()
        assert layer_type == "depthwise" or layer_type == "conv2d"
        if padding == "same":
            padding = get_same_padding(kernel_size=kernel_size, stride=stride, dilation=dilation) # "same" padding
        if layer_type == "depthwise":
            self.model = nn.Conv2d(in_channels=nin, out_channels=nin, kernel_size=kernel_size, stride=stride, \
                                       dilation=dilation, bias=use_bias, padding=padding, groups=nin)
        else:
            self.model = nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=kernel_size, stride=stride, \
                                       dilation=dilation, bias=use_bias, padding=padding)
    def forward(self, x):
        return self.model(x)
    
def get_conv_output_size_and_nb_param(input_size, layer_type, kernel_size, stride, expansion=1, out_channels=1):
    assert isinstance(input_size, tuple)
    assert len(input_size) == 3
    assert layer_type == "depthwise" or layer_type == "conv2d"
    input_size = [input_size]
    inputs = [torch.rand(2, *in_size).type(torch.FloatTensor) for in_size in input_size][0]
    if (layer_type == "depthwise"):
        model = Conv2D(inputs.size()[1], inputs.size()[1], "depthwise", kernel_size, stride)
    else:
        model = Conv2D(inputs.size()[1], out_channels*expansion, "conv2d", kernel_size, stride)
    params = torch.prod(torch.tensor(model.model.weight.size()))
    output = model(inputs)
    return tuple(output.size()[1:]), params, model.model.weight.size()[2], model.model.weight.size()[3]

def get_linear_output_size_and_nb_param(in_features, out_features):
    params = in_features * out_features
    return tuple((out_features, )), params, out_features, in_features
