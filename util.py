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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

class LabelSmoothing(nn.Module):
    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None
        
    def forward(self, x, target):
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (x.size(1) - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
