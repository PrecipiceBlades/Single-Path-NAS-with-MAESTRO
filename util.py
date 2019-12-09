import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import re
import argparse
from collections import OrderedDict
from ptflops import get_model_complexity_info

def get_same_padding(kernel_size, stride=1, dilation=1):
    assert isinstance(stride, int) or isinstance(stride, tuple) 
    if isinstance(stride, int):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    else:
        paddding0 = ((stride[0] - 1) + dilation * (kernel_size - 1)) // 2
        paddding1 = ((stride[1] - 1) + dilation * (kernel_size - 1)) // 2
        padding = (paddding0, paddding1) 
    return padding

class ConvBlock(nn.Module):
    def __init__(self, nin, nout, layer_type, kernel_size, stride=1, dilation=1, padding="same", use_bias=False, use_bn=True, use_act=True):
        super(ConvBlock, self).__init__()
        self.layer_type = layer_type
        assert layer_type == "DWConv" or layer_type == "SepConv" or layer_type == "Conv" 
        if padding == "same":
            padding = get_same_padding(kernel_size=kernel_size, stride=stride, dilation=dilation) # "same" padding
            
        if layer_type == "DWConv":
            self.conv = nn.Conv2d(in_channels=nin, out_channels=nin, kernel_size=kernel_size, stride=stride, \
                                       dilation=dilation, bias=use_bias, padding=padding, groups=nin)
            self.bn = nn.BatchNorm2d(nin)
            self.act = nn.ReLU(inplace=True)
            self.model = nn.Sequential(*([self.conv] + use_bn * [self.bn] + use_act * [self.act]))
            
        elif layer_type == "SepConv":
            self.layer1 = nn.Conv2d(in_channels=nin, out_channels=nin, kernel_size=kernel_size, stride=stride, \
                                       dilation=dilation, bias=use_bias, padding= \
                                       get_same_padding(kernel_size=kernel_size, stride=stride, dilation=dilation),\
                                       groups=nin)
            self.layer2 = nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=1, stride=1, \
                                       dilation=1, bias=use_bias, padding=\
                                       get_same_padding(kernel_size=1, stride=1, dilation=1))
            
            self.bn1, self.bn2 = nn.BatchNorm2d(nin), nn.BatchNorm2d(nout)
            self.act = nn.ReLU(inplace=True)          
            self.part1 = nn.Sequential(*([self.layer1] + use_bn * [self.bn1] + use_act * [self.act]))
            self.model = nn.Sequential(*([self.layer1] + use_bn * [self.bn1] + use_act * [self.act] + [self.layer2] + use_bn * [self.bn2]))

        else:
            self.conv = nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=kernel_size, stride=stride, \
                                       dilation=dilation, bias=use_bias, padding=padding)  
            self.bn = nn.BatchNorm2d(nout)
            self.act = nn.ReLU(inplace=True)
            self.model = nn.Sequential(*([self.conv] + use_bn * [self.bn] + use_act * [self.act]))
            
    def forward(self, x):
        if self.layer_type == "SepConv":
            return self.part1(x), self.model(x)
        else:
            return self.model(x)
        
class Seq_Ex_Block(nn.Module):
    def __init__(self, in_ch, ratio, bias=True):
        super(Seq_Ex_Block, self).__init__()
        self.part1 = [GlobalAvgPool(), nn.Linear(in_ch, int(ratio * in_ch), bias=bias)]
        self.part2 = [nn.ReLU(inplace=True), nn.Linear(int(ratio * in_ch), in_ch, bias=bias), nn.Sigmoid()]
        self.model = nn.Sequential(*(self.part1 + self.part2))
    def forward(self, x):
        output1 = nn.Sequential(*self.part1)(x)
        output2 = x.mul(self.model(x).unsqueeze(-1).unsqueeze(-1))
        return output1, output2
    
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearBlock, self).__init__()
        self.model = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, x):
        return self.model(x)

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
    def forward(self, x):
        return nn.AdaptiveAvgPool2d(1)(x).squeeze(dim=-1).squeeze(dim=-1)
    
def get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, expansion=1, out_channels=1, use_bn=True, use_act=True):
    assert isinstance(input_size, tuple)
    assert len(input_size) == 3
    assert layer_type == "DWConv" or layer_type == "SepConv" or layer_type == "Conv"
    inputs = [torch.rand(1, *in_size).type(torch.FloatTensor) for in_size in [input_size]][0]
    if layer_type == "DWConv":
        model = ConvBlock(inputs.size()[1], inputs.size()[1], "DWConv", kernel_size, stride, use_bn=True, use_act=True)
    elif layer_type == "SepConv":
        model = ConvBlock(inputs.size()[1], out_channels, "SepConv", kernel_size, stride, use_bn=True, use_act=True)
    else:
        model = ConvBlock(inputs.size()[1], int(out_channels*expansion), "Conv", kernel_size, stride, use_bn=True, use_act=True)
    if layer_type == "SepConv":
        flops, params = get_model_complexity_info(model.model, input_size, as_strings=False, print_per_layer_stat=False)
        _, params_part1 = get_model_complexity_info(model.part1, input_size, as_strings=False, print_per_layer_stat=False)
        output1, output2 = model(inputs)
        return tuple(output1.size()[1:]), params_part1, model.layer1.weight.size()[2], model.layer1.weight.size()[3],\
              tuple(output2.size()[1:]), params-params_part1, model.layer2.weight.size()[2], model.layer2.weight.size()[3], flops
    else:
        output = model(inputs)
        flops, params = get_model_complexity_info(model.model, input_size, as_strings=False, print_per_layer_stat=False)
        return tuple(output.size()[1:]), params, model.conv.weight.size()[2], model.conv.weight.size()[3], flops
    
def get_se_output_and_params_and_flops(input_size, expansion=0.25, bias=True):
    assert isinstance(input_size, tuple)
    assert len(input_size) == 3
    inputs = [torch.rand(1, *in_size).type(torch.FloatTensor) for in_size in [input_size]][0]
    model = Seq_Ex_Block(inputs.size()[1], expansion, bias=bias)
    flops, _ = get_model_complexity_info(model.model, input_size, as_strings=False, print_per_layer_stat=False)
    flops += np.prod(input_size)
    net_with_weights = [model.model[1], model.model[3]]
    params = [torch.prod(torch.tensor(x.weight.size())) for x in net_with_weights]
    output1, output2 = model(inputs)
    return tuple(output1.size()[1:]), params[0] + bias * output1.size()[1], model.part1[1].weight.size()[0], model.part1[1].weight.size()[1], \
       tuple(output2.size()[1:]), params[1] + bias * output2.size()[1], model.part2[1].weight.size()[0], model.part2[1].weight.size()[1], flops

def get_linear_output_size_and_nb_param(input_size, out_features, bias=True, use_pool=True):
    assert isinstance(input_size, tuple)
    if use_pool:
        assert len(input_size) == 3
        inputs = [torch.rand(1, *in_size).type(torch.FloatTensor) for in_size in [input_size]][0]
        model = GlobalAvgPool()
        flop, _ = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
        inputs = model(inputs)
        in_features = input_size[0]
        model = LinearBlock(in_features, out_features, bias=True)
        flops, params = get_model_complexity_info(model.model, (in_features, ), as_strings=False, print_per_layer_stat=False)
        flops += flop
    else:
        assert len(input_size) == 1
        in_features = input_size[0]
        model = LinearBlock(in_features, out_features, bias=True)
        flops, params = get_model_complexity_info(model.model, (in_features, ), as_strings=False, print_per_layer_stat=False)       
    return tuple((out_features, )), params, out_features, in_features, flops

def profile_blockargs(blocks_string, input_size, use_bias=True):
    assert isinstance(blocks_string, str)
    options = {}
    ops = blocks_string.split('_')
    options["Squeeze_excitation"] = False
    if ops[-2] == "se":
        options["Squeeze_excitation"] = True
        options["se_ratio"] = float(ops[-1])
        ops = ops[:-2]
    for op in ops:
        splits = re.split(r'(\d.*)', op)
        if len(splits) >= 2:
            key, value = splits[:2]
            options[key] = value
    if 's' not in options or len(options['s']) != 2:
        raise ValueError('Strides options should be a pair of integers.')

    options["kernel_size"]    = int(options["k"])
    options["num_repeat"]     = int(options["r"])
    options["input_filters"]  = int(options["i"])
    options["output_filters"] = int(options["o"])
    options["stride"]         = (int(options['s'][0]), int(options['s'][1]))
    options["expansion"]      = float(options["e"])
    
    output_size = input_size
    out_channels = options["input_filters"]
    layers = []
    flops = 0
    for key, r in enumerate(range(options["num_repeat"])):
        layer_type1, input_size, kernel_size, stride, expansion = \
            "Conv", output_size, 1, (1, 1), options["expansion"] 
        output_size, nb_params, R, S, flop = \
            get_conv_output_and_params_and_flops(input_size, layer_type1, kernel_size, stride, expansion, out_channels)
        layers.append(tuple((layer_type1, input_size, output_size, stride, nb_params, R, S)))
        flops += flop
        
        layer_type2, input_size, kernel_size, stride = "DWConv", output_size, options["kernel_size"], options["stride"]
        output_size, nb_params, R, S, flop = \
            get_conv_output_and_params_and_flops(input_size, layer_type2, kernel_size, stride)
        layers.append(tuple((layer_type2, input_size, output_size, stride, nb_params, R, S)))
        flops += flop
        
        if options["Squeeze_excitation"]:
            input_size, expansion = output_size, options["se_ratio"]
            output_size1, nb_params1, R1, S1, output_size, nb_params2, R2, S2, flop = \
                get_se_output_and_params_and_flops(input_size, expansion=expansion, bias=use_bias)
            layers.append(tuple(("Linear", input_size, output_size1, None, nb_params1, R1, S1)))
            layers.append(tuple(("Linear", output_size1, output_size, None, nb_params2, R2, S2)))
            flops += flop
    
        layer_type3, input_size, kernel_size, stride, out_channels = "Conv", output_size, 1, (1, 1), options["output_filters"]
        output_size, nb_params, R, S, flop = \
            get_conv_output_and_params_and_flops(input_size, layer_type3, kernel_size, stride, out_channels=out_channels, use_act=False)
        layers.append(tuple((layer_type3, input_size, output_size, stride, nb_params, R, S)))
        flops += flop
        if key == 0:
            options["stride"] = (1, 1)
            
    return layers, output_size, flops

def make_summary(profiled_layers, dataflow, model_name):
    # Make the summary dict
    summary = OrderedDict()
    for key, layer in enumerate(profiled_layers):
        batch_size = -1
        m_key = "%s-%i" % ("Conv", key+1) if layer[0] == "DWConv" else "%s-%i" % (layer[0], key+1)
        summary[m_key] = OrderedDict()
        summary[m_key]["type"] = "CONV"
        summary[m_key]["input_shape"] = [batch_size] + list(layer[1])
        summary[m_key]["output_shape"] = [batch_size] + list(layer[2])
        summary[m_key]["trainable"] = True
        summary[m_key]["nb_params"] = layer[4]
        if layer[0] != "Linear":
            C, Y, X = layer[1]
            K, Yo, Xo = layer[2]
            R, S = layer[5], layer[6]
            N = batch_size
            summary[m_key]["stride"] = layer[3]
            if layer[0] == "DWConv":
                summary[m_key]["type"]= "DSCONV"
                K = 1
        else:
            K, C = layer[5], layer[6]
            X, Xo, Y, Yo, R, S = 1,1,1,1,1,1
            summary[m_key]["stride"] = None
        summary[m_key]["dimension_ic"] = (N, K, C, R, S, Y, X)
        summary[m_key]["dimension_oc"] = (N, K, C, R, S, Yo, Xo)
        
    # Save the dataflow
    outfile = model_name + ".m"
    with open(dataflow + ".m", "r") as fd:
        with open("dpt.m", "r") as fdpt:
            with open("out/"+ outfile, "w") as fo:
                first_line = "Network " + model_name + " {\n"
                fo.write(first_line)
                for key, val in summary.items():
                    pc = re.compile("^Conv")
                    pl = re.compile("^Linear")
                    match_pc = pc.match(key)
                    match_pl = pl.match(key)
                    if match_pc or match_pl:
                        fo.write("Layer {} {{\n".format(key))
                        types = val["type"]
                        fo.write("Type: {}\n".format(types))
                        if not match_pl:
                            fo.write("Stride {{ X: {}, Y: {} }}\n".format(*val["stride"]))
                        fo.write("Dimensions {{ K: {}, C: {}, R: {}, S: {}, Y: {}, X: {} }}\n".format(
                            *val["dimension_ic"][1:]))
                        if types == "CONV":
                            fd.seek(0)
                            fo.write(fd.read())
                        else:
                            fdpt.seek(0)
                            fo.write(fdpt.read())
                        fo.write("}\n")
                fo.write("}")
    return summary 
                
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