import torch
import torchvision.models as models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import csv
from util import *

def profile_MnasNet(dataflow):
    model_name = "MnasNet-A1"
    print("="*50)
    print("Profiling model: ", model_name)
    print("="*50)
    input_size = (3, 224, 224)
    num_classes = 1000
    flops = 0
    profiled_layers = []
    blocks_args = []
    with open("data/" + model_name + ".csv", mode='r') as model_file:
        model_reader = csv.reader(model_file, delimiter=',')
        for row in model_reader:
            blocks_args += row

    # Stem part
    layer_type, kernel_size, stride, out_channels = "Conv", 3, (2, 2), 32
    output_size, nb_params, R, S, flop = \
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
    profiled_layers.append(tuple((layer_type, input_size, output_size, stride, nb_params, R, S)))
    flops += flop

    # MnasNet-A1: SepConv
    layer_type, input_size, kernel_size, stride, out_channels = "SepConv", output_size, 3, (1, 1), 16
    output_size1, nb_params1, R1, S1, output_size, nb_params2, R2, S2, flop = \
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
    profiled_layers.append(tuple(("DWConv", input_size, output_size1, stride, nb_params1, R1, S1)))
    profiled_layers.append(tuple(("Conv", output_size1, output_size, stride, nb_params2, R2, S2)))
    flops += flop

    # Backbone part
    for blocks_string in blocks_args:
        layers, output_size, flop = profile_blockargs(blocks_string, output_size)
        profiled_layers += layers
        flops += flop

    # Head part
    layer_type, input_size, kernel_size, stride, out_channels = "Conv", output_size, 1, (1, 1), 1280
    output_size, nb_params, R, S, flop = \
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
    profiled_layers += [tuple((layer_type, input_size, output_size, stride, nb_params, R, S))]
    flops += flop

    layer_type, input_size, out_features = "Linear", output_size, num_classes
    output_size, nb_params, R, S, flop = get_linear_output_size_and_nb_param(input_size, out_features)
    profiled_layers += [tuple((layer_type, input_size, output_size, None, nb_params, R, S))]
    flops += flop

    print("Total number of flops:", flops)
    summary = make_summary(profiled_layers, dataflow=dataflow, model_name=model_name)
    # Get number of parameters
    layer_names = list(summary.keys())
    params = list(map(lambda x: int(summary[x]['nb_params']), layer_names))
    print("Total number of parameters:", sum(params))
    
def profile_MobileNetV2(dataflow):
    model_name = "MobileNet-V2"
    print("="*50)
    print("Profiling model: ", model_name)
    print("="*50)
    input_size = (3, 224, 224)
    num_classes = 1000
    flops = 0
    profiled_layers = []
    blocks_args = []
    with open("data/" + model_name + ".csv", mode='r') as model_file:
        model_reader = csv.reader(model_file, delimiter=',')
        for row in model_reader:
            blocks_args += row

    # Stem part
    layer_type, kernel_size, stride, out_channels = "Conv", 3, (2, 2), 32
    output_size, nb_params, R, S, flop =\
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
    profiled_layers.append(tuple((layer_type, input_size, output_size, stride, nb_params, R, S)))
    flops += flop

    # Mobilenet-V2: 
    layer_type, input_size, kernel_size, stride, out_channels = "SepConv", output_size, 3, (1, 1), 16
    output_size1, nb_params1, R1, S1, output_size, nb_params2, R2, S2, flop = \
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
    profiled_layers.append(tuple(("DWConv", input_size, output_size1, stride, nb_params1, R1, S1)))
    profiled_layers.append(tuple(("Conv", output_size1, output_size, stride, nb_params2, R2, S2)))
    flops += flop

    # Backbone part
    for blocks_string in blocks_args:
        layers, output_size, flop = profile_blockargs(blocks_string, output_size)
        profiled_layers += layers
        flops += flop

    # Head part
    layer_type, input_size, kernel_size, stride, out_channels = "Conv", output_size, 1, (1, 1), 1280
    output_size, nb_params, R, S, flop = \
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
    profiled_layers += [tuple((layer_type, input_size, output_size, stride, nb_params, R, S))]
    flops += flop

    layer_type, input_size, out_features = "Linear", output_size, num_classes
    output_size, nb_params, R, S, flop = get_linear_output_size_and_nb_param(input_size, out_features)
    profiled_layers += [tuple((layer_type, input_size, output_size, None, nb_params, R, S))]
    flops += flop

    print("Total number of flops:", flops)

    summary = make_summary(profiled_layers, dataflow=dataflow, model_name=model_name)
    # Get number of parameters
    layer_names = list(summary.keys())
    params = list(map(lambda x: int(summary[x]['nb_params']), layer_names))
    print("Total number of parameters:", sum(params))
    
def profile_MobileNetV3_large(dataflow):
    model_name = "MobileNet-V3-large"
    print("="*50)
    print("Profiling model: ", model_name)
    print("="*50)
    input_size = (3, 224, 224)
    num_classes = 1000
    flops = 0
    profiled_layers = []
    blocks_args = []
    with open("data/" + model_name + ".csv", mode='r') as model_file:
        model_reader = csv.reader(model_file, delimiter=',')
        for row in model_reader:
            blocks_args += row

    # Stem part
    layer_type, kernel_size, stride, out_channels = "Conv", 3, (2, 2), 16
    output_size, nb_params, R, S, flop = \
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
    profiled_layers.append(tuple((layer_type, input_size, output_size, stride, nb_params, R, S)))
    flops += flop

    # MobileNet-V3-large: None

    # Backbone part
    for blocks_string in blocks_args:
        layers, output_size, flop = profile_blockargs(blocks_string, output_size, use_bias=False)
        profiled_layers += layers
        flops += flop

    # Head part
    layer_type, input_size, kernel_size, stride, out_channels = "Conv", output_size, 1, (1, 1), 960
    output_size, nb_params, R, S, flop = \
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
    profiled_layers += [tuple((layer_type, input_size, output_size, stride, nb_params, R, S))]
    flops += flop

    layer_type, input_size, out_features = "Linear", output_size, 1280
    output_size, nb_params, R, S, flop = get_linear_output_size_and_nb_param(input_size, out_features, use_pool=True)
    profiled_layers += [tuple((layer_type, input_size, output_size, None, nb_params, R, S))]
    flops += flop

    layer_type, input_size, out_features = "Linear", output_size, num_classes
    output_size, nb_params, R, S, flop = get_linear_output_size_and_nb_param(input_size, out_features, use_pool=False)
    profiled_layers += [tuple((layer_type, input_size, output_size, None, nb_params, R, S))]
    flops += flop

    print("Total number of flops:", flops)

    summary = make_summary(profiled_layers, dataflow=dataflow, model_name=model_name)
    # Get number of parameters
    layer_names = list(summary.keys())
    params = list(map(lambda x: int(summary[x]['nb_params']), layer_names))
    print("Total number of parameters:", sum(params))
    
def profile_MobileNetV3_small(dataflow):    
    model_name = "MobileNet-V3-small"
    print("="*50)
    print("Profiling model: ", model_name)
    print("="*50)
    input_size = (3, 224, 224)
    num_classes = 1000
    flops = 0
    profiled_layers = []
    blocks_args = []
    with open("data/" + model_name + ".csv", mode='r') as model_file:
        model_reader = csv.reader(model_file, delimiter=',')
        for row in model_reader:
            blocks_args += row

    # Stem part
    layer_type, kernel_size, stride, out_channels = "Conv", 3, (2, 2), 16
    output_size, nb_params, R, S, flop = \
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
    profiled_layers.append(tuple((layer_type, input_size, output_size, stride, nb_params, R, S)))
    flops += flop

    # MobileNet-V3: None

    # Backbone part
    for blocks_string in blocks_args:
        layers, output_size, flop = profile_blockargs(blocks_string, output_size, use_bias=False)
        profiled_layers += layers
        flops += flop

    # Head part
    layer_type, input_size, kernel_size, stride, out_channels = "Conv", output_size, 1, (1, 1), 576
    output_size, nb_params, R, S, flop = \
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
    profiled_layers += [tuple((layer_type, input_size, output_size, stride, nb_params, R, S))]
    flops += flop

    # MobileNet-V3-small: SE
    input_size, expansion = output_size, 0.25
    output_size1, nb_params1, R1, S1, output_size, nb_params2, R2, S2, flop = \
        get_se_output_and_params_and_flops(input_size, expansion=expansion, bias=False)
    layers.append(tuple(("Linear", input_size, output_size1, None, nb_params1, R1, S1)))
    layers.append(tuple(("Linear", output_size1, output_size, None, nb_params2, R2, S2)))
    flops += flop

    layer_type, input_size, out_features = "Linear", output_size, 1024
    output_size, nb_params, R, S, flop = get_linear_output_size_and_nb_param(input_size, out_features, use_pool=True)
    profiled_layers += [tuple((layer_type, input_size, output_size, None, nb_params, R, S))]
    flops += flop

    layer_type, input_size, out_features = "Linear", output_size, num_classes
    output_size, nb_params, R, S, flop = get_linear_output_size_and_nb_param(input_size, out_features, use_pool=False)
    profiled_layers += [tuple((layer_type, input_size, output_size, None, nb_params, R, S))]
    flops += flop

    print("Total number of flops:", flops)

    summary = make_summary(profiled_layers, dataflow=dataflow, model_name=model_name)
    # Get number of parameters
    layer_names = list(summary.keys())
    params = list(map(lambda x: int(summary[x]['nb_params']), layer_names))
    print("Total number of parameters:", sum(params))
    
def profile_ProxylessNAS(dataflow):
    model_name = "ProxylessNAS"
    print("="*50)
    print("Profiling model: ", model_name)
    print("="*50)
    input_size = (3, 224, 224)
    num_classes = 1000
    flops = 0
    profiled_layers = []
    blocks_args = []
    with open("data/" + model_name + ".csv", mode='r') as model_file:
        model_reader = csv.reader(model_file, delimiter=',')
        for row in model_reader:
            blocks_args += row
    # Stem part
    layer_type, kernel_size, stride, out_channels = "Conv", 3, (2, 2), 32
    output_size, nb_params, R, S, flop = \
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
    profiled_layers.append(tuple((layer_type, input_size, output_size, stride, nb_params, R, S)))
    flops += flop

    # ProxylessNAS: 
    layer_type, input_size, kernel_size, stride, out_channels = "SepConv", output_size, 3, (1, 1), 16
    output_size1, nb_params1, R1, S1, output_size, nb_params2, R2, S2, flop = \
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
    profiled_layers.append(tuple(("DWConv", input_size, output_size1, stride, nb_params1, R1, S1)))
    profiled_layers.append(tuple(("Conv", output_size1, output_size, stride, nb_params2, R2, S2)))
    flops += flop

    # Backbone part
    for blocks_string in blocks_args:
        layers, output_size, flop = profile_blockargs(blocks_string, output_size)
        profiled_layers += layers
        flops += flop

    # Head part
    layer_type, input_size, kernel_size, stride, out_channels = "Conv", output_size, 1, (1, 1), 1280
    output_size, nb_params, R, S, flop = \
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels, use_bn=True)
    profiled_layers += [tuple((layer_type, input_size, output_size, stride, nb_params, R, S))]
    flops += flop

    layer_type, input_size, out_features = "Linear", output_size, num_classes
    output_size, nb_params, R, S, flop = get_linear_output_size_and_nb_param(input_size, out_features, use_pool=True)
    profiled_layers += [tuple((layer_type, input_size, output_size, None, nb_params, R, S))]
    flops += flop

    print("Total number of flops:", flops)

    summary = make_summary(profiled_layers, dataflow=dataflow, model_name=model_name)
    # Get number of parameters
    layer_names = list(summary.keys())
    params = list(map(lambda x: int(summary[x]['nb_params']), layer_names))
    print("Total number of parameters:", sum(params))
    
def profile_SinglepathNAS(dataflow):
    model_name = "SinglepathNAS"
    print("="*50)
    print("Profiling model: ", model_name)
    print("="*50)
    input_size = (3, 224, 224)
    num_classes = 1000
    flops = 0
    profiled_layers = []
    blocks_args = []
    with open("data/" + model_name + ".csv", mode='r') as model_file:
        model_reader = csv.reader(model_file, delimiter=',')
        for row in model_reader:
            blocks_args += row
    # Stem part
    layer_type, kernel_size, stride, out_channels = "Conv", 3, (2, 2), 32
    output_size, nb_params, R, S, flop = \
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
    profiled_layers.append(tuple((layer_type, input_size, output_size, stride, nb_params, R, S)))
    flops += flop

    # SinglepathNAS: None

    # Backbone part
    for blocks_string in blocks_args:
        layers, output_size, flop = profile_blockargs(blocks_string, output_size)
        profiled_layers += layers
        flops += flop

    # Head part
    layer_type, input_size, kernel_size, stride, out_channels = "Conv", output_size, 1, (1, 1), 1280
    output_size, nb_params, R, S, flop = \
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
    profiled_layers += [tuple((layer_type, input_size, output_size, stride, nb_params, R, S))]
    flops += flop

    layer_type, input_size, out_features = "Linear", output_size, num_classes
    output_size, nb_params, R, S, flop = get_linear_output_size_and_nb_param(input_size, out_features, use_pool=True)
    profiled_layers += [tuple((layer_type, input_size, output_size, None, nb_params, R, S))]
    flops += flop

    print("Total number of flops:", flops)

    summary = make_summary(profiled_layers, dataflow=dataflow, model_name=model_name)
    # Get number of parameters
    layer_names = list(summary.keys())
    params = list(map(lambda x: int(summary[x]['nb_params']), layer_names))
    print("Total number of parameters:", sum(params))
