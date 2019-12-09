import torch
import torchvision.models as models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import csv
from util import *
from model import *

parser = argparse.ArgumentParser(description='PyTorch Model to MAESTRO')
parser.add_argument('--dataflow', type=str, default='rs',
                    help='rs, os, ws or dla')
parser.add_argument('--model', type=str, default='MnasNet-A1',
                    help='MnasNet-A1, MobileNet-V2, MobileNet-V3-large, MobileNet-V3-small, ProxylessNAS, SinglepathNAS')
parser.add_argument('--blockargs', type=list, default=None,
                    help='You need to specify block arguments as well as stem / head structures yourself!')

args = parser.parse_args()

dataflow = args.dataflow
model = args.model

assert model == "MnasNet-A1" or model == "MobileNet-V2" or model == "MobileNet-V3-large" \
    or model == "MobileNet-V3-small" or model == "ProxylessNAS" or model == "SinglepathNAS"

if args.blockargs == None:
    if args.model == "MnasNet-A1":
        profile_MnasNet(dataflow)
    elif args.model == "MobileNet-V2":
        profile_MobileNetV2(dataflow)
    elif args.model == "MobileNet-V3-large":
        profile_MobileNetV3_large(dataflow)
    elif args.model == "MobileNet-V3-small":
        profile_MobileNetV3_small(dataflow)
    elif args.model == "ProxylessNAS":
        profile_ProxylessNAS(dataflow)
    else:
        profile_SinglepathNAS(dataflow)
else:
    
    ############################################
    #  You must define your models here first  #
    ############################################
    
    input_size = (3, 224, 224)
    num_classes = 1000
    flops = 0
    profiled_layers = []
    
    # Stem part
    layer_type, kernel_size, stride, out_channels = "Conv", 3, (2, 2), 32
    output_size, nb_params, R, S, flop = \
        get_conv_output_and_params_and_flops(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
    profiled_layers.append(tuple((layer_type, input_size, output_size, stride, nb_params, R, S)))
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
    output_size, nb_params, R, S, flop = get_linear_output_size_and_nb_param(input_size, out_features, use_pool=True)
    profiled_layers += [tuple((layer_type, input_size, output_size, None, nb_params, R, S))]
    flops += flop

    print("Total number of flops:", flops)
    
    summary = make_summary(profiled_layers, dataflow=dataflow, model_name=model_name)
    # Get number of parameters
    layer_names = list(summary.keys())
    params = list(map(lambda x: int(summary[x]['nb_params']), layer_names))
    print("Total number of parameters:", sum(params))
