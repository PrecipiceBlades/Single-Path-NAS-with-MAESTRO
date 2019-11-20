import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from maestro_summary import summary
import re
from collections import OrderedDict
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def profile_blockargs(blocks_string, input_size):
    assert isinstance(blocks_string, str)
    ops = blocks_string.split('_')
    options = {}
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
    options["expansion"]      = int(options["e"])

    output_size = input_size
    layers = []
    for key, r in enumerate(range(options["num_repeat"])):
        layer_type1, input_size, kernel_size, stride, expansion, out_channels = "conv2d", output_size, 1, (1, 1), options["expansion"], options["input_filters"]
        output_size, nb_params, R, S = util.get_conv_output_size_and_nb_param(input_size, layer_type1, kernel_size, stride, expansion, out_channels)
        layers.append(tuple((layer_type1, input_size, output_size, stride, nb_params, R, S)))
        layer_type2, input_size, kernel_size, stride = "depthwise", output_size, options["kernel_size"], options["stride"]
        output_size, nb_params, R, S = util.get_conv_output_size_and_nb_param(input_size, layer_type2, kernel_size, stride)
        layers.append(tuple((layer_type2, input_size, output_size, stride, nb_params, R, S)))
        layer_type3, input_size, kernel_size, stride, out_channels = "conv2d", output_size, 1, (1, 1), options["output_filters"]
        output_size, nb_params, R, S = util.get_conv_output_size_and_nb_param(input_size, layer_type3, kernel_size, stride, out_channels=out_channels)
        layers.append(tuple((layer_type3, input_size, output_size, stride, nb_params, R, S)))
        if key == 0:
            options["stride"] = 1
    return layers, output_size

profiled_layers = []
input_size = (3, 224, 224)
num_classes = 32
blocks_args = [
      'r1_k3_s11_e1_i32_o16_noskip', 
      'r4_k5_s22_e6_i16_o24',
      'r4_k5_s22_e6_i24_o40', 
      'r4_k5_s22_e6_i40_o80',
      'r4_k5_s11_e6_i80_o96', 
      'r4_k5_s22_e6_i96_o192',
      'r1_k3_s11_e6_i192_o320_noskip'
]

# Stem part
layer_type, kernel_size, stride, out_channels = "conv2d", 3, (2, 2), 32
output_size, nb_params, R, S = util.get_conv_output_size_and_nb_param(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
profiled_layers.append(tuple((layer_type, input_size, output_size, stride, nb_params, R, S)))

# Backbone part

for blocks_string in blocks_args:
    layers, output_size = profile_blockargs(blocks_string, output_size)
    profiled_layers += layers
    
# Head part

layer_type, input_size, kernel_size, stride, out_channels = "conv2d", output_size, 1, (1, 1), 1280
output_size, nb_params, R, S = util.get_conv_output_size_and_nb_param(input_size, layer_type, kernel_size, stride, out_channels=out_channels)
profiled_layers += [tuple((layer_type, input_size, output_size, stride, nb_params, R, S))]

layer_type, input_size, in_features, out_features = "linear", (1280, ), 1280, num_classes
output_size, nb_params, R, S = util.get_linear_output_size_and_nb_param(in_features, out_features)
profiled_layers += [tuple((layer_type, input_size, output_size, stride, nb_params, R, S))]

summary = OrderedDict()
for key, layer in enumerate(profiled_layers):
    batch_size = -1
    m_key = "%s-%i" % (layer[0], key+1)
    summary[m_key] = OrderedDict()
    summary[m_key]["type"] = "CONV"
    summary[m_key]["input_shape"] = [batch_size] + list(layer[1])
    summary[m_key]["output_shape"] = [batch_size] + list(layer[2])
    summary[m_key]["trainable"] = True
    summary[m_key]["nb_params"] = layer[4]
    if layer[0] != "linear":
        C, Y, X = layer[1]
        K, Yo, Xo = layer[2]
        R, S = layer[5], layer[6]
        N = batch_size
        summary[m_key]["stride"] = layer[3]
        if layer[0] == "depthwise":
            summary[m_key]["type"]= "DSCONV"
            K = 1
    else:
        K, C = layer[5], layer[6]
        X, Xo, Y, Yo, R, S = 1,1,1,1,1,1
        summary[m_key]["stride"] = None
    summary[m_key]["dimension_ic"] = (N, K, C, R, S, Y, X)
    summary[m_key]["dimension_oc"] = (N, K, C, R, S, Yo, Xo)

print(summary)
