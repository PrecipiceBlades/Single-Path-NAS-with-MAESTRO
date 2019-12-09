model_name = "MobileNet-V2"
dataflow = "rs"

input_size = (3, 224, 224)
num_classes = 1000
flops = 0
profiled_layers = []
blocks_args = [
      'r2_k3_s22_e6_i16_o24',
      'r3_k3_s22_e6_i24_o32', 
      'r4_k3_s22_e6_i32_o64',
      'r3_k3_s11_e6_i64_o96', 
      'r3_k3_s22_e6_i96_o160',
      'r1_k3_s11_e6_i160_o320',
]

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