model_name = "ProxylessNAS"
dataflow = "rs"

input_size = (3, 224, 224)
num_classes = 1000
flops = 0
profiled_layers = []
blocks_args = [
      'r1_k5_s22_e3.0_i16_o32', 
      'r1_k3_s11_e3.0_i32_o32',
      'r1_k7_s22_e3.0_i32_o40', 
      'r1_k3_s11_e3.0_i40_o40',
      'r1_k5_s11_e3.0_i40_o40',
      'r1_k5_s11_e3.0_i40_o40',
      'r1_k7_s22_e6.0_i40_o80', 
      'r1_k5_s11_e3.0_i80_o80',
      'r1_k5_s11_e3.0_i80_o80', 
      'r1_k5_s11_e3.0_i80_o80',
      'r1_k5_s11_e6.0_i80_o96',
      'r1_k5_s11_e3.0_i96_o96', 
      'r1_k5_s11_e3.0_i96_o96',
      'r1_k5_s11_e3.0_i96_o96', 
      'r1_k7_s22_e6.0_i96_o192',
      'r1_k7_s11_e6.0_i192_o192',
      'r1_k7_s11_e3.0_i192_o192',
      'r1_k7_s11_e3.0_i192_o192', 
      'r1_k7_s11_e6.0_i192_o320',  
]

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