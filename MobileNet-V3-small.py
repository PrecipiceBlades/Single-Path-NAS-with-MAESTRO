model_name = "MobileNet-V3"
dataflow = "rs"

input_size = (3, 224, 224)
num_classes = 1000
flops = 0
profiled_layers = []
blocks_args = [
      'r1_k3_s22_e1.0_i16_o16_se_0.25',
      'r1_k3_s22_e4.5_i16_o24', 
      'r1_k3_s11_e3.7_i24_o24',
      'r1_k5_s22_e4.0_i24_o40_se_0.25', 
      'r1_k5_s11_e6.0_i40_o40_se_0.25',
      'r1_k5_s11_e6.0_i40_o40_se_0.25',
      'r1_k5_s11_e3.0_i40_o48_se_0.25',
      'r1_k5_s11_e3.0_i48_o48_se_0.25', 
      'r1_k5_s22_e6.0_i48_o96_se_0.25',
      'r1_k5_s11_e6.0_i96_o96_se_0.25', 
      'r1_k5_s11_e6.0_i96_o96_se_0.25',
]

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