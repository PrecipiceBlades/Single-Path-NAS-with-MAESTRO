# Test code for Pytorch & Tensorflow equivalance for Depthwise Convolution

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import InputSpec

padding = util.get_padding(kernel_size=3, stride=1)
inputs = torch.ones(1,2,5,5)
depth_filters = torch.ones(2,1,3,3)

kernel_size = 3
stride = 1
nin = 2
padding = util.get_padding(kernel_size=kernel_size, stride=stride) # "same" padding
depthwise = nn.Conv2d(in_channels=nin, out_channels=nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin, bias=False)
depthwise.weight = nn.Parameter(data=depth_filters)
print("Pytorch_version: ", depthwise(inputs))

inputs = tf.ones([1,2,5,5])

model = tf.keras.models.Sequential([tf.keras.layers.DepthwiseConv2D(
                kernel_size=[3, 3],
                strides=[1, 1],
                depthwise_initializer=tf.ones_initializer(),
                padding='same',
                data_format='channels_first',
                use_bias=False)])(inputs)

with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    print("Tensorflow version: ", model.eval())
