from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import InputSpec
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util

import collections
import six
from six.moves import xrange
# import tensorflow as tf
import json

# dstamoulis: definition of masked layer (DepthwiseConv2DMasked)
from pytorch_version.superkernel import *

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'depth_multiplier', 'depth_divisor', 'min_depth', 'search_space',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

# TODO(hongkuny): Consider rewrite an argument class with encoding/decoding.
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

def conv_kernel_initializer(x):
    """
    Initialization for convolutional kernels.

    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.contrib.layers.variance_scaling_initializer uses a truncated normal with
    a corrected standard deviation.

    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    Returns:
      an initialization for the variable
    """
    
    _, out_filters, kernel_height, kernel_width = x.data.size()
    fan_out = int(kernel_height * kernel_width * out_filters)
    x.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))


def dense_kernel_initializer(x):
    """
    Initialization for dense kernels.

    This initialization is equal to
    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',distribution='uniform').
    It is written out explicitly here for clarity.

    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused

    Returns:
      an initialization for the variable
    """

    init_range = 1.0 / np.sqrt(x.data.shape[1])
    x.uniform_(a = -init_range, b = init_range)


def round_filters(filters, global_params):
    """
    Round number of filters based on depth multiplier.
    """
    multiplier = global_params.depth_multiplier
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return new_filters

class MBConvBlock(object):
    """A class of MnasNet/MobileNetV2 Inveretd Residual Bottleneck.

    Attributes:
      has_se: boolean. Whether the block contains a Squeeze and Excitation layer
        inside.
      endpoints: dict. A list of internal tensors.
    """

    def __init__(self, block_args, global_params, layer_runtimes, dropout_rate):
        """
        Initializes a MBConv block.

        Args:
            block_args: BlockArgs, arguments to create a MnasBlock.
            global_params: GlobalParams, a set of global parameters.
        """
        self._block_args = block_args
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        self._channel_axis = 1
        self._spatial_dims = [2, 3]
        self.has_se = (self._block_args.se_ratio is not None) and (
            self._block_args.se_ratio > 0) and (self._block_args.se_ratio <= 1)

        self.endpoints = None
        self.runtimes = layer_runtimes
        self.dropout_rate = dropout_rate

        self._search_space = global_params.search_space
        # Builds the block accordings to arguments.

    def _build(self):
        """
        Builds MBConv block according to the arguments.
        """
        
#       filters: Integer, the dimensionality of the output space
#           (i.e. the number of output filters in the convolution).
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        in_channels = inputs.shape[1]
        
        if self._block_args.expand_ratio != 1:
          # Expansion phase:
            self._expand_conv = nn.Conv2d(
                in_channels=in_channels,  # Not Sure ???
                out_channels=filters,
                kernel_size=1,
                stride=1,
                padding=util.get_same_padding(1, 1),
                bias=False)
            conv_kernel_initializer(self._expand_conv.weight)
            
            self._bn0 = nn.BatchNorm2d(
                num_features=filters,    # No sure ??? 
                eps=self._batch_norm_epsilon, 
                momentum=self._batch_norm_momentum)
            
        kernel_size = self._block_args.kernel_size
        if self._search_space is None:  #  for "default" layers

          # Default depth-wise convolution phase:
            self._depthwise_conv = DepthwiseConv2D(
                nin=filters,             #  Not Sure ???
                kernel_size=kernel_size,
                stride=self._block_args.strides,
                padding='same',
                use_bias=False)
            conv_kernel_initializer(self._depthwise_conv.weight)

        # Learnable Depth-wise convolution Superkernel
        elif self._search_space == 'mnasnet': 
            
            self._depthwise_conv = DepthwiseConv2DMasked(
                kernel_size=kernel_size,
                stride=self._block_args.strides,
                depthwise_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=False,
                runtimes=self.runtimes,
                dropout_rate=self.dropout_rate)
            conv_kernel_initializer(self._depthwise_conv.weight)

        else:
            raise NotImplementedError('DepthConv not defined for %s' % self._search_space)

        self._bn1 = nn.BatchNorm2d(
            num_features=filters,  # No sure ??? 
            eps=self._batch_norm_epsilon, 
            momentum=self._batch_norm_momentum)

        if self.has_se:
            # why would you have SE in the supernet during search?
            assert 1 == 0 
            num_reduced_filters = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            # Squeeze and Excitation layer.
            self._se_reduce = nn.Conv2d(
                in_channels=self._block_args.input_filters,  # Not Sure ???
                out_channels=num_reduced_filters,
                kernel_size=1,
                stride=1,
                padding=util.get_same_padding(1, 1),
                bias=True)
            conv_kernel_initializer(self._se_reduce.weight)
            
            self._se_expand = nn.Conv2d(
                in_channels=num_reduced_filters,           # Not Sure ???
                out_channels=filters,
                kernel_size=1,
                stride=1,
                padding=util.get_same_padding(1, 1),
                bias=True)
            conv_kernel_initializer(self._se_expand.weight)

        # Output phase:
        
        self._project_conv = nn.Conv2d(
            in_channels=filters,                           # Not Sure ???
            out_channels=self._block_args.output_filters,
            kernel_size=1,
            stride=1,
            padding=util.get_same_padding(1, 1),
            bias=True)
        conv_kernel_initializer(self._project_conv.weight)
        
        self._bn2 = nn.BatchNorm2d(
            num_features=self._block_args.output_filter,    # No sure ??? 
            eps=self._batch_norm_epsilon, 
            momentum=self._batch_norm_momentum)

    def _call_se(self, input_tensor):
        """
        Call Squeeze and Excitation layer.
        Args:
          input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.

        Returns:
          A output tensor, which should have the same shape as input.
        """
        se_tensor = torch.mean(input_tensor, axis=self._spatial_dims, keepdims=True)
        se_tensor = self._se_expand(nn.ReLU(self._se_reduce(se_tensor)))
        tf.logging.info('Built Squeeze and Excitation with tensor shape: %s' %
                        (se_tensor.shape))                  # TODO
        return torch.sigmoid(se_tensor) * input_tensor

    def forward(self, inputs, runtime):
        """Implementation of MBConvBlock call().

        Args:
          inputs: the inputs tensor.
          training: boolean, whether the model is constructed for training.

        Returns:
          A output tensor.
        """
        self.inputs = inputs
        self._build()
        
        if self._block_args.expand_ratio != 1:
            x = nn.ReLU(self._bn0(self._expand_conv(inputs)))
        else:
            x = inputs

        x, runtime = self._depthwise_conv(x, runtime)
        x = nn.ReLU(self._bn1(x))

        if self.has_se:
            x = self._call_se(x)

        self.endpoints = {'expansion_output': x}

        x = self._bn2(self._project_conv(x))
        if self._block_args.id_skip:
            if all(s == 1 for s in self._block_args.strides) and \
            (self._block_args.input_filters == self._block_args.output_filters):
                x = torch.add(x, inputs)
        return x, runtime
    
class SinglePathSuperNet(tf.keras.Model):
    """
    class implements tf.keras.Model for SinglePath Supernet with superkernels
    More details: Fig.2 -- Single-Path NAS: https://arxiv.org/abs/(TBD)
    Based on MNasNet search space: https://arxiv.org/abs/1807.11626
    
    """
    def __init__(self, blocks_args=None, global_params=None,dropout_rate=None):
        """Initializes an `SuperNet` instance.

        Args:
          blocks_args: A list of BlockArgs to construct MBConv block modules.
          global_params: GlobalParams, a set of global parameters.

        Raises:
          ValueError: when blocks_args is not specified as a list.
        """
        super(SinglePathSuperNet, self).__init__()
        if not isinstance(blocks_args, list):
            raise ValueError('blocks_args should be a list.')
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.endpoints = None
        self.dropout_rate = dropout_rate

        self._search_space = global_params.search_space

        tf.logging.info('Runtime model parsed')
        assert self._search_space == 'mnasnet' # currently supported one
        lutmodel_filename = "./pixel1_runtime_model.json"
        with open(lutmodel_filename, 'r') as f:
            self._runtime_lut = json.load(f)

    def _build(self):
        """
        Builds the supernet.
        """
        self._blocks = []
        in_channels = self.input.shape[1]
        # Builds blocks.
        for block_args in self._blocks_args:
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params))

            # The first block needs to take care of stride and filter size increase.
            layer_runtimes = [self._runtime_lut[str(len(self._blocks))][str(i)] 
                for i in range(len(self._runtime_lut[str(len(self._blocks))].keys()))]
            self._blocks.append(MBConvBlock(block_args, self._global_params, 
                                        layer_runtimes, self.dropout_rate))
            if block_args.num_repeat > 1:
            # pylint: disable=protected-access
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])
            # pylint: enable=protected-access
            for _ in xrange(block_args.num_repeat - 1):
                layer_runtimes = [self._runtime_lut[str(len(self._blocks))][str(i)] 
                    for i in range(len(self._runtime_lut[str(len(self._blocks))].keys()))] + [0.7] 
                # neglibible (ms) value for skip-op (non-zero handling purposes)
                self._blocks.append(MBConvBlock(block_args, self._global_params, 
                                          layer_runtimes, self.dropout_rate))

        batch_norm_momentum = self._global_params.batch_norm_momentum
        batch_norm_epsilon = self._global_params.batch_norm_epsilon
        channel_axis = 1
    
        filters = round_filters(32, self._global_params)
        # Stem part.
        self._conv_stem = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=3,
            stride=2,
            padding=util.get_same_padding(3, 2),
            bias=True)
        conv_kernel_initializer(self._conv_stem.weight)
        
        self._bn0 = nn.BatchNorm2d(
            num_features=round_filters(32, self._global_params), 
            eps=batch_norm_epsilon, 
            momentum=batch_norm_momentum)
        
        # Head part.
        self._conv_head = nn.Conv2d(
            in_channels=filters,
            out_channels=1280,
            kernel_size=1,
            stride=1,
            padding=util.get_same_padding(1, 1),
            bias=False)
        conv_kernel_initializer(self._conv_head.weight)
        
        self._bn1 = nn.BatchNorm2d(
            num_features=1280, 
            eps=batch_norm_epsilon, 
            momentum=batch_norm_momentum)
        
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)   #  Not sure. See https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721/15
                                                      #  try nn.AdaptiveAvgPool2d(1) works fine when x.shape=N * C * H * W 
        self._fc = nn.Linear(
            in_features=1280,
            out_features=self._global_params.num_classes)
        dense_kernel_initializer(self._fc.weight)
        
        if self._global_params.dropout_rate > 0:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
        else:
            self._dropout = None

    def forward(self, inputs, training=True):
        """Implementation of SuperNet call().

        Args:
          inputs: input tensors.
          training: boolean, whether the model is constructed for training.

        Returns:
          output tensors.
        """
        outputs = None
        self.endpoints = {}
        self.indicators = {}
        self.input = inputs
        self._build()

        # rest of runtime (i.e., stem, head, logits, block0, block21)
        total_runtime = 19.5999

        # Calls Stem layers
#         with tf.variable_scope('mnas_stem'):
        outputs = nn.ReLU(self._bn0(self._conv_stem(inputs)))
#         tf.logging.info('Built stem layers with output shape: %s' % outputs.shape)
        self.endpoints['stem'] = outputs
        # Calls blocks.
        for idx, block in enumerate(self._blocks):
#             with tf.variable_scope('mnas_blocks_%s' % idx):
            outputs, total_runtime = block(outputs, total_runtime)
            self.endpoints['block_%s' % idx] = outputs
            # the indicator decisions 
            if block._depthwise_conv.custom:
                self.indicators['block_%s' % idx] = {
                    'd5x5': block._depthwise_conv.d5x5,
                    'd50c': block._depthwise_conv.d50c,
                    'd100c': block._depthwise_conv.d100c}

            if block.endpoints:
                for k, v in six.iteritems(block.endpoints):
                    self.endpoints['block_%s/%s' % (idx, k)] = v
        # Calls final layers and returns logits.
#         with tf.variable_scope('mnas_head'):
        outputs = nn.ReLU(self._bn1(self._conv_head(outputs)))
        outputs = self._avg_pooling(outputs)
        if self._dropout:
            outputs = self._dropout(outputs)
        outputs = self._fc(outputs)
        self.endpoints['head'] = outputs

        return outputs, total_runtime