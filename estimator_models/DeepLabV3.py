"""DeepLab v3 models based on slim library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.slim.python.slim.nets import resnet_utils

from estimator_models.estimator_model_utils import get_estimator_spec, get_preprocess_by_frontend
from builders import frontend_builder

_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4


def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
  """Atrous Spatial Pyramid Pooling.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    is_training: A boolean denoting whether the input is for training.
    depth: The depth of the ResNet unit output.

  Returns:
    The atrous spatial pyramid pooling output.
  """
  with tf.variable_scope("aspp"):
    if output_stride not in [8, 16]:
      raise ValueError('output_stride must be either 8 or 16.')

    atrous_rates = [6, 12, 18]
    if output_stride == 8:
      atrous_rates = [2*rate for rate in atrous_rates]

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      with arg_scope([layers.batch_norm], is_training=is_training):
        inputs_size = tf.shape(inputs)[1:3]
        # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # the rates are doubled when output stride = 8.
        conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
        conv_3x3_1 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
        conv_3x3_2 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
        conv_3x3_3 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

        # (b) the image-level features
        with tf.variable_scope("image_level_features"):
          # global average pooling
          image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
          # 1x1 convolution with 256 filters( and batch normalization)
          image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
          # bilinearly upsample features
          image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

        return net


def deeplab_v3_generator(num_classes,
                         output_stride,
                         frontend,
                         pretrained_dir,
                         batch_norm_decay,
                         weight_decay):
  """Generator for DeepLab v3 models.

  Args:
    num_classes: The number of possible classes for image classification.
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    frontend: The architecture of base Resnet building block.
    pretrained_dir: The path to the directory that contains pre-trained models.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    weight_decay: 

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the DeepLab v3 model.
  """

  if batch_norm_decay is None:
    batch_norm_decay = _BATCH_NORM_DECAY
  
  if weight_decay is None:
    weight_decay = _WEIGHT_DECAY

  def model(inputs, is_training):
    logits, end_points, frontend_scope, init_fn = frontend_builder.build_frontend(inputs, frontend,
                                                                                  pretrained_dir=pretrained_dir,
                                                                                  is_training=is_training,
                                                                                  weight_decay=weight_decay, 
                                                                                  batch_norm_decay=batch_norm_decay,
                                                                                  output_stride=output_stride,)
    with tf.contrib.slim.arg_scope(
      [tf.contrib.slim.conv2d],
      weights_regularizer=tf.contrib.slim.l2_regularizer(weight_decay)):
      inputs_size = tf.shape(inputs)[1:3]
      net = end_points[frontend_scope + '/block4']
      net = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)
      with tf.variable_scope("upsampling_logits"):
        net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
        logits = tf.image.resize_bilinear(net, inputs_size, name='upsample')
      return logits, init_fn

  return model


def get_deeplabv3_model_fn(num_classes, 
                           frontend="ResNet101", 
                           pretrained_dir="/ssd/zhangyiyang/data/slim"):
  def deeplabv3_model_fn(features, labels, mode, params):
    _preprocess = get_preprocess_by_frontend(frontend=frontend)
    preprocessed_image = _preprocess(features['image'])

    network = deeplab_v3_generator(num_classes,
                                  params['output_stride'],
                                  frontend,
                                  pretrained_dir,
                                  _BATCH_NORM_DECAY,
                                  params['weight_decay'])

    logits, init_fn = network(preprocessed_image, (mode == tf.estimator.ModeKeys.TRAIN))
    return get_estimator_spec(mode, logits, init_fn, labels=labels, num_classes=num_classes, params=params,
                              features=features, )

  return deeplabv3_model_fn
