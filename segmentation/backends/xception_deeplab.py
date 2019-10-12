import tensorflow as tf

layers = tf.keras.layers


def SepConv_BN(x, filters, prefix,
               stride=1, kernel_size=3, rate=1,
               depth_activation=False, epsilon=1e-3):
    """
    SepConv with BN between depthwise & pointwise.
    Optionally add activation after BN
    Implements right "same" padding for even kernel sizes
    Args:
        x: input tensor
        filters: num of filters in pointwise convolution
        prefix: prefix before name
        stride: stride at depthwise conv
        kernel_size: kernel size for depthwise convolution
        rate: atrous rate for depthwise convolution
        depth_activation: flag to use activation between depthwise & poinwise
        epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = layers.Activation('relu')(x)
    x = layers.DepthwiseConv2D((kernel_size, kernel_size),
                               strides=(stride, stride),
                               dilation_rate=(rate, rate),
                               padding=depth_padding, use_bias=False,
                               name=prefix + '_depthwise')(x)
    x = layers.BatchNormalization(
        name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (1, 1), padding='same',
                      use_bias=False, name=prefix + '_pointwise')(x)
    x = layers.BatchNormalization(
        name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = layers.Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return layers.Conv2D(filters,
                             (kernel_size, kernel_size),
                             strides=(stride, stride),
                             padding='same', use_bias=False,
                             dilation_rate=(rate, rate),
                             name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = layers.ZeroPadding2D((pad_beg, pad_end))(x)
        return layers.Conv2D(filters,
                             (kernel_size, kernel_size),
                             strides=(stride, stride),
                             padding='valid', use_bias=False,
                             dilation_rate=(rate, rate),
                             name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """
    Basic building block of modified Xception network
    Args:
        inputs: input tensor
        depth_list: number of filters in each SepConv layer.
            len(depth_list) == 3
        prefix: prefix before name
        skip_connection_type: one of {'conv','sum','none'}
        stride: stride at last depthwise conv
        rate: atrous rate for depthwise convolution
        depth_activation: flag to use activation between depthwise & pointwise
        return_skip: flag to return additional tensor after
            2 SepConvs for decoder
    """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = layers.BatchNormalization(
            name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def Xception(input_tensor=None,
             input_shape=(512, 512, 3),
             OS=16,
             activation=None,
             ):
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        img_input = input_tensor

    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)

    # entry flow
    x = layers.Conv2D(32, (3, 3), strides=(2, 2),
                      name='entry_flow_conv1_1',
                      use_bias=False, padding='same')(img_input)
    x = layers.BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = layers.Activation('relu')(x)

    x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    x = layers.BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = layers.Activation('relu')(x)

    x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                        skip_connection_type='conv', stride=2,
                        depth_activation=False)
    x, skip = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                              skip_connection_type='conv', stride=2,
                              depth_activation=False, return_skip=True)

    x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                        skip_connection_type='conv',
                        stride=entry_block3_stride,
                        depth_activation=False)

    # middle flow
    for i in range(16):
        x = _xception_block(x, [728, 728, 728],
                            'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type='sum', stride=1,
                            rate=middle_block_rate,
                            depth_activation=False)

    # exit flow
    x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                        skip_connection_type='conv', stride=1,
                        rate=exit_block_rates[0],
                        depth_activation=False)
    x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                        skip_connection_type='none', stride=1,
                        rate=exit_block_rates[1],
                        depth_activation=True)
    model = tf.keras.Model(img_input, [x, skip], name='xception')
    return model
