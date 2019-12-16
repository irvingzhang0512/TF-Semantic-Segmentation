import tensorflow as tf

layers = tf.keras.layers
backend = tf.keras.backend


def Xception(xception_type='xception',
             input_shape=None,
             dilation=None,
             output_stages='c5',):
    if dilation is None:
        strides = [2, 2]
    else:
        strides = [2 if dilation[0] == 1 else 1] + \
            [2 if dilation[1] == 1 else 1]
    if xception_type == 'xception':
        rm_pool = False
        num_middle_flow = 8
    elif xception_type == 'xception-deeplab':
        rm_pool = True
        num_middle_flow = 16
    else:
        raise ValueError('unknown xception type %d' % xception_type)

    img_input = layers.Input(shape=input_shape)
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    # entry flow part1
    x = layers.Conv2D(32, (3, 3),
                      strides=(2, 2),
                      use_bias=False,
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv2D(64, (3, 3), padding='same',
                      use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    # entry flow part2
    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv1')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual])
    c1 = x

    # entry flow part3
    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv1')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block3_sepconv2_bn')(x)

    if rm_pool:
        x = layers.Activation('relu', name='block3_sepconv3_act')(x)
        x = layers.SeparableConv2D(256, (3, 3),
                                   strides=(2, 2),
                                   padding='same',
                                   use_bias=False,
                                   name='block3_sepconv3')(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name='block3_sepconv3_bn')(x)
    else:
        x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same',
                                name='block3_pool')(x)

    x = layers.add([x, residual])
    c2 = x

    # entry flow part 4
    residual = layers.Conv2D(728, (1, 1),
                             strides=strides[0],
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv2')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block4_sepconv2_bn')(x)

    if rm_pool:
        x = layers.Activation('relu', name='block4_sepconv3_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   strides=strides[0],
                                   padding='same',
                                   use_bias=False,
                                   name='block4_sepconv3')(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name='block4_sepconv3_bn')(x)
    else:
        x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same',
                                name='block4_pool')(x)
    x = layers.add([x, residual])
    c3 = x

    # middle flow
    for i in range(num_middle_flow):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])
    c4 = x

    # exit flow part1
    residual = layers.Conv2D(1024, (1, 1), strides=strides[1],
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    id = 5 + num_middle_flow
    x = layers.Activation(
        'relu', name='block{id}_sepconv1_act'.format(id=id))(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block{id}_sepconv1'.format(id=id))(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block{id}_sepconv1_bn'.format(id=id))(x)
    x = layers.Activation(
        'relu', name='block{id}_sepconv2_act'.format(id=id))(x)
    x = layers.SeparableConv2D(1024, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block{id}_sepconv2'.format(id=id))(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block{id}_sepconv2_bn'.format(id=id))(x)

    if rm_pool:
        x = layers.Activation(
            'relu', name='block{id}_sepconv3_act'.format(id=id))(x)
        x = layers.SeparableConv2D(1024, (3, 3),
                                   strides=strides[1],
                                   padding='same',
                                   use_bias=False,
                                   name='block{id}_sepconv3'.format(id=id))(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name='block{id}_sepconv3_bn'.format(id=id))(x)
    else:
        x = layers.MaxPooling2D((3, 3),
                                strides=(2, 2),
                                padding='same',
                                name='block{id}_pool'.format(id=id))(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(1536, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block{id}_sepconv1'.format(id=id + 1))(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block{id}_sepconv1_bn'.format(id=id + 1))(x)
    x = layers.Activation(
        'relu', name='block{id}_sepconv1_act'.format(id=id + 1))(x)

    if xception_type == 'xception-deepLab':
        x = layers.SeparableConv2D(1536, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name='block{id}_sepconv1_1'.format(id=id + 1))(x)
        x = layers.BatchNormalization(
            axis=channel_axis, name='block{id}_sepconv1_1_bn'.format(id=id + 1))(x)
        x = layers.Activation(
            'relu', name='block{id}_sepconv1_1_act'.format(id=id + 1))(x)

    x = layers.SeparableConv2D(2048, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block{id}_sepconv2'.format(id=id + 1))(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='block{id}_sepconv2_bn'.format(id=id + 1))(x)
    x = layers.Activation(
        'relu', name='block{id}_sepconv2_act'.format(id=id + 1))(x)

    c5 = x

    output_dict = {
        'c1': c1,
        'c2': c2,
        'c3': c3,
        'c4': c4,
        'c5': c5,
    }

    if not isinstance(output_stages, list):
        outputs = output_dict[output_stages]
    else:
        outputs = [output_dict[ci] for ci in output_stages]
    model = tf.keras.models.Model(img_input, outputs, name='xception')

    return model
