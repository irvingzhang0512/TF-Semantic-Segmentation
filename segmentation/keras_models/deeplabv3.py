import tensorflow as tf
from backends.xception_deeplab import SepConv_BN
from tensorflow.python.keras.utils.data_utils import get_file

WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_X_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5"
WEIGHTS_PATH_MOBILE_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5"


layers = tf.keras.layers


def _aspp(x, OS, backend_type):
    # branching for Atrous Spatial Pyramid Pooling
    if OS == 8:
        atrous_rates = (12, 24, 36)
    else:
        atrous_rates = (6, 12, 18)

    # Image Feature branch
    b4 = layers.GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = layers.Lambda(lambda x: tf.expand_dims(x, 1))(b4)
    b4 = layers.Lambda(lambda x: tf.expand_dims(x, 1))(b4)
    b4 = layers.Conv2D(256, (1, 1), padding='same',
                       use_bias=False, name='image_pooling')(b4)
    b4 = layers.BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = layers.Activation('relu')(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = layers.Lambda(lambda x: tf.image.resize(x, size_before[1:3],
                                                 align_corners=True))(b4)
    # simple 1x1
    b0 = layers.Conv2D(256, (1, 1), padding='same',
                       use_bias=False, name='aspp0')(x)
    b0 = layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = layers.Activation('relu', name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    if backend_type == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0],
                        depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1],
                        depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2],
                        depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        x = layers.Concatenate()([b4, b0, b1, b2, b3])
    else:
        x = layers.Concatenate()([b4, b0])

    x = layers.Conv2D(256, (1, 1), padding='same',
                      use_bias=False, name='concat_projection')(x)
    x = layers.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)


def _deeplab_v3_plus_decoder(x, skip, img_shape, backend_type,
                             weights, num_classes, activation):
    if backend_type == 'xception':
        # Feature projection
        # x4 (x2) block
        x = layers.Lambda(lambda xx: tf.image.resize(x,
                                                     skip.shape[1:3],
                                                     align_corners=True))(x)

        dec_skip1 = layers.Conv2D(48, (1, 1), padding='same',
                                  use_bias=False,
                                  name='feature_projection0')(skip)
        dec_skip1 = layers.BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = layers.Activation('relu')(dec_skip1)
        x = layers.Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)

    # you can use it with arbitary number of classes
    if (weights == 'pascal_voc' and num_classes == 21) or \
            (weights == 'cityscapes' and num_classes == 19):
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = layers.Conv2D(num_classes, (1, 1),
                      padding='same', name=last_layer_name)(x)
    x = layers.Lambda(lambda xx: tf.image.resize(xx,
                                                 img_shape[0:2],
                                                 align_corners=True))(x)

    if activation in {'softmax', 'sigmoid'}:
        x = tf.keras.layers.Activation(activation)(x)


def DeepLabV3(backend,
              backend_type='xception',
              weights=None,
              num_classes=21,
              activation=None,
              img_shape=(512, 512, 3), OS=16):
    input_tensor = layers.Input(img_shape)
    extractor_output, skip = backend(input_tensor)
    aspp_output = _aspp(
        x=extractor_output,
        OS=OS,
        backend_type=backend_type,
    )
    final_output = _deeplab_v3_plus_decoder(
        x=aspp_output,
        skip=skip,
        img_shape=img_shape,
        backend_type=backend_type,
        weights=weights,
        num_classes=num_classes,
        activation=activation,
    )
    model = tf.keras.Model(
        input_tensor,
        final_output,
        name='deeplab_v3_plus',
    )

    # load weights
    if weights == 'pascal_voc':
        if backend_type == 'xception':
            weights_path = get_file(
                'deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH_X,
                cache_subdir='models'
            )
        else:
            weights_path = get_file(
                'deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH_MOBILE,
                cache_subdir='models'
            )
        model.load_weights(weights_path, by_name=True)
    elif weights == 'cityscapes':
        if backend_type == 'xception':
            weights_path = get_file(
                'deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
                WEIGHTS_PATH_X_CS,
                cache_subdir='models'
            )
        else:
            weights_path = get_file(
                'deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5',
                WEIGHTS_PATH_MOBILE_CS,
                cache_subdir='models'
            )
        model.load_weights(weights_path, by_name=True)
    return model
