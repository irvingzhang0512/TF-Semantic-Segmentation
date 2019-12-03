import logging
import tensorflow as tf
from ..builders import backend_builder
from ..utils.pretrained_utils import load_pretrained_weights

logger = logging.getLogger('tensorflow')

layers = tf.keras.layers


def Upsample(tensor, size):
    '''bilinear upsampling'''
    name = tensor.name.split('/')[0] + '_upsample'

    def bilinear_upsample(x, size):
        resized = tf.image.resize(
            images=x, size=size)
        return resized
    y = layers.Lambda(lambda x: bilinear_upsample(x, size),
                      output_shape=size, name=name)(tensor)
    return y


def _aspp_v2(tensor):
    '''atrous spatial pyramid pooling'''
    dims = tensor.get_shape().as_list()

    y_pool = layers.AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = layers.Conv2D(filters=256, kernel_size=1,
                           padding='same', kernel_initializer='he_normal',
                           name='pool_1x1conv2d', use_bias=False)(y_pool)
    y_pool = layers.BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = layers.Activation('relu', name=f'relu_1')(y_pool)

    y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])

    y_1 = layers.Conv2D(filters=256, kernel_size=1,
                        dilation_rate=1, padding='same',
                        kernel_initializer='he_normal',
                        name='ASPP_conv2d_d1', use_bias=False)(tensor)
    y_1 = layers.BatchNormalization(name=f'bn_2')(y_1)
    y_1 = layers.Activation('relu', name=f'relu_2')(y_1)

    y_6 = layers.Conv2D(filters=256, kernel_size=3,
                        dilation_rate=6, padding='same',
                        kernel_initializer='he_normal',
                        name='ASPP_conv2d_d6', use_bias=False)(tensor)
    y_6 = layers.BatchNormalization(name=f'bn_3')(y_6)
    y_6 = layers.Activation('relu', name=f'relu_3')(y_6)

    y_12 = layers.Conv2D(filters=256, kernel_size=3,
                         dilation_rate=12, padding='same',
                         kernel_initializer='he_normal',
                         name='ASPP_conv2d_d12', use_bias=False)(tensor)
    y_12 = layers.BatchNormalization(name=f'bn_4')(y_12)
    y_12 = layers.Activation('relu', name=f'relu_4')(y_12)

    y_18 = layers.Conv2D(filters=256, kernel_size=3,
                         dilation_rate=18, padding='same',
                         kernel_initializer='he_normal',
                         name='ASPP_conv2d_d18', use_bias=False)(tensor)
    y_18 = layers.BatchNormalization(name=f'bn_5')(y_18)
    y_18 = layers.Activation('relu', name=f'relu_5')(y_18)

    y = layers.concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')

    y = layers.Conv2D(filters=256, kernel_size=1,
                      dilation_rate=1, padding='same',
                      kernel_initializer='he_normal',
                      name='ASPP_conv2d_final', use_bias=False)(y)
    y = layers.BatchNormalization(name=f'bn_final')(y)
    y = layers.Activation('relu', name=f'relu_final')(y)
    return y


def _decoder(x_a, x_b, img_shape, num_classes):
    x_a = Upsample(tensor=x_a, size=[x_b.shape[1], x_b.shape[2]])

    x_b = layers.Conv2D(filters=48, kernel_size=1, padding='same',
                        kernel_initializer='he_normal',
                        name='low_level_projection', use_bias=False)(x_b)
    x_b = layers.BatchNormalization(name=f'bn_low_level_projection')(x_b)
    x_b = layers.Activation('relu', name='low_level_activation')(x_b)

    x = layers.concatenate([x_a, x_b], name='decoder_concat')

    x = layers.Conv2D(filters=256, kernel_size=3,
                      padding='same', activation='relu',
                      kernel_initializer='he_normal',
                      name='decoder_conv2d_1', use_bias=False)(x)
    x = layers.BatchNormalization(name=f'bn_decoder_1')(x)
    x = layers.Activation('relu', name='activation_decoder_1')(x)

    x = layers.Conv2D(filters=256, kernel_size=3,
                      padding='same', activation='relu',
                      kernel_initializer='he_normal',
                      name='decoder_conv2d_2', use_bias=False)(x)
    x = layers.BatchNormalization(name=f'bn_decoder_2')(x)
    x = layers.Activation('relu', name='activation_decoder_2')(x)
    x = Upsample(x, [img_shape[0], img_shape[1]])

    x = layers.Conv2D(num_classes, (1, 1), name='output_layer')(x)

    return x


def DeepLabV3Plus(backend_type='xception',
                  weights=None,
                  num_classes=21,
                  activation=None,
                  input_shape=(513, 513, 3),
                  OS=16,
                  fine_tune_batch_norm=False,
                  ):
    preprocess_fn = backend_builder.build_preprocess_fn(backend_type)
    backend = backend_builder.build_backend(
        backend_type=backend_type,
        input_shape=input_shape,
        OS=OS,
        fine_tune_batch_norm=fine_tune_batch_norm,
    )

    input_tensor = layers.Input(input_shape)

    preprocessed_tensor = layers.Lambda(
        lambda xx: preprocess_fn(xx),
        output_shape=input_shape,
        trainable=False,
        name='preprocess_fn',
    )(input_tensor)

    extractor_output, skip = backend(preprocessed_tensor)

    aspp_output = _aspp_v2(
        extractor_output
    )

    final_output = _decoder(
        aspp_output,
        skip,
        img_shape=input_shape,
        num_classes=num_classes,
    )

    model = tf.keras.Model(
        input_tensor,
        final_output,
        name='deeplab_v3_plus',
    )

    if weights is not None:
        load_pretrained_weights(model, weights, backend_type)

    return model
