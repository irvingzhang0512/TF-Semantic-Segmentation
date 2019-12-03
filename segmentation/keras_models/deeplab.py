import logging
import tensorflow as tf
from ..backends.xception_deeplab import SepConv_BN
from ..builders import backend_builder
from ..utils.pretrained_utils import load_pretrained_weights

logger = logging.getLogger('tensorflow')

layers = tf.keras.layers


class BilinearUpsampling(layers.Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        if output_size:
            self.upsample_size = output_size
            self.upsampling = None
        else:
            self.upsampling = upsampling

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.upsample_size[0]
            width = self.upsample_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return tf.compat.v1.image.resize_bilinear(
                inputs,
                (inputs.shape[1] * self.upsampling[0],
                 inputs.shape[2] * self.upsampling[1]),
                align_corners=True,)
        else:
            return tf.compat.v1.image.resize_bilinear(
                inputs,
                (self.upsample_size[0], self.upsample_size[1]),
                align_corners=True,)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'upsample_size': self.upsample_size}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _aspp(x, OS, backend_type,
          fine_tune_batch_norm=False,):
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
    b4 = layers.BatchNormalization(
        name='image_pooling_BN',
        epsilon=1e-5,
        trainable=fine_tune_batch_norm,)(b4)
    b4 = layers.Activation('relu')(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = BilinearUpsampling(output_size=size_before[1:3])(b4)
    # simple 1x1
    b0 = layers.Conv2D(256, (1, 1), padding='same',
                       use_bias=False, name='aspp0')(x)
    b0 = layers.BatchNormalization(
        name='aspp0_BN',
        epsilon=1e-5,
        trainable=fine_tune_batch_norm,)(b0)
    b0 = layers.Activation('relu', name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    if backend_type == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0],
                        depth_activation=True, epsilon=1e-5,
                        fine_tune_batch_norm=fine_tune_batch_norm,)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1],
                        depth_activation=True, epsilon=1e-5,
                        fine_tune_batch_norm=fine_tune_batch_norm,)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2],
                        depth_activation=True, epsilon=1e-5,
                        fine_tune_batch_norm=fine_tune_batch_norm,)

        # concatenate ASPP branches & project
        x = layers.Concatenate()([b4, b0, b1, b2, b3])
    else:
        x = layers.Concatenate()([b4, b0])

    x = layers.Conv2D(256, (1, 1), padding='same',
                      use_bias=False, name='concat_projection')(x)
    x = layers.BatchNormalization(
        name='concat_projection_BN',
        epsilon=1e-5,
        trainable=fine_tune_batch_norm,)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)

    return x


def _deeplab_v3_plus_decoder(x, skip, img_shape, backend_type,
                             weights, num_classes, activation,
                             fine_tune_batch_norm=False,):
    if backend_type == 'xception':
        # Feature projection
        # x4 (x2) block
        x = BilinearUpsampling(output_size=skip.shape[1:3])(x)

        dec_skip1 = layers.Conv2D(48, (1, 1), padding='same',
                                  use_bias=False,
                                  name='feature_projection0')(skip)
        dec_skip1 = layers.BatchNormalization(
            name='feature_projection0_BN',
            epsilon=1e-5,
            trainable=fine_tune_batch_norm,)(dec_skip1)
        dec_skip1 = layers.Activation('relu')(dec_skip1)
        x = layers.Concatenate()([x, dec_skip1])

        x = SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5,
                       fine_tune_batch_norm=fine_tune_batch_norm,)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5,
                       fine_tune_batch_norm=fine_tune_batch_norm,)

    # you can use it with arbitary number of classes
    if (weights == 'pascal_voc' and num_classes == 21) or \
            (weights == 'cityscapes' and num_classes == 19):
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = layers.Conv2D(num_classes, (1, 1),
                      padding='same', name=last_layer_name)(x)
    x = BilinearUpsampling(output_size=img_shape[0:2])(x)

    if activation in {'softmax', 'sigmoid'}:
        x = tf.keras.layers.Activation(activation)(x)

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

    # (33, 33, 1024), (129, 129, 256)
    extractor_output, skip = backend(preprocessed_tensor)

    aspp_output = _aspp(
        x=extractor_output,
        OS=OS,
        backend_type=backend_type,
        fine_tune_batch_norm=fine_tune_batch_norm,
    )

    final_output = _deeplab_v3_plus_decoder(
        x=aspp_output,
        skip=skip,
        img_shape=input_shape,
        backend_type=backend_type,
        weights=weights,
        num_classes=num_classes,
        activation=activation,
        fine_tune_batch_norm=fine_tune_batch_norm,
    )

    model = tf.keras.Model(
        input_tensor,
        final_output,
        name='deeplab_v3_plus',
    )

    if weights is not None:
        load_pretrained_weights(model, weights, backend_type)

    return model
