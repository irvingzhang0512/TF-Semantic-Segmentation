import tensorflow as tf
from ..builders import backend_builder
from ..utils.pretrained_utils import load_pretrained_weights

layers = tf.keras.layers
models = tf.keras.models
backend = tf.keras.backend


class Concatenate(layers.Concatenate):
    def __init__(self, out_size=None, axis=-1, name=None):
        super(Concatenate, self).__init__(axis=axis, name=name)
        self.out_size = out_size

    def call(self, inputs):
        return backend.concatenate(inputs, self.axis)

    def build(self, input_shape):
        pass

    def compute_output_shape(self, input_shape):
        if self.out_size is None:
            return super(Concatenate, self).compute_output_shape(input_shape)
        else:
            if not isinstance(input_shape, list):
                raise ValueError('A `Concatenate` layer should be called '
                                 'on a list of inputs.')
            input_shapes = input_shape
            output_shape = list(input_shapes[0])
            for shape in input_shapes[1:]:
                if output_shape[self.axis] is None or shape[self.axis] is None:
                    output_shape[self.axis] = None
                    break
                output_shape[self.axis] += shape[self.axis]
            return tuple([output_shape[0]] + list(self.out_size) + [output_shape[-1]])

    def get_config(self):
        config = super(Concatenate, self).get_config()
        config['out_size'] = self.out_size
        return config


class GlobalAveragePooling2D(layers.GlobalAveragePooling2D):
    def __init__(self, keep_dims=False, **kwargs):
        super(GlobalAveragePooling2D, self).__init__(**kwargs)
        self.keep_dims = keep_dims

    def call(self, inputs):
        if not self.keep_dims:
            return super(GlobalAveragePooling2D, self).call(inputs)
        else:
            return backend.mean(inputs, axis=[1, 2], keepdims=True)

    def compute_output_shape(self, input_shape):
        if self.keep_dims is False:
            return super(GlobalAveragePooling2D, self).compute_output_shape(input_shape)
        else:
            input_shape = tf.TensorShape(input_shape).as_list()
            return tf.TensorShape([input_shape[0], 1, 1, input_shape[3]])

    def get_config(self):
        config = super(GlobalAveragePooling2D, self).get_config()
        config['keep_dim'] = self.keep_dims
        return config


def _aspp(x, out_filters, aspp_size):
    xs = list()
    x1 = layers.Conv2D(out_filters, 1, strides=1)(x)
    xs.append(x1)

    for i in range(3):
        xi = layers.Conv2D(out_filters, 3, strides=1,
                           padding='same', dilation_rate=6 * (i + 1))(x)
        xs.append(xi)
    img_pool = GlobalAveragePooling2D(keep_dims=True)(x)
    img_pool = layers.Conv2D(
        out_filters, 1, 1, kernel_initializer='he_normal')(img_pool)
    img_pool = layers.UpSampling2D(
        size=aspp_size, interpolation='bilinear')(img_pool)
    xs.append(img_pool)

    x = Concatenate(out_size=aspp_size)(xs)
    x = layers.Conv2D(out_filters, 1, strides=1,
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    return x


def _conv_bn_relu(x, filters, kernel_size, strides=1):
    x = layers.Conv2D(filters, kernel_size,
                      strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def DeepLabV3Plus(input_shape=(512, 512, 3),
                  num_classes=21,
                  backend_type='xception-deeplab',
                  weights=None,
                  ):
    aspp_size = (input_shape[0] // 16, input_shape[1] // 16)
    if backend_type in ['vgg16',
                        'vgg19',
                        'resnet50',
                        'resnet101',
                        'resnet152',
                        'mobilenetv1',
                        'mobilenetv2']:
        output_stages = ['c2', 'c5']
    else:
        output_stages = ['c1', 'c5']

    preprocess_fn = backend_builder.build_preprocess_fn(backend_type)
    backend_fn = backend_builder.build_backend(
        backend_type,
        input_shape=input_shape,
        output_stages=output_stages,
    )

    img_input = layers.Input(shape=input_shape)
    preprocessed_img = preprocess_fn(img_input)
    c2, c5 = backend_fn(preprocessed_img)
    x = _aspp(c5, 256, aspp_size)
    x = layers.Dropout(rate=0.5)(x)

    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    x = _conv_bn_relu(x, 48, 1, strides=1)

    x = Concatenate(out_size=aspp_size)([x, c2])
    x = _conv_bn_relu(x, 256, 3, 1)
    x = layers.Dropout(rate=0.5)(x)

    x = _conv_bn_relu(x, 256, 3, 1)
    x = layers.Dropout(rate=0.1)(x)

    x = layers.Conv2D(num_classes, 1, strides=1)(x)
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    outputs = x
    model = models.Model(img_input, outputs, name='deeplabv3_plus')
    if weights is not None:
        load_pretrained_weights(model, weights, backend_type)

    return model
