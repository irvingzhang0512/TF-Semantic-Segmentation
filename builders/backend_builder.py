import collections
import tensorflow as tf
from backends.backends_utils import build_warm_start_settings \
    as build_warm_start_settings_lib
from backends import vgg16, vgg19, resnet, xception, densenet, inception_v3, \
    inception_resnet_v2, nasnet

BackendDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
        'input_size',
        'downsample_stride',
        'preprocess_fn'
    ])

VGG16 = 'vgg16'
VGG19 = 'vgg19'
RESNET50 = 'resnet50'
RESNET101 = 'resnet101'
RESNET152 = 'resnet152'
RESNET50V2 = 'resnet50v2'
RESNET101V2 = 'resnet101v2'
RESNET152V2 = 'resnet152v2'
RESNEXT50 = 'resnext50'
RESNEXT101 = 'resnext101'
XCEPTION = 'xception'
DENSENET121 = 'densenet121'
DENSENET169 = 'densenet169'
DENSENET201 = 'densenet201'
INCEPTIONV3 = 'inception_v3'
INCEPTIONRESNETV2 = 'inception_resnet_v2'
NASNETMOBILE = 'nasnetmobile'
NASNETLARGE = 'nasnetlarge'

_MEAN_BGR = [103.939, 116.779, 123.68]


def _preprocess_caffe(inputs, dtype=tf.float32):
    """convert RGB to BGR and subject means"""
    mean_bgr = tf.reshape(_MEAN_BGR, [1, 1, 1, 3])
    inputs = tf.reverse(inputs, axis=[-1])
    return tf.cast(tf.cast(inputs, tf.float32) - mean_bgr,
                   dtype=dtype)


def _preprocess_tf(inputs, dtype=tf.float32):
    """Map image values from [0, 255] to [-1, 1]."""
    preprocessed_inputs = (1.0 / 127.5) * tf.cast(inputs, tf.float32) - 1.0
    return tf.cast(preprocessed_inputs, dtype=dtype)


def _preprocess_torch(inputs, dtype=tf.float32):
    inputs = tf.cast(inputs, tf.float32) / 255.
    mean_rgb = tf.reshape([0.485, 0.456, 0.406], [1, 1, 1, 3])
    std_rgb = tf.reshape([0.229, 0.224, 0.225], [1, 1, 1, 3])
    inputs = (inputs - mean_rgb) / std_rgb
    return inputs


_VGG19_INFORMATION = BackendDescriptor(
    input_size=(224, 224, 3),
    downsample_stride=32,
    preprocess_fn=_preprocess_caffe,
)

_VGG16_INFORMATION = BackendDescriptor(
    input_size=(224, 224, 3),
    downsample_stride=32,
    preprocess_fn=_preprocess_caffe,
)

_RESNET50_INFORMATION = BackendDescriptor(
    input_size=(224, 224, 3),
    downsample_stride=32,
    preprocess_fn=_preprocess_caffe,
)

_XCEPTION_INFORMATION = BackendDescriptor(
    input_size=(224, 224, 3),
    downsample_stride=32,
    preprocess_fn=_preprocess_tf,
)

_DENSENET121_INFORMATION = BackendDescriptor(
    input_size=(224, 224, 3),
    downsample_stride=32,
    preprocess_fn=_preprocess_torch,
)

_INCEPTION_RESNET_V2_INFORMATION = BackendDescriptor(
    input_size=(224, 224, 3),
    downsample_stride=32,
    preprocess_fn=_preprocess_tf,
)

_INCEPTION_V3_INFORMATION = BackendDescriptor(
    input_size=(224, 224, 3),
    downsample_stride=32,
    preprocess_fn=_preprocess_tf,
)


BACKEND_INFORMATION = {
    VGG16: _VGG16_INFORMATION,
    VGG19: _VGG19_INFORMATION,
    RESNET50: _RESNET50_INFORMATION,
    XCEPTION: _XCEPTION_INFORMATION,
    DENSENET121: _DENSENET121_INFORMATION,
    INCEPTIONV3: _INCEPTION_V3_INFORMATION,
    INCEPTIONRESNETV2: _INCEPTION_RESNET_V2_INFORMATION,
}

MODEL_DIR = 'models'


def build_backend(backend_type, input_shape=(None, None, 3)):
    if backend_type == 'vgg16':
        return vgg16.VGG16(include_top=False,
                           weights=None,
                           input_shape=input_shape)
    elif backend_type == 'vgg19':
        return vgg19.VGG19(include_top=False,
                           weights=None,
                           input_shape=input_shape)
    elif backend_type == 'resnet50':
        return resnet.ResNet50(include_top=False,
                               weights=None,
                               input_shape=input_shape,)
    elif backend_type == 'xception':
        return xception.Xception(include_top=False,
                                 weights=None,
                                 input_shape=input_shape)
    elif backend_type == 'densenet121':
        return densenet.DenseNet121(include_top=False,
                                    weights=None,
                                    input_shape=input_shape)
    elif backend_type == 'inception_v3':
        return inception_v3.InceptionV3(include_top=False,
                                        weights=None,
                                        input_shape=input_shape)
    elif backend_type == 'inception_resnet_v2':
        return inception_resnet_v2.InceptionResNetV2(include_top=False,
                                                     weights=None,
                                                     input_shape=input_shape)
    elif backend_type == 'NasnetLarge':
        return nasnet.NASNetLarge(include_top=False,
                                  weights=None,
                                  input_shape=input_shape)
    raise ValueError('unknown backend type {}'.format(backend_type))


def build_warm_start_settings(backend_type, ckpt_path):
    return build_warm_start_settings_lib(backend_type, ckpt_path)
