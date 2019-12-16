import collections
import tensorflow as tf
from ..backends import xception_deeplab, xception

BackendDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
        'input_size',
        'downsample_stride',
        'preprocess_fn'
    ])


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
    return tf.cast(inputs, dtype)


def _preporcess_zero_to_one(inputs, dtype=tf.float32):
    return tf.cast(tf.cast(inputs, tf.float32)/255., dtype)


XCEPTION = 'xception'
_XCEPTION_INFORMATION = BackendDescriptor(
    input_size=(299, 299, 3),
    downsample_stride=32,
    preprocess_fn=_preporcess_zero_to_one,
)

XCEPTION_DEEPLAB = 'xception-deeplab'
_XCEPTION_DEEPLAB_INFORMATION = BackendDescriptor(
    input_size=(299, 299, 3),
    downsample_stride=32,
    preprocess_fn=_preprocess_tf,
)


BACKEND_INFORMATION = {
    XCEPTION: _XCEPTION_INFORMATION,
    XCEPTION_DEEPLAB: _XCEPTION_DEEPLAB_INFORMATION,
}


def build_preprocess_fn(backend_type):
    if backend_type in BACKEND_INFORMATION:
        return BACKEND_INFORMATION[backend_type].preprocess_fn
    raise ValueError('unknown backend type {}'.format(backend_type))


def build_backend(backend_type,
                  input_shape=(513, 513, 3),
                  **kwargs):
    if backend_type == XCEPTION:
        return xception_deeplab.Xception(
            input_shape=input_shape,
            OS=kwargs.get('OS'),
            fine_tune_batch_norm=kwargs.get('fine_tune_batch_norm'),
        )
    if backend_type == XCEPTION_DEEPLAB:
        return xception.Xception(
            input_shape=input_shape,
            output_stages=kwargs.get('output_stages'),
            xception_type=backend_type,
            dilation=[1, 2],
        )
    elif backend_type == RESNET50:
        return resnet50.ResNet50(input_shape=input_shape)
    elif backend_type == VGG16:
        return vgg16.VGG16(input_shape=input_shape)

    raise ValueError('unknown backend type {}'.format(backend_type))
