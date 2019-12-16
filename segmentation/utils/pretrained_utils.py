import logging
import tensorflow as tf


logger = logging.getLogger('tensorflow')

MODELS = "models"

WEIGHTS_PATH_X = (
    "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download"
    "/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
)
WEIGHTS_PATH_MOBILE = (
    "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download"
    "/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"
)
WEIGHTS_PATH_X_CS = (
    "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download"
    "/1.2/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5"
)
WEIGHTS_PATH_MOBILE_CS = (
    "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download"
    "/1.2/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5"
)
RESNET50_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                   'releases/download/v0.2/'
                   'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

VGG16_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

XCEPTION_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.4/'
    'xception_weights_tf_dim_ordering_tf_kernels_notop.h5')


def load_pretrained_weights(model, weights, backend_type):
    """load h5 pre-trained model for models."""
    weights_path = None
    if weights == 'pascal_voc':
        if backend_type == 'xception':
            weights_path = tf.keras.utils.get_file(
                'deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH_X,
                cache_subdir=MODELS
            )
            model.get_layer('xception').load_weights(
                weights_path, by_name=True)
    elif weights == 'cityscapes':
        if backend_type == 'xception':
            weights_path = tf.keras.utils.get_file(
                'deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
                WEIGHTS_PATH_X_CS,
                cache_subdir=MODELS
            )
            model.get_layer('xception').load_weights(
                weights_path, by_name=True)
    elif weights == 'imagenet':
        if backend_type == 'resnet50':
            weights_path = tf.keras.utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                RESNET50_NO_TOP,
                cache_subdir=MODELS,
                md5_hash='a268eb855778b3df3c7506639542a6af')
            model.get_layer('resnet50').load_weights(
                weights_path, by_name=True
            )
        if backend_type == 'vgg16':
            weights_path = tf.keras.utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                VGG16_NO_TOP,
                cache_subdir=MODELS,
                md5_hash='6d6bbae143d832006294945121d1f1fc')
            model.get_layer('vgg16').load_weights(
                weights_path, by_name=True
            )
        if backend_type.startswith('xception'):
            weights_path = tf.keras.utils.get_file(
                'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                XCEPTION_NO_TOP,
                cache_subdir=MODELS,
                md5_hash='b0042744bf5b25fce3cb969f33bebb97')
            model.get_layer('xception').load_weights(
                weights_path, by_name=True
            )
    elif weights is not None:
        weights_path = weights

    if weights_path is None:
        raise ValueError('unknown weights {} for backend {}'.format(
            weights, backend_type
        ))
    model.load_weights(weights_path, by_name=True)
    logger.info('successfully load %s weights in %s for %s' % (
        weights, weights_path, backend_type
    ))
