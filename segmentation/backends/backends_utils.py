import tensorflow as tf
import tensorflow.contrib.slim as slim

__all__ = ['build_assign_fn', 'build_warm_start_settings']


def build_assign_fn(backend_type, ckpt_path):
    if backend_type.startswith('vgg'):
        return _get_vgg_assign_fn(ckpt_path)
    elif backend_type == 'xception':
        return _get_xception_assign_fn(ckpt_path)

    return None


def _get_vgg_assign_fn(ckpt_path):
    block_var_list = slim.get_variables_to_restore(include=['block'])
    for var in slim.get_variables_to_restore():
        tf.logging.debug(var.op.name)

    # ckpt name to tf.Variable
    var_dict = {var.op.name: var for var in block_var_list}

    tf.logging.info(
        'creating vgg assign fn with {} vars'.format(len(var_dict)))
    return slim.assign_from_checkpoint_fn(ckpt_path, var_dict, True, True)


def _get_xception_assign_fn(ckpt_path):
    block_var_list = slim.get_variables_to_restore(include=['block'])
    for var in slim.get_variables_to_restore():
        tf.logging.debug(var.op.name)

    # ckpt name to tf.Variable
    var_dict = {var.op.name: var for var in block_var_list}

    tf.logging.debug('vgg assign fn vars list')
    for var in var_dict:
        tf.logging.debug(str(var))

    tf.logging.info(
        'creating vgg16 assign fn with {} vars'.format(len(var_dict)))
    return slim.assign_from_checkpoint_fn(ckpt_path, var_dict, True, True)


def build_warm_start_settings(backend_type, ckpt_path):
    if backend_type.startswith('vgg'):
        return _get_vgg_ws_settings(ckpt_path)
    elif backend_type == 'resnet50':
        return _get_resnet50_ws_settings(ckpt_path)
    elif backend_type == 'xception':
        return _get_xception_ws_settings(ckpt_path)
    elif backend_type == 'inception_v3':
        return _get_inception_v3_ws_settings(ckpt_path)
    elif backend_type == 'inception_resnet_v2':
        return _get_inception_resnet_v2_ws_settings(ckpt_path)
    return None


def _get_vgg_ws_settings(ckpt_path):
    return tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=ckpt_path,
        vars_to_warm_start=['block.+kernel[^/]', 'block.+bias[^/]'],
    )


def _get_xception_ws_settings(ckpt_path):
    return tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=ckpt_path,
        vars_to_warm_start=['block.+kernel[^/]',
                            'conv.+kernel[^/]',

                            'block.+gamma[^/]',
                            'block.+beta[^/]',
                            'block.+moving_mean[^/]',
                            'block.+moving_variance[^/]',
                            'batch.+gamma[^/]',
                            'batch.+beta[^/]',
                            'batch.+moving_mean[^/]',
                            'batch.+moving_variance[^/]', ],
    )


def _get_resnet50_ws_settings(ckpt_path):
    return tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=ckpt_path,
        vars_to_warm_start=['res.+kernel[^/]',
                            'res.+bias[^/]',
                            'conv.+kernel[^/]',
                            'conv.+bias[^/]',

                            'bn.+gamma[^/]',
                            'bn.+beta[^/]',
                            'bn.+moving_mean[^/]',
                            'bn.+moving_variance[^/]', ],
    )


def _get_inception_v3_ws_settings(ckpt_path):
    return tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=ckpt_path,
        vars_to_warm_start=['conv2d.+kernel[^/]',

                            'batch_normalization.+gamma[^/]',
                            'batch_normalization.+beta[^/]',
                            'batch_normalization.+moving_mean[^/]',
                            'batch_normalization.+moving_variance[^/]', ],
    )


def _get_inception_resnet_v2_ws_settings(ckpt_path):
    return tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=ckpt_path,
        vars_to_warm_start=['conv2d.+kernel[^/]',

                            'block.+kernel[^/]',
                            'block.+bias[^/]',

                            'batch.+gamma[^/]',
                            'batch.+beta[^/]',
                            'batch.+moving_mean[^/]',
                            'batch.+moving_variance[^/]',
                            'conv_.+gamma[^/]',
                            'conv_.+beta[^/]',
                            'conv_.+moving_mean[^/]',
                            'conv_.+moving_variance[^/]', ],
    )
