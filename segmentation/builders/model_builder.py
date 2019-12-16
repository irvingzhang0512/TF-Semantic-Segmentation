from ..keras_models import deeplab, deeplabv3_plus


def build_model(model_type='deeplab_v3_plus',
                backend_type='xception',
                weights=None,
                num_classes=21,
                input_shape=(None, None, 3),
                OS=16,
                fine_tune_batch_norm=False,
                ):
    if model_type == 'deeplab_v3_plus':
        return deeplab.DeepLabV3Plus(
            backend_type=backend_type,
            weights=weights,
            num_classes=num_classes,
            input_shape=input_shape,
            OS=OS,
            fine_tune_batch_norm=fine_tune_batch_norm,
        )
    if model_type == 'deeplab_v3_plus_v2':
        return deeplabv3_plus.DeepLabV3Plus(
            backend_type=backend_type,
            weights=weights,
            num_classes=num_classes,
            input_shape=input_shape,
        )
    raise ValueError('unknown model type {}'.format(model_type))
