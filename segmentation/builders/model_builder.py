from ..keras_models import deeplab


def build_model(model_type='deeplab_v3_plus',
                backend_type='xception',
                weights=None,
                num_classes=21,
                activation=None,
                input_shape=(None, None, 3),
                OS=16,
                ):
    if model_type == 'deeplab_v3_plus':
        return deeplab.DeepLabV3Plus(
            backend_type=backend_type,
            weights=weights,
            num_classes=num_classes,
            activation=activation,
            input_shape=input_shape,
            OS=OS,
        )
    raise ValueError('unknown model type {}'.format(model_type))
