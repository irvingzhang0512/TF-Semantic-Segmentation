import tensorflow as tf
from estimator_models.estimator_model_utils import preprocess_zero_to_one, get_estimator_spec
from models.FC_DenseNet_Tiramisu import build_fc_densenet

__all__ = ['get_fc_densenet_model_fn']


def get_fc_densenet_model_fn(num_classes, preset_model="FC-DenseNet56", ):
    def _fc_densenet_model_fn(features, labels, mode, params, config):
        features['image'] = preprocess_zero_to_one(features['image'])
        logits = build_fc_densenet(features['image'], num_classes, 
                                    preset_model=preset_model,
                                    weight_decay=params['weight_decay'])
        return get_estimator_spec(mode, logits, None, labels=labels, num_classes=num_classes, params=params)

    return _fc_densenet_model_fn
