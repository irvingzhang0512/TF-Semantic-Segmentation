import tensorflow as tf
from estimator_models.estimator_model_utils import get_preprocess_by_frontend, get_estimator_spec
from models.DeepLabV3 import build_deeplabv3


__all__ = ['get_deeplabv3_model_fn']


def get_deeplabv3_model_fn(num_classes, 
                           frontend="ResNet101", 
                           pretrained_dir="/ssd/zhangyiyang/data/slim"):
    def _deeplabv3_model_fn(features, labels, mode, params, config):
        _preprocess = get_preprocess_by_frontend(frontend=frontend)
        features['image'] = _preprocess(features['image'])
        is_training = True if mode == tf.estimator.ModeKeys.TRAIN else False
        logits, init_fn = build_deeplabv3(features['image'], num_classes,
                                          frontend=frontend, weight_decay=params['weight_decay'],
                                          pretrained_dir=pretrained_dir, is_training=is_training)
        
        
        return get_estimator_spec(mode, logits, init_fn, labels=labels, num_classes=num_classes, params=params)

    return _deeplabv3_model_fn
