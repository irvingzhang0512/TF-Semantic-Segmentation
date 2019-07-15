import tensorflow as tf
from estimator_models.estimator_spec_utils import get_predict_spec, get_evaluation_spec, get_train_spec, \
    preprocess_zero_to_one, get_optimizer
from models.DeepLabV3 import build_deeplabv3


__all__ = ['get_deeplabv3_model_fn']


def get_deeplabv3_model_fn(num_classes, 
                           frontend="Res101", 
                           weight_decay=1e-5,
                           pretrained_dir="/ssd/zhangyiyang/data/slim"):
    def _deeplabv3_model_fn(features, labels, mode, params, config):
        optimizer = get_optimizer(params)
        features['image'] = preprocess_zero_to_one(features['image'])
        if mode == tf.estimator.ModeKeys.TRAIN:
            # def init_fn(session):
            logits, init_fn = build_deeplabv3(features['image'], num_classes,
                                              frontend=frontend, weight_decay=weight_decay,
                                              pretrained_dir=pretrained_dir, is_training=True)
            def _init_fn(scaffold, session):
                init_fn(session)
            scaffold = tf.train.Scaffold(init_fn=_init_fn)
            return get_train_spec(logits, labels, optimizer, 
                                  num_classes=num_classes, scaffold=scaffold)

        logits, init_fn = build_deeplabv3(features['image'], num_classes,
                                          frontend=frontend, weight_decay=weight_decay,
                                          pretrained_dir=pretrained_dir, is_training=False)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return get_predict_spec(logits, features['image_name'])
        if mode == tf.estimator.ModeKeys.EVAL:
            return get_evaluation_spec(logits, labels, num_classes)

    return _deeplabv3_model_fn
