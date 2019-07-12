import tensorflow as tf
from estimator_models.estimator_spec_utils import get_predict_spec, get_evaluation_spec, get_train_spec, \
    preprocess_zero_mean_unit_range
from models.Encoder_Decoder import build_encoder_decoder


def get_encoder_decoder_model_fn(num_classes, optimizer, preset_model="Encoder-Decoder",):
    def _encoder_decoder_model_fn(features, labels, mode, params, config):
        features['image'] = preprocess_zero_mean_unit_range(features['image'])

        if mode == tf.estimator.ModeKeys.PREDICT:
            logits = build_encoder_decoder(features['images'], num_classes, preset_model=preset_model)
            return get_predict_spec(logits, features['image_name'])

        logits = build_encoder_decoder(features['image'], num_classes, preset_model=preset_model)
        if mode == tf.estimator.ModeKeys.TRAIN:
            return get_train_spec(logits, labels, optimizer)
        if mode == tf.estimator.ModeKeys.EVAL:
            return get_evaluation_spec(logits, labels, num_classes)

    return _encoder_decoder_model_fn
