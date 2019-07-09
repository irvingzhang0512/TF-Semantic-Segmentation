import tensorflow as tf
from estimator_models.estimator_spec_utils import get_predict_spec, get_evaluation_spec, get_train_spec
from models.Encoder_Decoder import build_encoder_decoder


def get_encoder_decoder_model_fn(num_classes, optimizer, preset_model="Encoder-Decoder", ):
    def _encoder_decoder_model_fn(features, labels, mode, params, config):
        if mode == tf.estimator.ModeKeys.PREDICT:
            logits = build_encoder_decoder(features['images'], num_classes, preset_model=preset_model)
            return get_predict_spec(logits, features['target_file_names'])

        logits = build_encoder_decoder(features, num_classes, preset_model=preset_model)
        if mode == tf.estimator.ModeKeys.TRAIN:
            return get_train_spec(logits, labels, optimizer)
        if mode == tf.estimator.ModeKeys.EVAL:
            return get_evaluation_spec(logits, labels, num_classes)

    return _encoder_decoder_model_fn
