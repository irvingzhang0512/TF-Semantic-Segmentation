import tensorflow as tf
from estimator_models.estimator_model_utils import get_estimator_spec, preprocess_zero_to_one
from models.Encoder_Decoder import build_encoder_decoder


def get_encoder_decoder_model_fn(num_classes, preset_model="Encoder-Decoder",):
    def _encoder_decoder_model_fn(features, labels, mode, params, config):
        features['image'] = preprocess_zero_to_one(features['image'])
        logits = build_encoder_decoder(features['images'], num_classes, 
                                        preset_model=preset_model,
                                        weight_decay=params['weight_decay'])

        return get_estimator_spec(mode, logits, None, labels=labels, num_classes=num_classes, params=params)

    return _encoder_decoder_model_fn
