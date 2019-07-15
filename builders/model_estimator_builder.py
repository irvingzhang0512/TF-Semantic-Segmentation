from estimator_models.Encoder_Decoder import get_encoder_decoder_model_fn
from estimator_models.FC_DenseNet_Tiramisu import get_fc_densenet_model_fn
from estimator_models.DeepLabV3 import get_deeplabv3_model_fn


def build_model_fn(model_name, num_classes, frontend=None, pretrained_dir=None):
    if model_name == "Encoder-Decoder" or model_name == "Encoder-Decoder-Skip":
        return get_encoder_decoder_model_fn(preset_model=model_name,
                                            num_classes=num_classes)
    if model_name == "FC-DenseNet56" or model_name == "FC-DenseNet67" or model_name == "FC-DenseNet103":
        return get_fc_densenet_model_fn(preset_model=model_name,
                                        num_classes=num_classes)
    if model_name == "DeepLabV3":
        return get_deeplabv3_model_fn(num_classes=num_classes,
                                      frontend=frontend,
                                      pretrained_dir=pretrained_dir)

    raise ValueError('unknown model name {}'.format(model_name))
