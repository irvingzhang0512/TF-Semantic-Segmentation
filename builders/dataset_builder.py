from datasets.camvid import get_camvid_dataset, get_camvid_default_config


__all__ = ['build_dataset', 'get_default_configs']


def build_dataset(dataset_type, mode, configs=None):
    if dataset_type == 'CamVid':
        if configs is None:
            configs = get_camvid_default_config()
        return get_camvid_dataset(mode, **configs)

    raise ValueError('unknown datasets type {}'.format(dataset_type))


def get_default_configs(dataset_type):
    if dataset_type == 'CamVid':
        return get_camvid_default_config()

    raise ValueError('unknown datasets type {}'.format(dataset_type))
