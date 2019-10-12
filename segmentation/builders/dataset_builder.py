import collections
from ..datasets.dataset_utils import get_dataset, get_estimator_dataset

__all__ = ['build_dataset', 'build_dataset_meta', 'build_dataset_configs']

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
        # Splits of the dataset into training, val and test.
        'splits_to_sizes',
        'num_classes',  # Number of semantic classes, including the
                        # background class (if exists). For example, there
                        # are 20 foreground classes + 1 background class in
                        # the PASCAL VOC 2012 dataset. Thus, we set
                        # num_classes=21.
        'ignore_label',  # Ignore label value.
    ])

_CITYSCAPES_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 2975,
        'val': 500,
    },
    num_classes=19,
    ignore_label=255,
)

_PASCAL_VOC_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 1464,
        'train_aug': 10582,
        'trainval': 2913,
        'val': 1449,
    },
    num_classes=21,
    ignore_label=255,
)

_ADE20K_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 20210,  # num of samples in images/training
        'val': 2000,  # num of samples in images/validation
    },
    num_classes=151,
    ignore_label=0,
)

_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
}


def build_dataset(dataset_name, split_name, with_label, dataset_configs):
    if dataset_name not in _DATASETS_INFORMATION:
        raise ValueError('dataset {} not supported'.format(dataset_name))
    dataset_meta = _DATASETS_INFORMATION[dataset_name]

    if split_name not in dataset_meta.splits_to_sizes:
        raise ValueError('unknown splits {} for dataset {}'.format(
            split_name, dataset_name))

    dataset_configs['dataset_name'] = dataset_name
    dataset_configs['split_name'] = split_name
    dataset_configs['ignore_label'] = dataset_meta.ignore_label
    dataset = get_dataset(**dataset_configs)
    dataset = get_estimator_dataset(
        dataset, with_label, dataset_meta.ignore_label)
    return dataset


def build_dataset_meta(dataset_name):
    if dataset_name not in _DATASETS_INFORMATION:
        raise ValueError('dataset {} not supported'.format(dataset_name))
    return _DATASETS_INFORMATION[dataset_name]


def build_dataset_configs(dataset_dir,
                          batch_size,
                          crop_size,
                          min_resize_value=None,
                          max_resize_value=None,
                          resize_factor=None,
                          min_scale_factor=1.,
                          max_scale_factor=1.,
                          scale_factor_step_size=0,
                          model_variant=None,
                          num_readers=1,
                          should_shuffle=False,
                          should_repeat=False,
                          is_training=False,):
    return {
        "dataset_dir": dataset_dir,
        "batch_size": batch_size,
        "crop_size": crop_size,
        "min_resize_value": min_resize_value,
        "max_resize_value": max_resize_value,
        "resize_factor": resize_factor,
        "min_scale_factor": min_scale_factor,
        "max_scale_factor": max_scale_factor,
        "scale_factor_step_size": scale_factor_step_size,
        "model_variant": model_variant,
        "num_readers": num_readers,
        "should_shuffle": should_shuffle,
        "should_repeat": should_repeat,
        "is_training": is_training,
    }
