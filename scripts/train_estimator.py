# coding=utf-8
import argparse
import tensorflow as tf
import os

from builders import model_estimator_builder, dataset_builder
from tensorflow.contrib.distribute import MirroredStrategy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
tf.logging.set_verbosity(tf.logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    # base configs
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of images in each batch per gpu. ')
    parser.add_argument('--weight_decay', type=float, default=0.00004)
    parser.add_argument('--debug_mode', action='store_true',
                        help='Whether to use debug mode.')

    # training
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--epoch_start_i', type=int, default=0)
    parser.add_argument('--logs_root_path', type=str, default="./logs",
                        help='path to save log dirs')
    parser.add_argument('--logs_name', type=str, default="default",
                        help='part of log dir name')
    parser.add_argument('--clean_model_dir', action='store_true',
                        help='Whether to clean up the model dir if present.')
    parser.add_argument('--freeze_batch_norm', action='store_true',
                        help='Whether to freeze batch norm.')

    # multi-gpu configs
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='num of gpus')
    parser.add_argument('--gpu_devices', type=str, default="3",
                        help='select gpus, use in "CUDA_VISIBLE_DEVICES"')

    # learning rate
    parser.add_argument('--learning_policy',
                        type=str, default='poly',
                        help='[poly, step, piecewise]')
    parser.add_argument('--base_learning_rate',
                        type=float, default=1e-3,
                        help='used in [poly, step, piecewise]')

    parser.add_argument('--learning_rate_decay_factor',
                        type=float, default=0.1, help='exponential_decay')
    parser.add_argument('--learning_rate_decay_step',
                        type=int, default=2000, help='exponential_decay')

    parser.add_argument('--learning_power',
                        type=float, default=0.9, help='polynomial_decay')
    parser.add_argument('--end_learning_rate',
                        type=float, default=1e-5, help='polynomial_decay')
    parser.add_argument('--training_number_of_steps',
                        type=int, default=10700*15, help='polynomial_decay')

    parser.add_argument('--learning_rate_boundaries',
                        nargs='+', type=int, help='piecewise_constant_decay')
    parser.add_argument('--learning_rate_values',
                        nargs='+', type=float, help='piecewise_constant_decay')

    parser.add_argument('--slow_start_step',
                        type=int, default=0,
                        help='The number of steps used for training')
    parser.add_argument('--slow_start_learning_rate',
                        type=float, default=1e-4,
                        help='The number of steps used for training')

    # optimizer
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='The momentum value to use')
    parser.add_argument('--use_adam', action='store_true')

    # val related
    parser.add_argument('--validation_step', type=int, default=1,
                        help='How often to perform validation (epochs)')

    # dataset
    parser.add_argument('--dataset_name', type=str, default="",
                        help='')
    parser.add_argument('--dataset_dir', type=str, default="",
                        help='')
    parser.add_argument('--train_split_name', type=str,
                        default="train", help='')
    parser.add_argument('--val_split_name', type=str, default="val", help='')

    # image preprocessing and argument
    parser.add_argument('--train_crop_height', type=int, default=513)
    parser.add_argument('--train_crop_width', type=int, default=513)
    parser.add_argument('--eval_crop_height', type=int, default=513)
    parser.add_argument('--eval_crop_width', type=int, default=513)
    parser.add_argument('--min_resize_value', type=int, default=None)
    parser.add_argument('--max_resize_value', type=int, default=None)
    parser.add_argument('--resize_factor', type=float, default=None)
    parser.add_argument('--min_scale_factor', type=float, default=.5)
    parser.add_argument('--max_scale_factor', type=float, default=2.)
    parser.add_argument('--scale_factor_step_size', type=float, default=0.25)
    parser.add_argument('--num_readers', type=int, default=4)

    # model related
    parser.add_argument('--model', type=str, default="DeepLabV3",
                        help='The model you are using. '
                             'See model_estimator_builder.py for more')
    parser.add_argument('--frontend', type=str, default="ResNet101",
                        help='The frontend you are using. '
                        'See frontend_builder.py for more.')
    parser.add_argument('--pretrained_dir', type=str,
                        default="/ssd/zhangyiyang/data/slim",
                        help='The directory to save all pre-tranied models.')

    # resnet frontend params
    parser.add_argument('--output_stride', type=int, default=16,
                        help='')

    # save, log, summary
    parser.add_argument('--saving_every_n_steps', type=int, default=200,
                        help='')
    parser.add_argument('--logging_every_n_steps', type=int, default=20,
                        help='')
    parser.add_argument('--summary_every_n_steps', type=int, default=20,
                        help='')
    parser.add_argument('--summary_image_max_number', type=int, default=6,
                        help='')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    dataset_meta = dataset_builder.build_dataset_meta(args.dataset_name)

    def _train_input_fn():
        dataset_configs = dataset_builder.build_dataset_configs(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            crop_size=(args.train_crop_height, args.train_crop_width),
            min_resize_value=args.min_resize_value,
            max_resize_value=args.max_resize_value,
            resize_factor=args.resize_factor,
            min_scale_factor=args.min_scale_factor,
            max_scale_factor=args.max_scale_factor,
            scale_factor_step_size=args.scale_factor_step_size,
            num_readers=args.num_readers,
            should_shuffle=True,
            is_training=True,
        )
        dataset = dataset_builder.build_dataset(
            args.dataset_name, args.train_split_name, True, dataset_configs)
        return dataset

    def _val_input_fn():
        dataset_configs = dataset_builder.build_dataset_configs(
            dataset_dir=args.dataset_dir,
            batch_size=1,
            crop_size=(args.eval_crop_height, args.eval_crop_width),
            should_shuffle=False,
            is_training=False,
        )
        dataset = dataset_builder.build_dataset(
            args.dataset_name, args.val_split_name, True, dataset_configs)
        return dataset

    # save logs in  `./{logs_root_path}/logs-{datasets}-{model}-{logs_name}`
    logs_path = os.path.join(
        args.logs_root_path, 'logs-{}-{}-{}'.format(args.dataset_name,
                                                    args.model,
                                                    args.logs_name))
    if args.clean_model_dir:
        import shutil
        shutil.rmtree(logs_path, ignore_errors=True)

    strategy = None if args.num_gpus == 1 else MirroredStrategy(
        num_gpus=args.num_gpus)
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True

    model_fn = model_estimator_builder.build_model_fn(
        model_name=args.model,
        num_classes=dataset_meta.num_classes,
        frontend=args.frontend,
        pretrained_dir=args.pretrained_dir
    )

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=args.saving_every_n_steps,
        save_summary_steps=args.summary_every_n_steps,
        train_distribute=strategy,
        session_config=session_config,
    )

    params = {
        # learning rate
        "learning_policy": args.learning_policy,
        'base_learning_rate': args.base_learning_rate,
        'learning_rate_decay_step': args.learning_rate_decay_step,
        'learning_rate_decay_factor': args.learning_rate_decay_factor,
        'learning_rate_boundaries': args.learning_rate_boundaries,
        'learning_rate_values': args.learning_rate_values,
        'training_number_of_steps': args.training_number_of_steps,
        'learning_power': args.learning_power,
        'end_learning_rate': args.end_learning_rate,
        'slow_start_step': args.slow_start_step,
        'slow_start_learning_rate': args.slow_start_learning_rate,

        # optimizer
        'momentum': args.momentum,
        'use_adam': args.use_adam,

        # base
        'weight_decay': args.weight_decay,

        # model
        'output_stride': args.output_stride,
        'freeze_batch_norm': args.freeze_batch_norm,

        # debug mode
        'batch_size': args.batch_size,
        'dataset_name': args.dataset_name,
        # only in training procedure
        'summary_image_max_number': args.summary_image_max_number,
        'debug_mode': args.debug_mode,
    }

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=logs_path,
                                       config=config,
                                       params=params)
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_px_accuracy': 'train_px_accuracy',
        'train_mean_iou': 'train_mean_iou',
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=args.logging_every_n_steps)
    train_hooks = [logging_hook]

    for i in range(args.epoch_start_i, args.num_epochs):
        # train
        estimator.train(_train_input_fn, hooks=train_hooks)

        if i % args.validation_step == 0:
            # val every {validation_step} steps
            estimator.evaluate(_val_input_fn, args.num_val_images)
