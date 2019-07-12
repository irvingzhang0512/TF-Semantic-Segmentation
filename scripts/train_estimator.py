# coding=utf-8
import argparse
import tensorflow as tf
import os

from builders import model_estimator_builder, dataset_config_builders
from datasets import dataset_utils
from tensorflow.contrib.distribute import MirroredStrategy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
tf.logging.set_verbosity(tf.logging.INFO)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()

    # 训练基本参数
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0,
                        help='Start counting epochs from this number')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of images in each batch per gpu. '
                             'If there are multiple gpus, tf.data will get batch_size * num_gpus.')
    parser.add_argument('--logs_root_path', type=str, default="./logs",
                        help='path to save log dirs')
    parser.add_argument('--logs_name', type=str, default="default",
                        help='part of log dir name')

    # multi-gpu configs
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='num of gpus')
    parser.add_argument('--gpu_devices', type=str, default="3",
                        help='select gpus, use in "CUDA_VISIBLE_DEVICES"')

    # 学习率相关
    parser.add_argument('--learning_rate_start', type=float, default=0.001,
                        help='')
    parser.add_argument('--optimizer_decay', type=float, default=0.995,
                        help='')

    # val相关
    parser.add_argument('--validation_step', type=int, default=1,
                        help='How often to perform validation (epochs)')
    parser.add_argument('--num_val_images', type=int, default=20,
                        help='The number of images to used for validations')

    # 数据集类型
    parser.add_argument('--dataset_name', type=str, default="",
                        help='')
    parser.add_argument('--dataset_dir', type=str, default="",
                        help='')    
    parser.add_argument('--train_split_name', type=str, default="train", help='')
    parser.add_argument('--val_split_name', type=str, default="val", help='')
    
    # 图像预处理参数（包括图像增广）
    parser.add_argument('--crop_height', type=int, default=512)
    parser.add_argument('--crop_width', type=int, default=512)
    parser.add_argument('--min_resize_value', type=int, default=None)
    parser.add_argument('--max_resize_value', type=int, default=None)
    parser.add_argument('--resize_factor', type=float, default=None) 
    parser.add_argument('--min_scale_factor', type=float, default=1.)
    parser.add_argument('--max_scale_factor', type=float, default=1.) 
    parser.add_argument('--scale_factor_step_size', type=float, default=0.)
    parser.add_argument('--num_readers', type=int, default=1)


    # 模型相关参数
    parser.add_argument('--model', type=str, default="Encoder-Decoder",
                        help='The model you are using. See model_estimator_builder.py for supported models')
    parser.add_argument('--frontend', type=str, default="ResNet101",
                        help='The frontend you are using. See frontend_builder.py for supported models')

    # save, log, summary 相关参数
    parser.add_argument('--saving_every_n_steps', type=int, default=400,
                        help='')
    parser.add_argument('--logging_every_n_steps', type=int, default=50,
                        help='')
    parser.add_argument('--summary_every_n_steps', type=int, default=50,
                        help='')

    return parser.parse_args()


def get_learning_rate(config):
    # TODO: use learning rate
    return config.learning_rate_start


def get_default_optimizer(config):
    return tf.train.RMSPropOptimizer(learning_rate=get_learning_rate(config),
                                     decay=config.optimizer_decay)


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    dataset_meta = dataset_utils.get_dataset_meta(args.dataset_name)
    

    def _train_input_fn(): 
        dataset_configs = dataset_config_builders.get_default_config(
            dataset_name=args.dataset_name,
            split_name=args.train_split_name, 
            dataset_dir=args.dataset_dir, 
            batch_size=args.batch_size,
            crop_size=(args.crop_height, args.crop_width),
            min_resize_value=args.min_resize_value,
            max_resize_value=args.max_resize_value,
            resize_factor=args.resize_factor,
            min_scale_factor=args.min_scale_factor,
            max_scale_factor=args.max_scale_factor,
            scale_factor_step_size=args.scale_factor_step_size,
            num_readers=args.num_readers,
            is_training=True,
            should_shuffle=True,
        )
        dataset = dataset_utils.get_dataset(**dataset_configs)
        dataset = dataset_utils.get_estimator_dataset(dataset, True, dataset_meta.ignore_label)
        return dataset

    def _val_input_fn():
        dataset_configs = dataset_config_builders.get_default_config(
            dataset_name=args.dataset_name,
            split_name=args.val_split_name, 
            dataset_dir=args.dataset_dir, 
            batch_size=1,
            crop_size=(args.crop_height, args.crop_width),
            is_training=False,
            should_shuffle=True,
        )
        dataset = dataset_utils.get_dataset(**dataset_configs)
        dataset = dataset_utils.get_estimator_dataset(dataset, True, dataset_meta.ignore_label)
        return dataset

    # 把logs保存到 ./{logs_root_path}/logs-{datasets}-{model}-{losg_name} 中
    logs_path = os.path.join(args.logs_root_path, 'logs-{}-{}-{}'.format(args.dataset_name, args.model, args.logs_name))

    strategy = None if args.num_gpus == 1 else MirroredStrategy(num_gpus=args.num_gpus)
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    optimizer = get_default_optimizer(args)
    estimator = tf.estimator.Estimator(model_fn=model_estimator_builder.build_model_fn(args.model,
                                                                                       dataset_meta.num_classes, 
                                                                                       optimizer),
                                       model_dir=logs_path,
                                       config=tf.estimator.RunConfig(
                                           save_checkpoints_steps=args.saving_every_n_steps,
                                           log_step_count_steps=args.logging_every_n_steps,
                                           save_summary_steps=args.summary_every_n_steps,
                                           train_distribute=strategy,
                                           session_config=session_config,
                                       ),
                                       params={})
    for i in range(args.epoch_start_i, args.num_epochs):
        # train
        estimator.train(_train_input_fn)

        if i % args.validation_step == 0:
            # val every {validation_step} steps
            estimator.evaluate(_val_input_fn, args.num_val_images)
