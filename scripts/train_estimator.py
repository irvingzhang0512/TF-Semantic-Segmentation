# coding=utf-8
import argparse
import tensorflow as tf
import os

from builders import model_estimator_builder, dataset_builder
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
    parser.add_argument('--datasets', type=str, default="CamVid",
                        help='Dataset you are using.')
    parser.add_argument('--dataset_root_path', type=str, default="./CamVid",
                        help='Dataset you are using.')

    # 图像预处理参数（包括图像增广）
    parser.add_argument('--crop_height', type=int, default=512,
                        help='Height of cropped input image to network')
    parser.add_argument('--crop_width', type=int, default=512,
                        help='Width of cropped input image to network')
    parser.add_argument('--h_flip', type=str2bool, default=True,
                        help='Whether to randomly flip the image horizontally for data augmentation')
    parser.add_argument('--v_flip', type=str2bool, default=False,
                        help='Whether to randomly flip the image vertically for data augmentation')
    parser.add_argument('--brightness', type=float, default=0.1,
                        help='Whether to randomly change the image brightness for data augmentation. '
                             'Specifies the max brightness change as a factor between 0.0 and 1.0. '
                             'For example, 0.1 represents a max brightness change of 10%% (+-).')
    parser.add_argument('--rotation', type=float, default=None,
                        help='Whether to randomly rotate the image for data augmentation. '
                             'Specifies the max rotation angle in degrees.')

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

    dataset_configs = dataset_builder.get_default_configs(args.dataset)
    dataset_configs['crop_height'] = args.crop_height
    dataset_configs['crop_width'] = args.crop_width
    dataset_configs['h_flip'] = args.h_flip
    dataset_configs['v_flip'] = args.v_flip
    dataset_configs['brightness'] = args.brightness
    dataset_configs['rotation'] = args.rotation
    dataset_configs['batch_size'] = args.batch_size
    dataset_configs['root_path'] = args.dataset_root_path
    dataset_configs['preprocessing_type'] = 'tf'

    num_classes = 32

    def _train_input_fn():
        train_set, _, _ = dataset_builder.build_dataset(args.dataset, 'train', dataset_configs)
        return train_set.make_one_shot_iterator().get_next()

    def _val_input_fn():
        val_set, _, _ = dataset_builder.build_dataset(args.dataset, 'val', dataset_configs)
        return val_set.make_one_shot_iterator().get_next()

    # 把logs保存到 ./{logs_root_path}/logs-{datasets}-{model}-{losg_name} 中
    logs_path = os.path.join(args.logs_root_path, 'logs-{}-{}-{}'.format(args.dataset, args.model, args.logs_name))

    strategy = None if args.num_gpus == 1 else MirroredStrategy(num_gpus=args.num_gpus)
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    optimizer = get_default_optimizer(args)
    estimator = tf.estimator.Estimator(model_fn=model_estimator_builder.build_model_fn(args.model,
                                                                                       num_classes, optimizer),
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
