import argparse
import tensorflow as tf
import os
import cv2
import numpy as np

from utils import helpers
from builders import model_estimator_builder, dataset_config_builders
from datasets import dataset_utils

tf.logging.set_verbosity(tf.logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    # multi-gpu configs
    parser.add_argument('--gpu_devices', type=str, default="3",
                        help='select gpus, use in "CUDA_VISIBLE_DEVICES"')

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
    parser.add_argument('--split_name', type=str, default="val", help='')
    
    # 图像预处理参数（包括图像增广）
    parser.add_argument('--crop_height', type=int, default=512)
    parser.add_argument('--crop_width', type=int, default=512)


    # 模型相关参数
    parser.add_argument('--model', type=str, default="Encoder-Decoder",
                        help='The model you are using. See model_estimator_builder.py for supported models')
    parser.add_argument('--frontend', type=str, default="ResNet101",
                        help='The frontend you are using. See frontend_builder.py for supported models')

    # pre-trained model 相关参数
    parser.add_argument('--target_path', type=str, default="./results",
                        help='path to save target pics.')
    parser.add_argument('--model_path', type=str, default="./ckpt",
                        help='path that saves all ckpt files.')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='ckpt files path.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    dataset_meta = dataset_utils.get_dataset_meta(args.dataset_name)

    def _test_input_fn(): 
        dataset_configs = dataset_config_builders.get_default_config(
            dataset_name=args.dataset_name,
            split_name=args.split_name, 
            dataset_dir=args.dataset_dir, 
            batch_size=1,
            crop_size=(args.crop_height, args.crop_width),
            num_readers=args.num_readers,
            is_training=False,
            should_shuffle=False,
        )
        dataset = dataset_utils.get_dataset(**dataset_configs)
        dataset = dataset_utils.get_estimator_dataset(dataset, False, dataset_meta.ignore_label)
        return dataset

    estimator = tf.estimator.Estimator(model_fn=model_estimator_builder.build_model_fn(args.model, dataset_meta.num_classes, None),
                                       model_dir=args.model_path)
    estimator_predictions = estimator.predict(input_fn=_test_input_fn,
                                              checkpoint_path=args.checkpoint_path)

    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path)

    # TODO: 保存预测图片到目标文件夹
    # for prediction in estimator_predictions:
    #     target_image = None
    #     cv2.imwrite(os.path.join(args.target_path, prediction['target_file_name']),
    #                 cv2.cvtColor(np.uint8(target_image), cv2.COLOR_RGB2BGR))
