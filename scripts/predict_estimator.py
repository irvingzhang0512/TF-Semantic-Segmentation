import argparse
import tensorflow as tf
import os
import cv2
import numpy as np
from utils import helpers

from builders import model_estimator_builder, dataset_builder

tf.logging.set_verbosity(tf.logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    # 基本参数
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of images in each batch')
    parser.add_argument('--target_path', type=str, default="./results",
                        help='path to save target pics.')
    parser.add_argument('--model_path', type=str, default="./ckpt",
                        help='path that saves all ckpt files.')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='ckpt files path.')

    # 数据集类型
    parser.add_argument('--datasets', type=str, default="CamVid",
                        help='Dataset you are using.')
    parser.add_argument('--dataset_root_path', type=str, default="./CamVid",
                        help='Dataset you are using.')

    # 模型相关参数
    parser.add_argument('--model', type=str, default="Encoder-Decoder",
                        help='The model you are using. See model_estimator_builder.py for supported models')
    parser.add_argument('--frontend', type=str, default="ResNet101",
                        help='The frontend you are using. See frontend_builder.py for supported models')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset_configs = dataset_builder.get_default_configs(args.dataset)
    dataset_configs['preprocessing_type'] = 'tf'
    test_set, num_classes, label_values = dataset_builder.build_dataset(args.dataset, 'test', dataset_configs)

    # TODO: 设置 feature columns，其中输入图片分为 images 和 target_file_names
    def test_input_fn():
        return test_set.make_one_shot_iterator().get_next()

    estimator = tf.estimator.Estimator(model_fn=model_estimator_builder.build_model_fn(args.model, num_classes, None),
                                       model_dir=args.model_path)
    estimator_predictions = estimator.predict(input_fn=test_input_fn,
                                              checkpoint_path=args.checkpoint_path)

    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path)

    # 保存预测图片到目标文件夹
    for prediction in estimator_predictions:
        target_image = helpers.colour_code_segmentation(prediction['class_ids'], label_values)
        cv2.imwrite(os.path.join(args.target_path, prediction['target_file_name']),
                    cv2.cvtColor(np.uint8(target_image), cv2.COLOR_RGB2BGR))
