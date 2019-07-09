import argparse
import tensorflow as tf

from builders import model_estimator_builder, dataset_builder

tf.logging.set_verbosity(tf.logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    # 基本参数
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of images in each batch')
    parser.add_argument('--model_path', type=str, default="./ckpt",
                        help='path that saves all ckpt files.')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='ckpt files path.')

    # 图像预处理参数
    parser.add_argument('--crop_height', type=int, default=512,
                        help='Height of cropped input image to network')
    parser.add_argument('--crop_width', type=int, default=512,
                        help='Width of cropped input image to network')

    # 数据集类型
    parser.add_argument('--datasets', type=str, default="CamVid",
                        help='Dataset you are using.')
    parser.add_argument('--dataset_root_path', type=str, default="./CamVid",
                        help='root path to save datasets files.')

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
    dataset_configs['crop_height'] = args.crop_height
    dataset_configs['crop_width'] = args.crop_width
    val_set, num_classes, label_values = dataset_builder.build_dataset(args.dataset, 'val', dataset_configs)

    def _val_input_fn():
        return val_set.make_one_shot_iterator().get_next()

    estimator = tf.estimator.Estimator(model_fn=model_estimator_builder.build_model_fn(args.model, num_classes, None),
                                       model_dir=args.model_path)
    estimator_evaluations = estimator.evaluate(input_fn=_val_input_fn,
                                               checkpoint_path=args.checkpoint_path)
