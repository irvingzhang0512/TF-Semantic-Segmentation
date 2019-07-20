import argparse
import tensorflow as tf

from builders import model_estimator_builder, dataset_builder

tf.logging.set_verbosity(tf.logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    # base
    parser.add_argument('--model_path', type=str, default="./logs")
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--gpu_devices', type=str, default="3")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.00004)

    # debug mode
    parser.add_argument('--debug_mode', action='store_true',
                        help='Whether to use debug mode.')

    # dataset
    parser.add_argument('--dataset_name', type=str, default="")
    parser.add_argument('--dataset_dir', type=str, default="")
    parser.add_argument('--split_name', type=str, default="val")
    parser.add_argument('--crop_height', type=int, default=513)
    parser.add_argument('--crop_width', type=int, default=513)
    parser.add_argument('--num_readers', type=int, default=4)

    # model
    parser.add_argument('--model', type=str, default="DeepLabV3")
    parser.add_argument('--frontend', type=str, default="ResNet101")
    parser.add_argument('--output_stride', type=int, default=16,
                        help='')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    dataset_meta = dataset_builder.build_dataset_meta(args.dataset_name)
    def _val_input_fn():
        dataset_configs = dataset_builder.build_dataset_configs(
            dataset_dir=args.dataset_dir, 
            batch_size=args.batch_size,
            crop_size=(args.crop_height, args.crop_width),
            should_shuffle=False,
            is_training=False,
        )
        dataset = dataset_builder.build_dataset(args.dataset_name, args.split_name, True, dataset_configs)
        return dataset

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    estimator = tf.estimator.Estimator(model_fn=model_estimator_builder.build_model_fn(model_name=args.model,
                                                                                       num_classes=dataset_meta.num_classes, 
                                                                                       frontend=args.frontend,
                                                                                       pretrained_dir=None),
                                       model_dir=args.model_path,
                                       config=tf.estimator.RunConfig(session_config=session_config),
                                       params={
                                           'weight_decay': args.weight_decay,

                                           # model
                                           'output_stride': args.output_stride,

                                           # debug mode
                                           'batch_size': args.batch_size,
                                           'dataset_name': args.dataset_name,
                                        }
                                    )
    estimator.evaluate(_val_input_fn, checkpoint_path=args.checkpoint_path)
