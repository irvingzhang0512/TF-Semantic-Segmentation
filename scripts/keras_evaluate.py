import os
import argparse
import tensorflow as tf
from segmentation.builders import model_builder, dataset_builder
from segmentation.utils import losses_utils, metrics_utils
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152


def parse_args():
    parser = argparse.ArgumentParser()

    # multi-gpu configs
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--gpu_devices', type=str, default="3")

    # dataset
    parser.add_argument('--dataset_name', type=str, default="", help='')
    parser.add_argument('--dataset_dir', type=str, default="", help='')
    parser.add_argument('--split_name', type=str, default="val", help='')
    parser.add_argument('--eval_crop_height', type=int, default=513)
    parser.add_argument('--eval_crop_width', type=int, default=513)

    # model related
    parser.add_argument('--model_type', type=str, default="", help='')
    parser.add_argument('--backend_type', type=str, default="", help='')
    parser.add_argument('--model_weights', type=str, default=None, help='')
    parser.add_argument('--output_stride', type=int, default=16, help='')
    parser.add_argument('--fine_tune_batch_norm', action='store_true',
                        help='Whether to fine tune bach norm.')

    return parser.parse_args()


def _get_datasets(args):
    val_dataset_configs = dataset_builder.build_dataset_configs(
        dataset_dir=args.dataset_dir,
        batch_size=1,
        crop_size=(args.eval_crop_height, args.eval_crop_width),
        should_shuffle=False,
        is_training=False,
        should_repeat=True,
    )
    val_dataset = dataset_builder.build_dataset(
        args.dataset_name, args.split_name, True, val_dataset_configs)

    return val_dataset


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    if tf.__version__.split('.')[0] == "1":
        tf.logging.set_verbosity(tf.logging.INFO)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True  # 不一定全部占满显存, 按需分配
        sess = tf.Session(config=tf_config)
        tf.keras.backend.set_session(sess)

    dataset_meta = dataset_builder.build_dataset_meta(args.dataset_name)
    val_dataset = _get_datasets(args)

    keras_model = model_builder.build_model(
        model_type=args.model_type,
        backend_type=args.backend_type,
        weights=args.model_weights,
        num_classes=dataset_meta.num_classes,
        OS=args.output_stride,
        input_shape=(args.eval_crop_height, args.eval_crop_width, 3),
        fine_tune_batch_norm=args.fine_tune_batch_norm,
    )

    if args.num_gpus > 1:
        keras_model = tf.keras.utils.multi_gpu_model(
            keras_model,
            gpus=args.num_gpus,
        )

    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=losses_utils.cross_entropy_loss,
        metrics=[metrics_utils.accuracy, metrics_utils.mean_iou],
    )

    # evaluate
    keras_model.evaluate(
        val_dataset,
        steps=dataset_meta.splits_to_sizes[args.split_name],
    )
