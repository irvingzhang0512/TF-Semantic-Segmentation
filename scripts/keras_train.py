import numpy as np
import os
import argparse
import tensorflow as tf
from segmentation.builders import model_builder, dataset_builder
from segmentation.utils import losses_utils, metrics_utils, \
    training_utils
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# tf.keras.backend.set_learning_phase(True)


def parse_args():
    parser = argparse.ArgumentParser()

    # base configs
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of images in each batch per gpu. ')
    parser.add_argument('--weight_decay', type=float, default=0.00004)
    parser.add_argument('--early_stopping_epochs', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default=None)

    # training
    parser.add_argument('--training_steps', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--initial_epoch', type=int, default=0)
    parser.add_argument('--logs_root_path', type=str, default="./logs",
                        help='path to save log dirs')
    parser.add_argument('--logs_name', type=str, default="default",
                        help='part of log dir name')
    parser.add_argument('--clean_model_dir', action='store_true',
                        help='Whether to clean up the model dir if present.')

    # optimizer
    parser.add_argument('--optimizer_type', type=str, default="adam",
                        help='adam/sgd/rmsprop')

    # multi-gpu configs
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--gpu_devices', type=str, default="3")

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
                        type=float, default=.0, help='polynomial_decay')
    parser.add_argument('--training_number_of_steps',
                        type=int, default=30000, help='polynomial_decay')
    parser.add_argument('--learning_rate_boundaries',
                        nargs='+', type=int, help='piecewise_constant_decay')
    parser.add_argument('--learning_rate_values',
                        nargs='+', type=float, help='piecewise_constant_decay')

    # dataset
    parser.add_argument('--dataset_name', type=str, default="", help='')
    parser.add_argument('--dataset_dir', type=str, default="", help='')
    parser.add_argument('--train_split_name', type=str, default="", help='')
    parser.add_argument('--val_split_name', type=str, default="val", help='')
    parser.add_argument('--num_readers', type=int, default=4)

    # image preprocessing and argument
    parser.add_argument('--crop_height', type=int, default=513)
    parser.add_argument('--crop_width', type=int, default=513)
    parser.add_argument('--min_resize_value', type=int, default=None)
    parser.add_argument('--max_resize_value', type=int, default=None)
    parser.add_argument('--resize_factor', type=float, default=None)
    parser.add_argument('--min_scale_factor', type=float, default=.5)
    parser.add_argument('--max_scale_factor', type=float, default=2.)
    parser.add_argument('--scale_factor_step_size', type=float, default=0.25)

    # model related
    parser.add_argument('--model_type', type=str, default="", help='')
    parser.add_argument('--backend_type', type=str, default="", help='')
    parser.add_argument('--model_weights', type=str, default=None, help='')
    parser.add_argument('--output_stride', type=int, default=16, help='')
    parser.add_argument('--fine_tune_batch_norm', action='store_true',
                        help='Whether to fine tune bach norm.')

    # summary
    parser.add_argument('--summary_every_n_steps', type=int, default=100)

    return parser.parse_args()


def _get_datasets(args):
    # train dataset
    train_dataset_configs = dataset_builder.build_dataset_configs(
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
        should_shuffle=True,
        is_training=True,
        should_repeat=True,
    )
    train_dataset = dataset_builder.build_dataset(
        args.dataset_name, args.train_split_name, True, train_dataset_configs)

    # val dataset
    val_dataset_configs = dataset_builder.build_dataset_configs(
        dataset_dir=args.dataset_dir,
        batch_size=1,
        crop_size=(args.crop_height, args.crop_width),
        min_resize_value=args.min_resize_value,
        max_resize_value=args.max_resize_value,
        should_shuffle=False,
        is_training=False,
        should_repeat=True,
    )
    val_dataset = dataset_builder.build_dataset(
        args.dataset_name, args.val_split_name, True, val_dataset_configs)

    return train_dataset, val_dataset


def _get_model_dir_name(args):
    if args.learning_policy == "piecewise":
        lr = args.learning_rate_values[0]
    else:
        lr = args.base_learning_rate
    return os.path.join(
        args.logs_root_path,
        'logs-{}-{}_{}-{}-lr_{}_{}-wd{}-{}'.format(
            args.dataset_name,  # dataset
            args.model_type, args.backend_type,  # model
            args.optimizer_type,  # optimizer
            args.learning_policy, lr,  # learning rate
            args.weight_decay,  # l2 loss
            args.logs_name,))  # others


def _add_l2(model, weight_decay):
    def _do_add_l2(layer):
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_regularizer = tf.keras.regularizers.l2(weight_decay)

    for l in model.layers:
        if isinstance(l, tf.keras.Model):
            for ll in l.layers:
                _do_add_l2(ll)
        else:
            _do_add_l2(l)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    if tf.__version__.split('.')[0] == "1":
        tf.logging.set_verbosity(tf.logging.INFO)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True  # 不一定全部占满显存, 按需分配
        sess = tf.Session(config=tf_config)
        tf.keras.backend.set_session(sess)

    dataset_meta = dataset_builder.build_dataset_meta(args.dataset_name)

    train_dataset, val_dataset = _get_datasets(args)

    model_dir = _get_model_dir_name(args)
    if args.clean_model_dir:
        import shutil
        shutil.rmtree(model_dir, ignore_errors=True)

    if args.ckpt is not None:
        model_weights = None
    else:
        model_weights = args.model_weights

    keras_model = model_builder.build_model(
        model_type=args.model_type,
        backend_type=args.backend_type,
        weights=model_weights,
        num_classes=dataset_meta.num_classes,
        OS=args.output_stride,
        input_shape=(args.crop_height, args.crop_width, 3),
        fine_tune_batch_norm=args.fine_tune_batch_norm,
    )

    if args.ckpt is not None:
        keras_model.load_weights(args.ckpt)

    if args.num_gpus > 1:
        keras_model = tf.keras.utils.multi_gpu_model(
            keras_model,
            gpus=args.num_gpus,
        )

    learning_rate_fn = training_utils.get_keras_learning_rate_fn(
        learning_policy=args.learning_policy,

        # for exponential decay and polynomial decay
        base_learning_rate=args.base_learning_rate,

        # exponential_decay
        learning_rate_decay_step=args.learning_rate_decay_step,
        learning_rate_decay_factor=args.learning_rate_decay_factor,

        # polynomial_decay
        training_number_of_steps=args.training_number_of_steps,
        learning_power=args.learning_power,
        end_learning_rate=args.end_learning_rate,

        # piecewise_constant_decay
        learning_rate_boundaries=args.learning_rate_boundaries,
        learning_rate_values=args.learning_rate_values,
    )

    # loss_fn = losses_utils.build_cross_entropy_loss_fn(
    #     num_classes=dataset_meta.num_classes,
    # )
    # _add_l2(keras_model, args.weight_decay)

    loss_fn = losses_utils.build_total_loss_fn(
        num_classes=dataset_meta.num_classes,
        trainable_variables=keras_model.trainable_variables,
        weight_decay=args.weight_decay,
    )

    keras_model.compile(
        optimizer=training_utils.get_optimizer(
            args.optimizer_type,
            learning_rate_fn,
        ),
        loss=loss_fn,
        metrics=[
            metrics_utils.build_accuracy_fn(dataset_meta.num_classes),
            metrics_utils.build_mean_iou_fn(dataset_meta.num_classes)
        ],
    )

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            model_dir,
            update_freq=args.summary_every_n_steps,
        ),
        tf.keras.callbacks.EarlyStopping(
            restore_best_weights=True,
            patience=args.early_stopping_epochs,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
            save_weights_only=True,
        ),
    ]

    if args.training_steps != 0:
        training_steps = args.training_steps
    else:
        training_steps = int(np.ceil(1.0*dataset_meta.splits_to_sizes[
            args.train_split_name]/args.batch_size/args.num_gpus))
    # training
    keras_model.fit(
        train_dataset,
        epochs=args.epochs,
        initial_epoch=args.initial_epoch,
        # validation_steps=10,
        steps_per_epoch=training_steps,
        validation_steps=dataset_meta.splits_to_sizes[args.val_split_name],
        validation_data=val_dataset,
        validation_freq=1,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    main(parse_args())
