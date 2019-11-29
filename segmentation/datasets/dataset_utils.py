import os
import tensorflow as tf
from .deeplab_utils import input_preprocess
from . import common


__all__ = ['get_dataset', 'get_estimator_dataset']


# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'


def _get_all_files(dataset_dir, split_name):
    """Gets all the files to read data from.

    Returns:
      A list of input files.
    """
    file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir,
                                file_pattern % split_name)
    return tf.io.gfile.glob(file_pattern)


def get_dataset(dataset_name,
                split_name,
                ignore_label,
                dataset_dir,
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
                is_training=False,
                should_shuffle=False,
                should_repeat=False):
    """Initializes the dataset.

    Args:
      dataset_name: Dataset name.
      split_name: A train/val Split name.
      dataset_dir: The directory of the dataset sources.
      batch_size: Batch size.
      crop_size: The size used to crop the image and label.
      min_resize_value: Desired size of the smaller image side.
      max_resize_value: Maximum allowed size of the larger image side.
      resize_factor: Resized dimensions are multiple of factor plus one.
      min_scale_factor: Minimum scale factor value.
      max_scale_factor: Maximum scale factor value.
      scale_factor_step_size: The step size from min scale factor to max scale
        factor. The input is randomly scaled based on the value of
        (min_scale_factor, max_scale_factor, scale_factor_step_size).
      model_variant: Model variant (string) for choosing how to mean-subtract
        the images.
      num_readers: Number of readers for data provider.
      is_training: Boolean, if dataset is for training or not.
      should_shuffle: Boolean, if should shuffle the input data.
      should_repeat: Boolean, if should repeat the input data.

    Raises:
      ValueError: Dataset name and split name are not supported.
    """
    def _parse_function(example_proto):
        """Function to parse the example proto.

        Args:
          example_proto: Proto in the format of tf.Example.

        Returns:
          A dictionary with parsed image, label, height, width and image name.

        Raises:
          ValueError: Label is of wrong shape.
        """
        # Currently only supports jpeg and png.
        # Need to use this logic because the shape is not known for
        # tf.image.decode_image and we rely on this info to
        # extend label if necessary.
        def _decode_image(content, channels):
            return tf.cond(
                tf.image.is_jpeg(content),
                lambda: tf.image.decode_jpeg(content, channels),
                lambda: tf.image.decode_png(content, channels))

        features = {
            common.TF_RECORD_IMAGE_ENCODED:
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            common.TF_RECORD_IMAGE_FILENAME:
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            common.TF_RECORD_IMAGE_FORMAT:
                tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
            common.TF_RECORD_IMAGE_HEIGHT:
                tf.io.FixedLenFeature((), tf.int64, default_value=0),
            common.TF_RECORD_IMAGE_WIDTH:
                tf.io.FixedLenFeature((), tf.int64, default_value=0),
            common.TF_RECORD_IMAGE_SEGMENTATION_CLASS_ENCODED:
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            common.TF_RECORD_IMAGE_SEGMENTATION_CLASS_FORMAT:
                tf.io.FixedLenFeature((), tf.string, default_value='png'),
        }

        parsed_features = tf.io.parse_single_example(example_proto, features)

        image = _decode_image(
            parsed_features[common.TF_RECORD_IMAGE_ENCODED], channels=3)

        label = None
        if split_name != common.TEST_SET:
            label = _decode_image(
                parsed_features[
                    common.TF_RECORD_IMAGE_SEGMENTATION_CLASS_ENCODED],
                channels=1
            )

        image_name = parsed_features[common.TF_RECORD_IMAGE_FILENAME]
        if image_name is None:
            image_name = tf.constant('')

        sample = {
            common.IMAGE: image,
            common.IMAGE_NAME: image_name,
            common.HEIGHT: parsed_features[common.TF_RECORD_IMAGE_HEIGHT],
            common.WIDTH: parsed_features[common.TF_RECORD_IMAGE_WIDTH],
        }

        if label is not None:
            if label.get_shape().ndims == 2:
                label = tf.expand_dims(label, 2)
            elif label.get_shape().ndims == 3 and label.shape.dims[2] == 1:
                pass
            else:
                raise ValueError('Input label shape must be[height, width],'
                                 ' or [height, width, 1].')

            label.set_shape([None, None, 1])

            sample[common.LABELS_CLASS] = label

        return sample

    def _preprocess_image(sample):
        """Preprocesses the image and label.

        Args:
          sample: A sample containing image and label.

        Returns:
          sample: Sample with preprocessed image and label.

        Raises:
          ValueError: Ground truth label not provided during training.
        """
        image = sample[common.IMAGE]
        label = sample[common.LABELS_CLASS]

        original_image, image, label = \
            input_preprocess.preprocess_image_and_label(
                image=image,
                label=label,
                crop_height=crop_size[0],
                crop_width=crop_size[1],
                min_resize_value=min_resize_value,
                max_resize_value=max_resize_value,
                resize_factor=resize_factor,
                min_scale_factor=min_scale_factor,
                max_scale_factor=max_scale_factor,
                scale_factor_step_size=scale_factor_step_size,
                ignore_label=ignore_label,
                is_training=is_training,
                model_variant=model_variant)

        sample[common.IMAGE] = image

        if not is_training:
            # Original image is only used during visualization.
            sample[common.ORIGINAL_IMAGE] = original_image

        if label is not None:
            sample[common.LABEL] = label

        # Remove LABEL_CLASS key in the sample since it is only used to
        # derive label and not used in training and evaluation.
        sample.pop(common.LABELS_CLASS, None)
        return sample

    files = _get_all_files(dataset_dir, split_name)
    dataset = (
        tf.data.TFRecordDataset(files, num_parallel_reads=num_readers)
        .map(_parse_function, num_parallel_calls=num_readers)
        .map(_preprocess_image, num_parallel_calls=num_readers))

    if should_shuffle:
        dataset = dataset.shuffle(buffer_size=100)

    if should_repeat:
        dataset = dataset.repeat()  # Repeat forever for training.
    else:
        dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size).prefetch(batch_size)

    return dataset


def get_estimator_dataset(dataset, with_label, ignore_label=255):
    def _parse_with_label_dataset(sample):
        labels = sample[common.LABEL]
        return sample[common.IMAGE], labels

    def _parse_non_with_label_dataset(sample):
        return sample[common.IMAGE]

    if with_label:
        _parse_dataset = _parse_with_label_dataset
    else:
        _parse_dataset = _parse_non_with_label_dataset
    return dataset.map(_parse_dataset)
