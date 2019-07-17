# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper for providing semantic segmentaion data.

The SegmentationDataset class provides both images and annotations (semantic
segmentation and/or instance segmentation) for TensorFlow. Currently, we
support the following datasets:

1. PASCAL VOC 2012 (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

PASCAL VOC 2012 semantic segmentation dataset annotates 20 foreground objects
(e.g., bike, person, and so on) and leaves all the other semantic classes as
one background class. The dataset contains 1464, 1449, and 1456 annotated
images for the training, validation and test respectively.

2. Cityscapes dataset (https://www.cityscapes-dataset.com)

The Cityscapes dataset contains 19 semantic labels (such as road, person, car,
and so on) for urban street scenes.

3. ADE20K dataset (http://groups.csail.mit.edu/vision/datasets/ADE20K)

The ADE20K dataset contains 150 semantic labels both urban street scenes and
indoor scenes.

References:
  M. Everingham, S. M. A. Eslami, L. V. Gool, C. K. I. Williams, J. Winn,
  and A. Zisserman, The pascal visual object classes challenge a retrospective.
  IJCV, 2014.

  M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson,
  U. Franke, S. Roth, and B. Schiele, "The cityscapes dataset for semantic urban
  scene understanding," In Proc. of CVPR, 2016.

  B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, A. Torralba, "Scene Parsing
  through ADE20K dataset", In Proc. of CVPR, 2017.
"""

import collections
import os
import tensorflow as tf
from datasets.deeplab_utils import input_preprocess


__all__ = ['get_dataset', 'get_estimator_dataset']


LABELS_CLASS = 'labels_class'
IMAGE = 'image'
ORIGINAL_IMAGE = 'original_image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
TEST_SET = 'test'


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
  return tf.gfile.Glob(file_pattern)



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
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/segmentation/class/format':
            tf.FixedLenFeature((), tf.string, default_value='png'),
    }

    parsed_features = tf.parse_single_example(example_proto, features)

    image = _decode_image(parsed_features['image/encoded'], channels=3)

    label = None
    if split_name != TEST_SET:
      label = _decode_image(
          parsed_features['image/segmentation/class/encoded'], channels=1)

    image_name = parsed_features['image/filename']
    if image_name is None:
      image_name = tf.constant('')

    sample = {
        IMAGE: image,
        IMAGE_NAME: image_name,
        HEIGHT: parsed_features['image/height'],
        WIDTH: parsed_features['image/width'],
    }

    if label is not None:
      if label.get_shape().ndims == 2:
        label = tf.expand_dims(label, 2)
      elif label.get_shape().ndims == 3 and label.shape.dims[2] == 1:
        pass
      else:
        raise ValueError('Input label shape must be [height, width], or '
                          '[height, width, 1].')

      label.set_shape([None, None, 1])

      sample[LABELS_CLASS] = label

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
    image = sample[IMAGE]
    label = sample[LABELS_CLASS]

    original_image, image, label = input_preprocess.preprocess_image_and_label(
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

    sample[IMAGE] = image

    if not is_training:
      # Original image is only used during visualization.
      sample[ORIGINAL_IMAGE] = original_image

    if label is not None:
      sample[LABEL] = label

    # Remove LABEL_CLASS key in the sample since it is only used to
    # derive label and not used in training and evaluation.
    sample.pop(LABELS_CLASS, None)
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
    labels = sample[LABEL]
    sample.pop(LABEL, None)
    return sample, labels

  def _parse_non_with_label_dataset(sample):
    return sample

  _parse_dataset = _parse_with_label_dataset if with_label else _parse_non_with_label_dataset
  return dataset.map(_parse_dataset)

