# Dataset Utils

+ [Dataset Utils](#dataset-utils)
  + [0. Preface](#0-preface)
  + [1. Overview](#1-overview)
  + [2. About the `Dataset` object.](#2-about-the-dataset-object)


## 0. Preface
+ This package is mainly depand on [deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab).
+ Target: Construct `tf.data.Dataset` object from tfrecord files.
+ python file: `dataset_utils.py`

## 1. Overview
+ Step 1: Parse `tf.data.TFRecordDataset`, get a python dict.
+ Step 2: Preprocess input image and label.
  + Resize images and labels to range(min edge or max edge).
  + Random scale size(only in training).
  + Pad image and label to have dimensions >= [crop_height, crop_width]
  + Pad image with mean pixel value.
  + Randomly crop the image and label(only in training).
  + Randomly left-right flip the image and label(only in training).
+ Step 3: `tf.data.Dataset` ops, such as shuffle, repeat, batch, prefetch.

## 2. About the `Dataset` object.
+ Crop ops only occur in training procedure.
+ If `batch_size` > 1, then make sure all images/labels have the same image shape. 
  + We have two options to have the same image shape:
    1. original images/labels have the same shape.
    2. Crop ops(only working if `is_training` is true).
+ Output is a python dict, with keys in `common.py`, such as `'image', 'image_name', 'height', 'width', 'label'`.

