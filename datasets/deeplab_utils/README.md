# Deeplab Dataset Utils

[TOC]

---

## 0. Preface
+ This package is mainly copy from [deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab).
+ Target:
  + Construct tfreocrd files for all datasets, such COCO, VOC2012, etc.
  + Construct `tf.data.Dataset` object from tfrecord files.

---

## 1. tfrecords

### 1.1. VOC2012
+ script: `build_voc2012_data.py`
+ Recommended directory structure:
```
+ VOCdevkit
    + VOC2012
        + JPEGImages
        + SegmentationClass
        + SegmentationClassRaw(new directory)
        + ImageSets
            + Segmentation
        + segmentation_tfrecords(new directory)
```
+ command
    + Step 1: under `./TF-Semantic-Segmentation` path, `python datasets/deeplab_utils/remove_gt_colormap.py --original_gt_folder {/path/to/SegmentationClass} --output_dir {/path/to/SegmentationClassRaw}`
    + Step 2: under `./TF-Semantic-Segmentation` path, `python datasets/deeplab_utils/build_voc2012_data.py --image_format jpg`
```
python datasets/deeplab_utils/build_voc2012_data.py \
    --image_format jpg \
    --image_folder /path/to/JPEGImages \
    --semantic_segmentation_folder /path/to/SegmentationClassRaw \
    --list_folder /path/to/Segmentation \
    --output_dir /path/to/segmentation_tfrecords
```

### 1.2. VOC2012 aug
+ script: `build_voc2012_data.py`
+ Recommended directory structure:
```
+ VOCdevkit
    + VOC2012
    + JPEGImages
    + SegmentationClassAug(Download from https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)
    + ImageSets
        + SegmentationAug(Download from https://github.com/rishizek/tensorflow-deeplab-v3/tree/master/dataset)
                         (rename train.txt to train_aug.txt, move val.txt, ignore test.txt)
    + segmentation_aug_tfrecords(new directory)
```
+ command
    + Step 2: under `./TF-Semantic-Segmentation` path, run the following command
```shell
python datasets/deeplab_utils/build_voc2012_data.py \
    --image_format jpg \
    --image_folder /path/to/JPEGImages \
    --semantic_segmentation_folder /path/to/SegmentationClassAug \
    --list_folder /path/to/SegmentationAug \
    --output_dir /path/to/segmentation_aug_tfrecords
```



### 1.3. ADE20k
+ script: `build_ade20k_data.py`
+ Recommended directory structure:
```
+ ADE20K
    + ADEChallengeData2016
        + images
            + training
            + validation
        + annotations
            + training
            + validation
    + tfrecords(new directory)
```
+ command (under `./TF-Semantic-Segmentation` path): `python datasets/deeplab_utils/build_ade20k_data.py --image_format jpg`


### 1.4. CityScapes
+ script: `build_cityscapes_data.py`
+ Recommended directory structure:
```
+ Cityscapes
    + cityscapesscripts
        + annotation
        + evaluation
        + helpers
        + preparation
        + viewer
    + gtFine
        + train
        + val
        + test
    + leftImg8bit
        + train
        + val
        + test
    + tfrecords(new directory)
```
+ command:
    + Step 1: under `./Cityscapes` path, `python ./cityscapesscripts/preparation/createTrainIdLabelImgs.py`.
    + Step 2: under `./TF-Semantic-Segmentation` path, `python datasets/deeplab_utils/build_cityscapes_data.py`.


---

## 2. `tf.data.Dataset`

### 2.1. Overview
+ Step 1: Parse `tf.data.TFRecordDataset`, get a python dict.
+ Step 2: Preprocess input image and label.
  + Resize images and labels to range(min edge or max edge).
  + Random scale size(only in training).
  + Pad image and label to have dimensions >= [crop_height, crop_width]
  + Pad image with mean pixel value.
  + Randomly crop the image and label(only in training).
  + Randomly left-right flip the image and label.
+ Step 3: `tf.data.Dataset` ops, such as shuffle, repeat, batch, prefetch.

### 2.2. About the `Dataset` object.
+ Crop ops only occur in training procedure.
+ If `batch_size` > 1, then make sure all images/labels have the same image shape. 
  + We have two options to have the same image shape:
    1. original images/labels have the same shape.
    2. Crop ops(only working if `is_training` is true).
+ Output is a python dict, with keys `'image', 'image_name', 'height', 'width', 'label'`.

