# Deeplab Dataset Utils

[TOC]

---

## 0. Preface
+ This package is mainly copy from [deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab).
+ The target of this package is to construct tfreocrd files for all datasets, such COCO, VOC2012, etc.

---

## 1. VOC2012
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


---


## 2. ADE20k
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

---

## 3. CityScapes
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