# TF Semantic Segmentation

+ [TF Semantic Segmentation](#tf-semantic-segmentation)
  + [0. Reference](#0-reference)
  + [1. Targets](#1-targets)
  + [2. TODO](#2-todo)
    + [2.1. Datasets](#21-datasets)
    + [2.2. Models](#22-models)
    + [2.3. example](#23-example)
    + [2.4. train/eval/predict](#24-trainevalpredict)
  + [3. Quick Start](#3-quick-start)
  + [4. Results](#4-results)
    + [4.1. VOC](#41-voc)
    + [4.2. ADE20k](#42-ade20k)
    + [4.3. Cityscapes](#43-cityscapes)

## 0. Reference
+ [GeorgeSeif/Semantic-Segmentation-Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite)
  + Project Structure.
+ [tensorflow/models/research/deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)
  + TFRecord utils & Image preprocessing & Data Argument
  + Training params.
+ [bonlime/keras-deeplab-v3-plus](https://github.com/bonlime/keras-deeplab-v3-plus)
  + DeepLab V3+ Model architecture.

## 1. Targets
+ generate tfrecord files for several open source semantic segmentation datasets.
+ generate dataset by `tf.data` utils.
+ generate model by `tf.keras` utils.
+ train/eval/test model by `tf.keras.Model().fit/predict/evaluate` utils.


## 2. TODO

### 2.1. Datasets
+ [x] Pascal VOC Augmentation
+ [x] Cityscaptes
+ [x] ADE20k
+ [ ] CamVid
+ More Datasets

### 2.2. Models
+ [x] DeepLabV3+
+ [x] fine_tune_batch_norm
+ [x] l2 loss
+ [x] BUG: model.call() and model.predict() generate different feature maps.
  + use upsample layer to replace lambda image resize layer.


### 2.3. example
+ scripts
  + [x] train script.
  + [x] evaluation script.
  + [ ] predict script.
+ jupyter
  + [x] show dataset.
  + [x] overfit on one sample jupyter.

### 2.4. train/eval/predict
+ [ ] multi scale prediction.
+ [x] Continue training.


## 3. Quick Start
+ Step 1: generate tfrecord files for datasets, following <a href='segmentation/datasets/README.md'>this doc</a>.
+ Step 2: run scripts in `scripts`.
  + In `examples` dir, there are some training/evaluating scripts.

## 4. Results

### 4.1. VOC

| model       | backend  | params | mIOU        | comment          |
| ----------- | -------- | ------ | ----------- | ---------------- |
| deeplab v3+ | xception | OS 16  | 82.20%(val) | original tf repo |

### 4.2. ADE20k
| model       | backend  | params                                             | mIOU        | comment          |
| ----------- | -------- | -------------------------------------------------- | ----------- | ---------------- |
| deeplab v3+ | xception | OS 8, [0.5:0.25:1.75] eval scales, 82.52% accuracy | 45.65%(val) | original tf repo |

### 4.3. Cityscapes
| model          | backend    | params                             | mIOU         | comment          |
| -------------- | ---------- | ---------------------------------- | ------------ | ---------------- |
| deeplab v3+    | xception65 | OS 16                              | 78.79%(val)  | original tf repo |
| deeplab v3+    | xception65 | OS 8, [0.75:0.25:1.75] eval scales | 80.42%(val)  | original tf repo |
| deeplab v3+    | xception   | OS 16, voc, lr 1e-3, bz 15, 768    | ?(gpu 0,1,2) |                  |
| deeplab v3+ v2 | resnet50   | imagenet, lr 1e-4, bz 16, 512      | ?(gpu 3)     |                  |
