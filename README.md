# TF Semantic Segmentation

+ [TF Semantic Segmentation](#tf-semantic-segmentation)
  + [0. Reference](#0-reference)
  + [1. Targets](#1-targets)
  + [2. TODO](#2-todo)
    + [2.1. Datasets](#21-datasets)
    + [2.2. Models](#22-models)
    + [2.2. example](#22-example)
  + [3. Quick Start](#3-quick-start)

## 0. Reference
+ [GeorgeSeif/Semantic-Segmentation-Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite)
  + Project Structure.
+ [tensorflow/models/research/deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)
  + TFRecord utils & Image preprocessing & Data Argument

## 1. Targets
+ generate tfrecord files for several open source semantic segmentation datasets.
+ generate dataset by `tf.data` utils.
+ generate model by `tf.keras` utils.
+ train/eval/test model by `tf.keras.Model().fit/predict/evaluate` utils.


## 2. TODO

### 2.1. Datasets
+ [x] Pascal VOC Augmentation
+ [ ] Cityscaptes
+ [ ] ADE20k
+ [ ] CamVid
+ [ ] More Datasets

### 2.2. Models
+ [x] DeepLabV3+
+ [ ] fine_tune_batch_norm
+ [ ] l2 loss


### 2.2. example
+ [x] train script.
+ [x] show dataset jupyter.
+ [x] evaluation script.
+ [ ] predict script.
+ [ ] train jupyter.
+ [ ] val jupyter.


## 3. Quick Start
+ Step 1: generate tfrecord files for datasets, following <a href='datasets/README.md'>this doc</a>.
+ Step 2: run scripts in `scripts`.
  + In `examples` dir, there are some training/evaluating scripts.