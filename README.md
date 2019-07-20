# TF Semantic Segmentation

+ Reference
  + [GeorgeSeif/Semantic-Segmentation-Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite)
    + Project Structure.
  + [tensorflow/models/research/deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)
    + TFRecord utils & Image preprocessing & Data Argument
  + [rishizek/tensorflow-deeplab-v3](https://github.com/rishizek/tensorflow-deeplab-v3)
    + DeepLabV3 model & Training procedure.

## 1. Targets
+ generate tfrecord files for several open source segmentation datasets.
+ generate dataset by `tf.data` utils.
+ generate model by `tf.keras` utils.
+ train/eval/test model by `tf.estimator` utils.


## 2. TODO

### 2.1. Datasets
+ [ ] CamVid.
+ [ ] More Datasets.

### 2.2. Models
+ [x] DeepLabV3
+ [ ] DeepLabV3+
+ [ ] get `tf.keras` Model.


### 2.2. example
+ [x] train script.
+ [x] show dataset jupyter.
+ [ ] evaluation script.
+ [ ] predict script.
+ [ ] train jupyter.
+ [ ] val jupyter.


## 3. Quick Start
+ Step 1: generate tfrecord files for datasets, following <a href='datasets/README.md'>this doc</a>.
+ Step 2: run scripts in `scripts`.
  + <a href='g3doc/deeplabv3.md'>g3doc/deeplabv3.md</a> provide DeepLabV3 training commands and results