# Semantic Segmentation Suite
+ original repo: (GeorgeSeif/Semantic-Segmentation-Suite)[https://github.com/GeorgeSeif/Semantic-Segmentation-Suite]
+ (tensorflow/models/research/deeplab)[https://github.com/tensorflow/models/tree/master/research/deeplab]

## 1. Targets
+ generate tfrecord files for several open source segmentation datasets.
+ generate dataset by `tf.data` utils.
+ generate model by `tf.keras` utils.
+ train/eval/test model by `tf.estimator` utils.


## 2. TODO

### 2.1. dataset:
+ [ ] use tensorflow to implement all data argument tools instead of `cv2`.
+ [ ] generate tfrecord utils.
+ [ ] utils to get `tf.data.Dataset` by tfrecord files.
+ [ ] ade20k
+ [ ] cityscape
+ [ ] voc
+ [ ] camvid


## 3. Project Architecture
+ `builders`
+ `datasets`
+ `models`
+ `estimator_models`
+ `frontends`
+ `scripts`
+ `utils`