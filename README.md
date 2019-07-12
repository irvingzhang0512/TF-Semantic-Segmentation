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
+ [x] use tensorflow to implement all data argument tools instead of `cv2`.
+ [x] generate tfrecord utils.
+ [x] utils to get `tf.data.Dataset` by tfrecord files.
+ [x] ade20k
+ [x] cityscape
+ [x] voc
+ [ ] camvid


## 3. Project Architecture
+ `builders`
+ `datasets`
+ `models`
+ `estimator_models`
+ `frontends`
+ `scripts`
+ `utils`


## 4. Scripts
```shell
python scripts/train_estimator.py \
    --num_epochs=25 \
    --epoch_start_i=0 \
    --batch_size=1 \
    --num_gpus=1 \
    --gpu_devices="3" \
    --learning_rate_start=0.001 \
    --optimizer_decay=0.995 \
    --dataset_name="pascal_voc_seg" \
    --dataset_dir="/ssd/zhangyiyang/data/VOCdevkit/segmentation_tfrecords" \
    --crop_height=512 \
    --crop_width=512 \
    --min_resize_value=512 \
    --min_scale_factor=1. \
    --max_scale_factor=1. \
    --model="Encoder-Decoder" \
    --saving_every_n_steps=100 \
    --logging_every_n_steps=5 \
    --summary_every_n_steps=10
```
