# Semantic Segmentation Suite
+ original repo: 
  + (GeorgeSeif/Semantic-Segmentation-Suite)[https://github.com/GeorgeSeif/Semantic-Segmentation-Suite]
  + (tensorflow/models/research/deeplab)[https://github.com/tensorflow/models/tree/master/research/deeplab]

## 1. Targets
+ generate tfrecord files for several open source segmentation datasets.
+ generate dataset by `tf.data` utils.
+ generate model by `tf.keras` utils.
+ train/eval/test model by `tf.estimator` utils.


## 2. TODO

### 2.1. dataset:
+ [ ] camvid


### 2.2. predict
+ [ ] get results with original image size(such as [500, 375]).


### 2.3. trainining
+ [ ] get training results for multi-model and multi-dataset.


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
    --batch_size=8 \
    --num_gpus=1 \
    --gpu_devices="3" \
    --dataset_name="pascal_voc_seg" \
    --dataset_dir="/ssd/zhangyiyang/data/VOCdevkit/segmentation_tfrecords" \
    --crop_height=513 \
    --crop_width=513 \
    --min_scale_factor=0.5 \
    --max_scale_factor=2. \
    --scale_factor_step_size=0.25 \
    --model="DeepLabV3" \
    --saving_every_n_steps=200 \
    --logging_every_n_steps=20 \
    --summary_every_n_steps=20 \
    --num_val_images=1000 \
    --logs_name 2
```
