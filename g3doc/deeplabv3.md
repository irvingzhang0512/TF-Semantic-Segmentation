# Training DeepLabV3

## 1. Results

### 1.1. voc aug
|model|OS|results|
|:----:|:-:|:----:|
|paper|16|77.21|
|repo: tensorflow-deeplab-v3|16|76.42|
|this repo|16|75.83|

### 1.2. cityscapes
|model|OS|results|
|:----:|:-:|:----:|
|paper|16|77.23|
|this repo|16|73.11|

## 2. Scripts
+ voc aug
```shell
python scripts/train_estimator.py \
    --train_split_name="train_aug" \
    --val_split_name="val" \
    --num_epochs=46 \
    --epoch_start_i=0 \
    --batch_size=5 \
    --num_gpus=1 \
    --dataset_name="pascal_voc_seg" \
    --dataset_dir="/ssd/zhangyiyang/data/VOCdevkit/segmentation_aug_tfrecords" \
    --train_crop_height=513 \
    --train_crop_width=513 \
    --eval_crop_height=513 \
    --eval_crop_width=513 \
    --min_scale_factor=0.5 \
    --max_scale_factor=2. \
    --scale_factor_step_size=0.25 \
    --model="DeepLabV3" \
    --saving_every_n_steps=2000 \
    --logging_every_n_steps=100 \
    --summary_every_n_steps=100 \
    --num_val_images=1449 \
    --base_learning_rate=7e-3 \
    --end_learning_rate=1e-5 \
    --training_number_of_steps=60000 \
    --weight_decay=1e-4 \
    --gpu_devices="2" \
    --output_stride=8 \
    --logs_name 4
```
+ cityscapes
```shell
python scripts/train_estimator.py \
    --num_epochs=140 \
    --epoch_start_i=0 \
    --batch_size=4 \
    --num_gpus=1 \
    --dataset_name="cityscapes" \
    --dataset_dir="/ssd/zhangyiyang/data/Cityscapes/tfrecords" \
    --train_crop_height=769 \
    --train_crop_width=769 \
    --eval_crop_height=1025 \
    --eval_crop_width=2049 \
    --training_number_of_steps=90000 \
    --min_scale_factor=0.5 \
    --max_scale_factor=2. \
    --scale_factor_step_size=0.25 \
    --model="DeepLabV3" \
    --saving_every_n_steps=700 \
    --logging_every_n_steps=50 \
    --summary_every_n_steps=50 \
    --num_val_images=500 \
    --validation_step=2 \
    --base_learning_rate=7e-3 \
    --end_learning_rate=1e-6 \
    --gpu_devices="2" \
    --logs_name 2
```
+ ade20k(not tested)
```shell
python scripts/train_estimator.py \
    --num_epochs=60 \
    --epoch_start_i=0 \
    --batch_size=8 \
    --num_gpus=1 \
    --dataset_name="ade20k" \
    --dataset_dir="/ssd/zhangyiyang/data/ADE20K/tfrecord" \
    --train_crop_height=513 \
    --train_crop_width=513 \
    --eval_crop_height=2100 \
    --eval_crop_width=2100 \
    --training_number_of_steps=150000 \
    --min_scale_factor=0.5 \
    --max_scale_factor=2. \
    --scale_factor_step_size=0.25 \
    --min_resize_value=600 \
    --max_resize_value=800 \
    --model="DeepLabV3" \
    --saving_every_n_steps=2000 \
    --logging_every_n_steps=100 \
    --summary_every_n_steps=100 \
    --num_val_images=2000 \
    --gpu_devices="2" \
    --logs_name 1
```