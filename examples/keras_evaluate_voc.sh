python ./scripts/keras_evaluate.py --num_gpus 1 \
                                   --gpu_devices 1 \
                                   --dataset_name pascal_voc_seg \
                                   --dataset_dir /hdd02/zhangyiyang/data/VOCdevkit/segmentation_aug_tfrecords \
                                   --split_name val \
                                   --eval_crop_height 513 \
                                   --eval_crop_width 513 \
                                   --model_type deeplab_v3_plus \
                                   --backend_type xception \
                                   --output_stride 16 \
                                   --model_weights_path None \
                                   --model_weights pascal_voc