# Dataset Utils

+ [Dataset Utils](#dataset-utils)
  + [<ol start="0">
<li>Preface</li>
</ol>](#ol-start0liprefaceliol)
  + [<ol>
<li>VOC2012</li>
</ol>](#ollivoc2012liol)
  + [<ol start="2">
<li>VOC2012 aug</li>
</ol>](#ol-start2livoc2012-augliol)
  + [<ol start="3">
<li>ADE20k</li>
</ol>](#ol-start3liade20kliol)
  + [<ol start="4">
<li>CityScapes</li>
</ol>](#ol-start4licityscapesliol)
  + [<ol start="5">
<li>CamVid</li>
</ol>](#ol-start5licamvidliol)


## 0. Preface
+ This package is mainly depand on [deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab).
+ Target: Construct tfreocrd files for datasets, such Cityscapes, VOC2012, etc.

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
    + Step 1: under `./TF-Semantic-Segmentation` path, 
```
python scripts/datasets/remove_gt_colormap.py \
    --original_gt_folder {/path/to/SegmentationClass} \
    --output_dir {/path/to/SegmentationClassRaw}
```
  + Step 2: under `./TF-Semantic-Segmentation` path
```
python scripts/datasets/build_voc2012_data.py \
    --image_format jpg \
    --image_folder /path/to/JPEGImages \
    --semantic_segmentation_folder /path/to/SegmentationClassRaw \
    --list_folder /path/to/Segmentation \
    --output_dir /path/to/segmentation_tfrecords
```

## 2. VOC2012 aug
+ script: `build_voc2012_data.py`
+ Recommended directory structure:
  + SegmentationClassAug: Download from [here](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)
  + SegmentationAug: Download from [here](https://github.com/rishizek/tensorflow-deeplab-v3/tree/master/dataset), rename `train.txt` to `train_aug.txt`, move `train_aug.txt` and `val.txt` to SegmentationAug, ignore `test.txt`.
```
+ VOCdevkit
    + VOC2012
    + JPEGImages
    + SegmentationClassAug(new directory)
    + ImageSets
        + SegmentationAug(new directory)
    + segmentation_aug_tfrecords(new directory)
```
+ command
    + under `./TF-Semantic-Segmentation` path, run the following command
```shell
python scripts/datasets/build_voc2012_data.py \
    --image_format jpg \
    --image_folder /path/to/JPEGImages \
    --semantic_segmentation_folder /path/to/SegmentationClassAug \
    --list_folder /path/to/SegmentationAug \
    --output_dir /path/to/segmentation_aug_tfrecords
```


## 3. ADE20k
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
+ command (under `./TF-Semantic-Segmentation` path): 

```
python scripts/datasets/build_ade20k_data.py \
    --image_format jpg \
    --train_image_folder /path/to/train_image \
    --train_image_label_folder /path/to/train_label \
    --val_image_folder /path/to/val_image \
    --val_image_label_folder /path/to/val_label \
    --output_dir /path/to/tfrecoreds
```


## 4. CityScapes
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
  + Step 1: under `./Cityscapes` path, `python ./cityscapesscripts/preparation/createTrainIdLabelImgs.py`
  + Step 2: under `./TF-Semantic-Segmentation` path, 

```
python scripts/datasets/build_cityscapes_data.py \
    --cityscapes_root /path/to/cityscapes_root \
    --output_dir /path/to/tfrecords
```

## 5. CamVid
+ script: `build_camvid_data.py`
+ 