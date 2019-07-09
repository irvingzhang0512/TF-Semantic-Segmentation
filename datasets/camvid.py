# coding=utf-8
import cv2
import os
import numpy as np
import tensorflow as tf

from utils import helpers, utils
from datasets.data_argument import cv2_data_argument, cv2_tf_preprocessing


def get_camvid_default_config():
    return {
        'root_path': './CamVid',
        'class_dict_file_name': 'class_dict.csv',
        'batch_size': 1,
        'crop_height': 512, 'crop_width': 512,
        'h_flip': False, 'v_flip': False,
        'brightness': None,
        'rotation': None,
        'preprocessing_type': 'tf'
    }


def get_camvid_dataset(mode='train',
                       root_path='./CamVid',
                       class_dict_file_name='class_dict.csv',
                       batch_size=1,
                       crop_height=512, crop_width=512,
                       h_flip=False, v_flip=False,
                       brightness=None,
                       rotation=None,
                       preprocessing_type='tf'):
    """
    对于训练集、验证集、测试集都有的操作：
    1. 读取 class_dict.csv 中的内容，获取 class_names_list（所有类的名称） 和 label_values（每一类的 rgb 分量）。
    2. 获取训练、验证、预测集的原始文件名称列表和GT文件名称列表。
    3. 读取图片、对原始图片进行 preprocessing_type 操作、对于 GT 图片进行 one-hot 操作、batch size操作。

    对于训练集特有的操作，数据增广，包括：
    1. 随机切片（crop_height, crop_width），如果原始图片尺寸不到这个范围，则报错。
    2. 随机水平、垂直镜像。
    3. 随机亮度变换。
    4. 随机移动。

    :param root_path:
    :param mode:
    :param class_dict_file_name:
    :param batch_size:
    :param crop_height:
    :param crop_width:
    :param h_flip:
    :param v_flip:
    :param brightness:
    :param rotation:
    :param preprocessing_type:
    :return:
    """
    if mode not in ['train', 'val', 'test']:
        raise ValueError('unknown mode {}'.format(mode))

    # 1. 读取 class_dict.csv 中的内容，获取 class_names_list（所有类的名称） 和 label_values（每一类的 rgb 分量）。
    # Get the names of the classes so we can record the evaluation results
    class_names_list, label_values = helpers.get_label_info(os.path.join(root_path, class_dict_file_name))
    class_names_string = ""
    for class_name in class_names_list:
        if not class_name == class_names_list[-1]:
            class_names_string = class_names_string + class_name + ", "
        else:
            class_names_string = class_names_string + class_name
    num_classes = len(label_values)

    # 2. 获取训练、验证、预测集的原始文件名称列表和GT文件名称列表。
    train_in_names, train_out_names, val_in_names, val_out_names, test_in_names, test_out_names = utils.prepare_data(
        root_path)

    if mode == 'train':
        # 3. 读取图片、进行数据增广、进行batch size操作。

        def _map_cv2_train(raw_path, gt_path):
            # 训练集操作：读取图片、图像增广、tf归一化、gt图片转换为 one-hot

            # 读取图片
            raw_image = cv2.cvtColor(cv2.imread(raw_path.decode(), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            gt_image = cv2.cvtColor(cv2.imread(gt_path.decode(), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            # 数据增广
            raw_image, gt_image = cv2_data_argument(raw_image, gt_image,
                                                    crop_height=crop_height, crop_width=crop_width,
                                                    h_flip=h_flip, v_flip=v_flip,
                                                    brightness=brightness,
                                                    rotation=rotation)

            # 原始图片进行初始化，gt图片转换为 one-hot 形式
            if preprocessing_type == 'tf':
                raw_image = cv2_tf_preprocessing(raw_image)
            gt_image = helpers.one_hot_it(label=gt_image, label_values=label_values)

            return raw_image, gt_image.astype(np.int32), crop_height, crop_width

        def _map_train_reshape(img1, var, h, w):
            # 因为通过 tf.py_func 产生的Iterator，其目标shape不明确，导致在构建计算图时失败
            # 构建计算图中，要求明确图片的depth，所以需要reshape为3
            return tf.reshape(img1, [h, w, 3]), var

        cur_dataset = tf.data.Dataset.from_tensor_slices((train_in_names, train_out_names))
        cur_dataset = cur_dataset.map(lambda raw_path, gt_path: tuple(tf.py_func(
            _map_cv2_train,
            [raw_path, gt_path],
            # [tf.float32, tf.int32, tf.int32, tf.int32]  # windows
            [tf.float32, tf.int32, tf.int64, tf.int64]  # linux
        ))).map(_map_train_reshape).batch(batch_size).prefetch(2)
    elif mode == 'val':
        # 3. 读取图片、tf归一化、gt图片转换为输出类别（而不是one-hot）、进行batch size操作

        def _map_cv2_val(raw_path, gt_path):
            # 读取图片
            raw_image = cv2.cvtColor(cv2.imread(raw_path.decode(), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            gt_image = cv2.cvtColor(cv2.imread(gt_path.decode(), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            # 数据增广
            raw_image, gt_image = cv2_data_argument(raw_image, gt_image,
                                                    crop_height=crop_height, crop_width=crop_width)

            # tf 归一化
            if preprocessing_type == 'tf':
                raw_image = cv2_tf_preprocessing(raw_image)

            # gt图片转换为输出类别
            gt_image = helpers.one_hot_it(label=gt_image, label_values=label_values)
            gt_image = helpers.reverse_one_hot(gt_image)

            return raw_image.astype(np.float32), gt_image.astype(np.int32), crop_height, crop_width

        def _map_val_reshape(img1, img2, h, w):
            # 因为通过 tf.py_func 产生的Iterator，其目标shape不明确，导致在构建计算图时失败
            # 构建计算图中，要求明确图片的depth，所以需要reshape为3
            return tf.reshape(img1, [h, w, 3]), tf.reshape(img2, [h, w])

        cur_dataset = tf.data.Dataset.from_tensor_slices((val_in_names, val_out_names))
        cur_dataset = cur_dataset.map(lambda raw_path, gt_path: tuple(tf.py_func(
            _map_cv2_val,
            [raw_path, gt_path],
            # [tf.float32, tf.int32, tf.int32, tf.int32]  # windows
            [tf.float32, tf.int32, tf.int64, tf.int64]  # linux
        ))).map(_map_val_reshape).batch(batch_size).prefetch(2)
        return cur_dataset, num_classes, label_values
    else:
        # 3. 读取图片、进行batch size操作

        def _map_cv2_predict(raw_path):
            # 测试集操作：读取图片、tf归一化、获取 target file name

            # TODO: get target file name by raw path
            file_name = raw_path

            # 读取图片
            raw_image = cv2.cvtColor(cv2.imread(raw_path.decode(), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            h, w, _ = raw_image.shape

            # TODO: when input height/width mod 32 != 0...

            # 原始图片进行初始化，gt图片转换为 one-hot 形式
            if preprocessing_type == 'tf':
                raw_image = cv2_tf_preprocessing(raw_image)

            return raw_image.astype(np.float32), file_name, h, w

        def _map_predict_reshape(img1, var, h, w):
            # 因为通过 tf.py_func 产生的Iterator，其目标shape不明确，导致在构建计算图时失败
            # 构建计算图中，要求明确图片的depth，所以需要reshape为3
            return tf.reshape(img1, [h, w, 3]), var

        cur_dataset = tf.data.Dataset.from_tensor_slices(test_in_names)
        cur_dataset = cur_dataset.map(lambda raw_path, target_file_name: tuple(tf.py_func(
            _map_cv2_predict,
            [raw_path, target_file_name],
            # [tf.float32, tf.int32, tf.int32, tf.int32]  # windows
            [tf.float32, tf.int32, tf.int64, tf.int64]  # linux
        ))).map(_map_predict_reshape).batch(batch_size).prefetch(2)

    return cur_dataset, num_classes, label_values
