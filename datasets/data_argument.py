# coding=utf-8
import cv2
import random
import numpy as np

from utils import utils


def cv2_tf_preprocessing(image):
    return image / 255.0


def cv2_data_argument(input_image, output_image,
                      crop_height=512, crop_width=512,
                      h_flip=False, v_flip=False,
                      brightness=None,
                      rotation=None):
    """
    对原始图片以及分割结果进行图像增强

    1. 随机切片
    2. 随机水平/垂直镜像
    3. 随机亮度变换
    4. 随机旋转

    :param input_image:
    :param output_image:
    :param crop_height:
    :param crop_width:
    :param h_flip:
    :param v_flip:
    :param brightness:
    :param rotation:
    :return:
    """
    # 随机切片
    if crop_height is not None and crop_width is not None:
        input_image, output_image = utils.random_crop(input_image, output_image, crop_height, crop_width)

    # 随机水平/垂直变换
    if h_flip and random.randint(0, 1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if v_flip and random.randint(0, 1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)

    # 随机亮度变换
    if brightness:
        factor = 1.0 + random.uniform(-1.0 * brightness, brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)

    # 随机旋转
    if rotation:
        angle = random.uniform(-1 * rotation, rotation)
        m = cv2.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, m, (input_image.shape[1], input_image.shape[0]),
                                     flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, m, (output_image.shape[1], output_image.shape[0]),
                                      flags=cv2.INTER_NEAREST)

    return np.array(input_image).astype(np.float32), np.array(output_image).astype(np.int32)
