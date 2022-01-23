# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: yolov5-jishi
File Name: data_gen.py
Author: chenming
Create Date: 2021/11/8
Description：
-------------------------------------------------
"""
# -*- coding: utf-8 -*-
# @Time    : 20210610
# @Author  : dejahu
# @File    : gen_yolo_data.py
# @Software: PyCharm
# @Brief   : 生成测试、验证、训练的图片和标签

import os
import shutil
from pathlib import Path
from shutil import copyfile

import cv2
from PIL import Image, ImageDraw
from xml.dom.minidom import parse
import numpy as np
import os.path as osp
import random

# todo 修改为你的数据的根目录
FILE_ROOT = "/scm/data/xianyu/Mask/"
IMAGE_SET_ROOT = FILE_ROOT + "VOC2021_Mask/ImageSets/Main"
IMAGE_PATH = FILE_ROOT + "VOC2021_Mask/JPEGImages"
ANNOTATIONS_PATH = FILE_ROOT + "VOC2021_Mask/Annotations"
LABELS_ROOT = FILE_ROOT + "VOC2021_Mask/Labels"
DEST_PPP = FILE_ROOT + "mask_yolo_format"
DEST_IMAGES_PATH = "mask_yolo_format/images"
DEST_LABELS_PATH = "mask_yolo_format/labels"

if osp.isdir(LABELS_ROOT):
    shutil.rmtree(LABELS_ROOT)
    print("Labels")

if osp.isdir(DEST_PPP):
    shutil.rmtree(DEST_PPP)
    print("Dest")

label_names = ['key']


def cord_converter(size, box):

    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    dw = np.float32(1. / int(size[0]))
    dh = np.float32(1. / int(size[1]))

    w = x2 - x1
    h = y2 - y1
    x = x1 + (w / 2)
    y = y1 + (h / 2)

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def save_file(img_jpg_file_name, size, img_box):
    save_file_name = LABELS_ROOT + '/' + img_jpg_file_name + '.txt'
    file_path = open(save_file_name, "a+")
    for box in img_box:
        box_name = box[0]
        cls_num = 0
        if box_name in label_names:
            cls_num = label_names.index(box_name)
        new_box = cord_converter(size, box[1:])
        file_path.write(f"{cls_num} {new_box[0]} {new_box[1]} {new_box[2]} {new_box[3]}\n")
    file_path.flush()
    file_path.close()


def test_dataset_box_feature(file_name, point_array):

    im = Image.open(rf"{IMAGE_PATH}\{file_name}")
    imDraw = ImageDraw.Draw(im)
    for box in point_array:
        x1 = box[1]
        y1 = box[2]
        x2 = box[3]
        y2 = box[4]
        imDraw.rectangle((x1, y1, x2, y2), outline='red')
    im.show()


def get_xml_data(file_path, img_xml_file):
    img_path = file_path + '/' + img_xml_file + '.xml'
    # print(img_path)
    dom = parse(img_path)
    root = dom.documentElement
    img_name = root.getElementsByTagName("filename")[0].childNodes[0].data
    img_jpg_file_name = img_xml_file + '.jpg'
    # print(img_jpg_file_name)
    cv2.imread(img_jpg_file_name)
    img_size = root.getElementsByTagName("size")[0]
    if len(img_size) == 0:
        img_h, img_w, c = cv2.imread(img_jpg_file_name).shape
    else:
        img_w = img_size.getElementsByTagName("width")[0].childNodes[0].data
        img_h = img_size.getElementsByTagName("height")[0].childNodes[0].data
        img_c = img_size.getElementsByTagName("depth")[0].childNodes[0].data
    objects = root.getElementsByTagName("object")

    img_box = []
    for box in objects:
        cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
        x1 = int(float(box.getElementsByTagName("xmin")[0].childNodes[0].data))
        y1 = int(float(box.getElementsByTagName("ymin")[0].childNodes[0].data))
        x2 = int(float(box.getElementsByTagName("xmax")[0].childNodes[0].data))
        y2 = int(float(box.getElementsByTagName("ymax")[0].childNodes[0].data))

        img_box.append([cls_name, x1, y1, x2, y2])
    save_file(img_xml_file, [img_w, img_h], img_box)


def copy_data(img_set_source, img_labels_root, imgs_source, type):
    file_name = img_set_source + '/' + type + ".txt"
    file = open(file_name)

    root_file = Path(FILE_ROOT + DEST_IMAGES_PATH + '/' + type)
    if not root_file.exists():
        print(f"Path {root_file} is not exit")
        os.makedirs(root_file)

    root_file = Path(FILE_ROOT + DEST_LABELS_PATH + '/' + type)
    if not root_file.exists():
        print(f"Path {root_file} is not exit")
        os.makedirs(root_file)

    for line in file.readlines():
        img_name = line.strip('\n')
        img_sor_file = imgs_source + '/' + img_name + '.jpg'
        label_sor_file = img_labels_root + '/' + img_name + '.txt'

        DICT_DIR = FILE_ROOT + DEST_IMAGES_PATH + '/' + type
        img_dict_file = DICT_DIR + '/' + img_name + '.jpg'
        copyfile(img_sor_file, img_dict_file)

        DICT_DIR = FILE_ROOT + DEST_LABELS_PATH + '/' + type
        img_dict_file = DICT_DIR + '/' + img_name + '.txt'
        copyfile(label_sor_file, img_dict_file)


if __name__ == '__main__':
    img_set_root = IMAGE_SET_ROOT
    imgs_root = IMAGE_PATH
    img_labels_root = LABELS_ROOT
    if osp.isdir(img_labels_root) == False:
        os.makedirs(img_labels_root)
    os.makedirs(DEST_PPP)
    names = os.listdir(ANNOTATIONS_PATH)
    root = ANNOTATIONS_PATH
    files_all = os.listdir(root)
    files = []
    for file in files_all:
        if file.split(".")[-1] == "xml":
            files.append(file.split(".")[0])
    for file in files:
        file_xml = file.split(".")
        get_xml_data(root, file_xml[0])

    print("Done！")
