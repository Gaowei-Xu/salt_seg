#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
#
#  By downloading, copying, installing or using the software you agree to this license.
#  If you do not agree to this license, do not download, install,
#  copy or use the software.
#
#
#       Shanghai ShanMing Information & Technology Ltd. License Agreement
#                For quant trade strategy and library
#
# Copyright (C) 2017, Shanghai ShanMing Information & Technology Ltd., all rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are NOT permitted.
#
# @Time    : 2018/9/2 下午10:42
# @Author  : Gaowei Xu
# @Email   : gaowxu@hotmail.com
# @File    : model.py

import numpy as np
import os
import cv2


class RLEncoder(object):
    """
    class of run-length encoder
    """
    def __init__(self):
        pass

    def encode(self, image, order='F', format=True):
        """
        encoding the image to run-length format
        :param image: image is binary mask image, shape is np.array([rows, columns]),
                      order is down-then-right, i.e. Fortran format determines if the
                      order needs to be preformatted (according to submission rules)
                      or not
        :param order:
        :param format:
        :return: run length as an array or string (if format is True)
        """
        bytes = image.reshape(image.shape[0] * image.shape[1], order=order)
        runs = []       # list of run lengths
        r = 0           # the current run length
        pos = 1         # count starts from 1 per WK
        for c in bytes:
            if c == 0:
                if r != 0:
                    runs.append((pos, r))
                    pos += r
                    r = 0
                pos += 1
            else:
                r += 1

        # if last run is unsaved (i.e. data ends with 1)
        if r != 0:
            runs.append((pos, r))
            pos += r
            r = 0

        if format:
            z = ''

            for rr in runs:
                z += '{} {} '.format(rr[0], rr[1])
            return z[:-1]
        else:
            return runs


if __name__ == '__main__':
    """
    test for run-length encoding algorithm
    :return:
    """
    train_imgs_root_dir = '../dataset/train/images/'
    train_masks_root_dir = '../dataset/train/masks/'

    train_csv_file_full_path = '../dataset/train.csv'

    rf = open(train_csv_file_full_path, 'rb')
    lines = rf.readlines() - 1  # skip first row
    amount = len(lines)
    codec = RLEncoder()

    for i, line in enumerate(lines):
        if 'id,rle_mask' in line:
            continue
        img_name = line.split(',')[0] + '.png'
        rle_gt = line.split(',')[1].strip()
        print 'Processing image {} ({}/{})...'.format(img_name, i, amount)

        image_full_path = os.path.join(train_imgs_root_dir, img_name)
        mask_full_path = os.path.join(train_masks_root_dir, img_name)
        image = cv2.imread(image_full_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)

        rle_output = codec.encode(mask)
        assert rle_output == rle_gt
        if rle_output != rle_gt:
            print 'Error: run-length codec is wrong!'
            break



