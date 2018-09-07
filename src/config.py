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
# @File    : config.py

import argparse
import os


def set_train_args():
    """
    configure the training arguments

    :return: args
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_images_root_dir', type=str, default='../dataset/train/images/',
                        help='root directory of training images')

    parser.add_argument('--train_masks_root_dir', type=str, default='../dataset/train/masks/',
                        help='root directory of training ground truth (masks)')

    parser.add_argument('--test_images_root_dir', type=str, default='../dataset/test/images/',
                        help='root directory of test images')

    parser.add_argument('--train_batch_size', type=int, default=256,
                        help='train batch size')

    parser.add_argument('--val_batch_size', type=int, default=256,
                        help='validation batch size')

    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='test batch size')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='training batch size')

    parser.add_argument('--img_width', type=int, default=128,
                        help='width of image/mask')

    parser.add_argument('--img_height', type=int, default=128,
                        help='height of image/mask')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for RNN')

    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')

    parser.add_argument('--save_every_epoch', type=int, default=1,
                        help='save model variables for every how many epoch(s)')

    parser.add_argument('--train_summary', type=str, default=os.path.join(os.getcwd(), 'train/'),
                        help='train summary FileWriter')

    parser.add_argument('--val_summary', type=str, default=os.path.join(os.getcwd(), 'val/'),
                        help='val summary FileWriter')

    parser.add_argument('--dump_model_para_root_dir', type=str, default='../model_params/',
                        help='directory path to dump model parameters while training')

    args = parser.parse_args()
    return args


def set_deploy_args():
    """
    configure the training arguments

    :return: args
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_images_root_dir', type=str, default='../dataset/train/images/',
                        help='root directory of training images')

    parser.add_argument('--train_masks_root_dir', type=str, default='../dataset/train/masks/',
                        help='root directory of training ground truth (masks)')

    parser.add_argument('--test_images_root_dir', type=str, default='../dataset/test/images/',
                        help='root directory of test images')

    parser.add_argument('--submit_dump_full_path', type=str, default='../output/submission.csv',
                        help='submission csv file full path')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for RNN')

    parser.add_argument('--train_batch_size', type=int, default=64,
                        help='train batch size')

    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='validation batch size')

    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='test batch size')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='deploy batch size')

    parser.add_argument('--img_width', type=int, default=128,
                        help='width of image/mask')

    parser.add_argument('--img_height', type=int, default=128,
                        help='height of image/mask')

    parser.add_argument('--dump_model_para_root_dir', type=str, default='../model_params/',
                        help='directory path to dump model parameters while training')

    parser.add_argument('--optimal_model_path', type=str, default='epoch1_0.725810_0.758864.ckpt',
                        help='optimal model path to load from')

    args = parser.parse_args()
    return args
