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
# @File    : infer.py
import os
import tensorflow as tf
from rle import RLEncoder
from config import set_deploy_args
from dataloader import DataLoader
from model import UNetModel
import numpy as np
import cv2


def infer(args):
    """
    Inference phase main process

    :param args:
    :return:
    """
    dataloader = DataLoader(
        train_images_root_dir=args.train_images_root_dir,
        train_masks_root_dir=args.train_masks_root_dir,
        test_images_root_dir=args.test_images_root_dir,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
        img_width=args.img_width,
        img_height=args.img_height
    )

    print 'Dataset loading successfully...'

    dump_root_dir = '../dump_comparison/'
    if not os.path.exists(dump_root_dir):
        os.makedirs(dump_root_dir)

    codec = RLEncoder()
    tf.reset_default_graph()
    model = UNetModel(args)

    # configure GPU training, soft allocation.
    gpuConfig = tf.ConfigProto(allow_soft_placement=True)
    gpuConfig.gpu_options.allow_growth = True

    wf = open(args.submit_dump_full_path, 'wb')
    wf.write('id,rle_mask\n')
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(args.dump_model_para_root_dir, args.optimal_model_path))

        dataloader.reset()

        for batch in range(dataloader.test_batch_amount):
            # input_batch shape = [batch_size, height, width]
            # gt_batch shape = [batch_size, height, width]
            input_batch, gt_batch, img_name = dataloader.next_batch(mode='test')

            infer_labels, gt_labels, loss = sess.run(
                fetches=[
                    model.infer_labels,
                    model.gt_labels,
                    model.loss,
                ],
                feed_dict={
                    model.input_data: input_batch,
                    model.ground_truth: gt_batch,
                })

            # process the result of infer_labels
            infer_labels = np.reshape(infer_labels[0], newshape=[args.img_height, args.img_width])
            infer_labels = np.uint8(np.round(infer_labels) * 255)

            print 'gt = {}'.format(gt_batch[0, :, :, 0])
            print 'infer_labels = {}\n'.format(infer_labels)

            cv2.imwrite(os.path.join(dump_root_dir, '{}_gt_mask.jpg'.format(str(batch))), gt_batch[0, :, :, 0])
            cv2.imwrite(os.path.join(dump_root_dir, '{}_infer_mask.jpg'.format(str(batch))), infer_labels)

            rle_bitstream = codec.encode(infer_labels)
            wf.write('{},{}\n'.format(img_name, rle_bitstream))
            print 'Processing batch {} (totally {} batches)...'.format(batch, dataloader.test_batch_amount)
    wf.close()

if __name__ == '__main__':
    args = set_deploy_args()
    infer(args)


