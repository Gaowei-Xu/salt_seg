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

import os
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib as plt

from config import set_train_args
from dataloader import DataLoader
from model import UNetModel


def train(args):
    """
    Train phase main process

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
        img_height=args.img_height,
        dump_norm_full_path=args.dump_norm_full_path
    )

    print 'Dataset loading successfully...'

    model = UNetModel(args)
    'Model initialized successfully...'

    # configure GPU training, soft allocation.
    gpuConfig = tf.ConfigProto(allow_soft_placement=True)
    gpuConfig.gpu_options.allow_growth = True

    # create two list to store cost values
    train_loss = np.zeros(args.num_epochs)
    val_loss = np.zeros(args.num_epochs)

    # create folders
    if not os.path.exists(args.train_summary):
        os.makedirs(args.train_summary)
    if not os.path.exists(args.val_summary):
        os.makedirs(args.val_summary)

    with tf.Session(config=gpuConfig) as sess:
        train_writer = tf.summary.FileWriter(args.train_summary, sess.graph)
        val_writer = tf.summary.FileWriter(args.val_summary, sess.graph)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(max_to_keep=None)

        print 'Start to train model:'
        train_step = 0
        val_step = 0
        for e in range(args.num_epochs):
            dataloader.reset()

            for batch in range(dataloader.train_batch_amount):
                # input_batch shape = [batch_size, height, width]
                # gt_batch shape = [batch_size, height, width]
                input_batch, gt_batch, _ = dataloader.next_batch(mode='train')

                infer_labels, gt_labels, loss, summary_op, optimizer, dice_coeff, probs = sess.run(
                    fetches=[
                        model.infer_labels,
                        model.gt_labels,
                        model.loss,
                        model.summary_op,
                        model.optimizer,
                        model.dice_coeff,
                        model.probs
                    ],
                    feed_dict={
                        model.input_data: input_batch,
                        model.ground_truth: gt_batch,
                    })
                print 'Epoch {} batch {}: loss = {}, dice_coeff = {}:\nsum(gt_labels) = {}, sum(infer_labels) = ' \
                      '{}\ngt_labels = {}, predicted_probs = {}...\n'.format(e, batch, loss, dice_coeff,
                                                           np.sum(gt_labels[0]), np.sum(infer_labels[0]),
                                                           gt_labels[0], probs[0])

                # add summary and accumulate stats
                train_writer.add_summary(summary_op, train_step)
                train_loss[e] += loss
                train_step += 1

            train_loss[e] /= dataloader.train_batch_amount

            for batch in range(dataloader.val_batch_amount):
                # input_batch shape = [batch_size, height, width]
                # gt_batch shape = [batch_size, height, width]
                input_batch, gt_batch, _ = dataloader.next_batch(mode='val')
                infer_labels, gt_labels, loss, summary_op = sess.run(
                    fetches=[
                        model.infer_labels,
                        model.gt_labels,
                        model.loss,
                        model.summary_op,
                    ],
                    feed_dict={
                        model.input_data: input_batch,
                        model.ground_truth: gt_batch,
                    })
                # add summary and accumulate stats
                val_writer.add_summary(summary_op, val_step)
                val_loss[e] += loss
                val_step += 1

            val_loss[e] /= dataloader.val_batch_amount

            # checkpoint model variable
            if (e + 1) % args.save_every_epoch == 0:
                model_name = 'epoch{}_{:2f}_{:2f}.ckpt'.format(e + 1, train_loss[e], val_loss[e])
                dump_model_full_path = os.path.join(args.dump_model_para_root_dir, model_name)
                saver.save(sess=sess, save_path=dump_model_full_path)

            print('Epoch {0:02d}: err(train)={1:.2f}, err(valid)={2:.2f}'.format(e + 1, train_loss[e], val_loss[e]))

        # close writer and session objects
        train_writer.close()
        val_writer.close()
        sess.close()


if __name__ == '__main__':
    args = set_train_args()
    train(args)
