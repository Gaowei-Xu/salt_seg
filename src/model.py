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

import tensorflow as tf
import numpy as np


class UNetModel(object):
    """
    UNet model
    Reference:  https://arxiv.org/pdf/1505.04597.pdf
                https://www.kaggle.com/shenmbsw/data-augmentation-and-tensorflow-u-net
                https://www.kaggle.com/divrikwicky/u-net-with-simple-resnet-blocks-forked
    """
    def __init__(self, args):
        """
        configure model parameters and build model
        :param args: model configuration parameters
        """
        self._image_width = args.img_width
        self._image_height = args.img_height
        self._learning_rate = args.learning_rate
        self._batch_size = args.batch_size
        self._num_cls = 2   # foreground and background
        self._channels = 1

        self._input_data = tf.placeholder(
            tf.float32, shape=[self._batch_size, self._image_height, self._image_width, self._channels], name="image")

        self._ground_truth = tf.placeholder(
            tf.float32, shape=[self._batch_size, self._image_height, self._image_width, self._num_cls], name="mask")

        self._global_step = tf.Variable(0, trainable=False)
        self._mask_labels = None
        self._infer_labels = None
        self._loss = None
        self._optimizer = None
        self._summary_op = None
        self._iou_score = None

        self.build_model()

    def build_model(self):
        """
        build Unet model
        :return:
        """
        def conv2d(name, inputs, filter_shape, strides, padding='SAME', activation=tf.nn.relu):
            """
            conv2d wrapper

            :param name:
            :param inputs: NHWC
            :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
            :param strides: specifying the strides of the convolution along the height and width
            :param padding:
            :param activation:
            :return:
            """
            conv = tf.layers.conv2d(
                inputs=inputs,
                filters=filter_shape[-1],
                kernel_size=filter_shape[0:2],
                strides=strides,
                padding=padding,
                activation=activation,
                kernel_initializer=tf.random_normal_initializer(),
                name=name
                )
            return tf.nn.relu(conv)

        def upconv2d(name, inputs, filter_shape, strides, padding='SAME'):
            """
            conv2d transpose wrapper

            :param name:
            :param inputs: NHWC
            :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
            :param strides: A list of 2 positive integers specifying the strides of the convolution
            :param padding:
            :return:
            """
            upconv = tf.layers.conv2d_transpose(
                inputs=inputs,
                filters=filter_shape[-1],
                kernel_size=filter_shape[0:2],
                strides=strides,
                padding=padding,
                activation=None,
                name=name
            )
            return upconv

        with tf.variable_scope("UNet"):
            # net down
            image = self._input_data
            conv1_1 = conv2d(name='conv1_1', inputs=image, filter_shape=[3, 3, 1, 16], strides=[1, 1])
            conv1_2 = conv2d(name='conv1_2', inputs=conv1_1, filter_shape=[3, 3, 16, 16], strides=[1, 1])
            pool_1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_1 = tf.nn.dropout(x=pool_1, keep_prob=0.85)

            conv2_1 = conv2d(name='conv2_1', inputs=pool_1, filter_shape=[3, 3, 16, 48], strides=[1, 1])
            conv2_2 = conv2d(name='conv2_2', inputs=conv2_1, filter_shape=[3, 3, 48, 48], strides=[1, 1])
            pool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_2 = tf.nn.dropout(x=pool_2, keep_prob=0.85)

            conv3_1 = conv2d(name='conv3_1', inputs=pool_2, filter_shape=[3, 3, 48, 96], strides=[1, 1])
            conv3_2 = conv2d(name='conv3_2', inputs=conv3_1, filter_shape=[3, 3, 96, 96], strides=[1, 1])
            pool_3 = tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_3 = tf.nn.dropout(x=pool_3, keep_prob=0.85)

            conv4_1 = conv2d(name='conv4_1', inputs=pool_3, filter_shape=[3, 3, 96, 128], strides=[1, 1])
            conv4_2 = conv2d(name='conv4_2', inputs=conv4_1, filter_shape=[3, 3, 128, 128], strides=[1, 1])
            pool_4 = tf.nn.max_pool(conv4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_4 = tf.nn.dropout(x=pool_4, keep_prob=0.85)

            # bottom
            conv5_1 = conv2d(name='conv5_1', inputs=pool_4, filter_shape=[3, 3, 128, 192], strides=[1, 1])
            conv5_2 = conv2d(name='conv5_2', inputs=conv5_1, filter_shape=[3, 3, 192, 192], strides=[1, 1])

            # net up
            upconv6_1 = upconv2d(name='upconv6_1', inputs=conv5_2, filter_shape=[3, 3, 192, 128], strides=[2, 2])
            concat6_1 = tf.concat([upconv6_1, conv4_2], axis=3)
            conv6_2 = conv2d(name='conv6_2', inputs=concat6_1, filter_shape=[3, 3, 256, 128], strides=[1, 1])
            conv6_3 = conv2d(name='conv6_3', inputs=conv6_2, filter_shape=[3, 3, 128, 128], strides=[1, 1])
            conv6_3 = tf.nn.dropout(x=conv6_3, keep_prob=0.85)

            upconv7_1 = upconv2d(name='upconv7_1', inputs=conv6_3, filter_shape=[2, 2, 128, 96], strides=[2, 2])
            concat7_1 = tf.concat([upconv7_1, conv3_2], axis=3)
            conv7_2 = conv2d(name='conv7_2', inputs=concat7_1, filter_shape=[3, 3, 192, 96], strides=[1, 1])
            conv7_3 = conv2d(name='conv7_3', inputs=conv7_2, filter_shape=[3, 3, 96, 96], strides=[1, 1])
            conv7_3 = tf.nn.dropout(x=conv7_3, keep_prob=0.85)

            upconv8_1 = upconv2d(name='upconv8_1', inputs=conv7_3, filter_shape=[2, 2, 96, 48], strides=[2, 2])
            concat8_1 = tf.concat([upconv8_1, conv2_2], axis=3)
            conv8_2 = conv2d(name='conv8_2', inputs=concat8_1, filter_shape=[3, 3, 96, 48], strides=[1, 1])
            conv8_3 = conv2d(name='conv8_3', inputs=conv8_2, filter_shape=[3, 3, 48, 48], strides=[1, 1])
            conv8_3 = tf.nn.dropout(x=conv8_3, keep_prob=0.85)

            upconv9_1 = upconv2d(name='upconv9_1', inputs=conv8_3, filter_shape=[2, 2, 48, 16], strides=[2, 2])
            concat9_1 = tf.concat([upconv9_1, conv1_2], axis=3)
            conv9_2 = conv2d(name='conv9_2', inputs=concat9_1, filter_shape=[3, 3, 32, 16], strides=[1, 1])
            conv9_3 = conv2d(name='conv9_3', inputs=conv9_2, filter_shape=[3, 3, 16, 16], strides=[1, 1])
            conv9_3 = tf.nn.dropout(x=conv9_3, keep_prob=0.85)

            logits = conv2d(name='conv9_4', inputs=conv9_3, filter_shape=[1, 1, 16, 1], strides=[1, 1], activation=tf.nn.sigmoid)

            self._infer_labels = tf.to_float(tf.round(logits))

        with tf.variable_scope("loss"):
            gt_labels = tf.reshape(self._ground_truth[:, :, :, 0], shape=[self._batch_size, -1])
            infer_probs = tf.reshape(logits[:, :, :, 0], shape=[self._batch_size, -1])
            infer_labels = tf.to_float(tf.round(infer_probs))

            # figure out IOU
            self._iou_score = UNetModel.get_iou(gt_labels=gt_labels, infer_labels=infer_labels)

            self._loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=gt_labels, y_pred=infer_probs))

            # add summary operations
            tf.summary.scalar('loss', self._loss)

        with tf.variable_scope("optimization"):
            train_op = tf.train.AdamOptimizer(self._learning_rate)
            self._optimizer = train_op.minimize(self._loss)
            self._summary_op = tf.summary.merge_all()

    @staticmethod
    def get_iou(gt_labels, infer_labels):
        inter = tf.reduce_sum(tf.multiply(gt_labels, infer_labels), axis=-1)
        union = tf.cast(tf.count_nonzero(tf.add(gt_labels, infer_labels), axis=-1), tf.float32)
        eps = tf.constant(0.001)
        iou_score = inter / (union + eps)
        return tf.reduce_mean(iou_score)

    @property
    def loss(self):
        return self._loss

    @property
    def infer_labels(self):
        return self._infer_labels

    @property
    def summary_op(self):
        return self._summary_op

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def ground_truth(self):
        return self._ground_truth

    @property
    def gt_labels(self):
        return self._ground_truth[:, :, :, 0]

    @property
    def input_data(self):
        return self._input_data

    @property
    def iou_score(self):
        return self._iou_score
