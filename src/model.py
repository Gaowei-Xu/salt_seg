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


class UNetModel(object):
    """
    UNet model
    Reference:  https://arxiv.org/pdf/1505.04597.pdf
                https://www.kaggle.com/shenmbsw/data-augmentation-and-tensorflow-u-net
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

    # def build_model(self):
    #     """
    #     build U-Net model
    #     :return:
    #     """
    #     def conv2d(inputs, filters, kernel_size, name, strides=(1, 1), padding="SAME"):
    #         return tf.layers.conv2d(
    #             inputs=inputs,
    #             filters=filters,
    #             kernel_size=kernel_size,
    #             strides=strides,
    #             padding=padding,
    #             activation=tf.nn.relu,
    #             name=name,
    #             use_bias=True,
    #             kernel_initializer=tf.random_normal_initializer(),
    #             bias_initializer=tf.constant_initializer(0),
    #             data_format='channels_last')
    #
    #     def deconv2d(inputs, kernel_size, output_size, in_channels, out_channels, name, strides=[1, 1, 1, 1], activation_is_relu=True):
    #         batch_size = inputs.get_shape()[0]
    #         out_shape = tf.stack([batch_size, output_size, output_size, out_channels])
    #         filter = tf.get_variable(name=name, shape=[kernel_size, kernel_size, out_channels, in_channels],
    #                                  initializer=tf.random_normal_initializer())
    #         h1 = tf.nn.conv2d_transpose(inputs, filter, out_shape, strides, padding='SAME', data_format="NHWC")
    #         if activation_is_relu:
    #             return tf.nn.relu(h1)
    #         else:
    #             return tf.nn.sigmoid(h1)
    #
    #     with tf.variable_scope("net_down"):
    #         image = self._input_data                                                            # (101, 101, 1)
    #         conv1 = conv2d(image, filters=8, kernel_size=3, name="conv_1", strides=(1, 1))      # (101, 101, 8)
    #         conv2 = conv2d(conv1, filters=16, kernel_size=3, name="conv_2", strides=(2, 2))     # (51, 51, 16)
    #         conv3 = conv2d(conv2, filters=16, kernel_size=3, name="conv_3", strides=(1, 1))     # (51, 51, 16)
    #         conv4 = conv2d(conv3, filters=32, kernel_size=3, name="conv_4", strides=(2, 2))     # (26, 26, 32)
    #         conv5 = conv2d(conv4, filters=32, kernel_size=3, name="conv_5", strides=(1, 1))     # (26, 26, 32)
    #         conv6 = conv2d(conv5, filters=64, kernel_size=3, name="conv_6", strides=(2, 2))     # (13, 13, 64)
    #         conv7 = conv2d(conv6, filters=64, kernel_size=3, name="conv_7", strides=(1, 1))     # (13, 13, 64)
    #         conv8 = conv2d(conv7, filters=96, kernel_size=3, name="conv_8", strides=(2, 2))     # (7, 7, 96)
    #         conv9 = conv2d(conv8, filters=96, kernel_size=3, name="conv_9", strides=(1, 1))     # (7, 7, 96)
    #         conv10 = conv2d(conv9, filters=128, kernel_size=3, name="conv_10", strides=(2, 2))  # (4, 4, 128)
    #
    #     with tf.variable_scope("net_up"):
    #         conv11 = deconv2d(conv10, kernel_size=3, output_size=4, in_channels=128, out_channels=128, name="deconv_11")                       # (4, 4, 128)
    #         conv12 = deconv2d(conv11, kernel_size=3, output_size=7, in_channels=128, out_channels=96, name="deconv_12", strides=[1, 2, 2, 1])  # (7, 7, 96)
    #         conv12 = tf.concat([conv9, conv12], axis=-1)                                                                                        # (7, 7, 192)
    #         conv13 = deconv2d(conv12, kernel_size=3, output_size=7, in_channels=192, out_channels=96, name="deconv_13")                        # (7, 7, 96)
    #
    #         conv14 = deconv2d(conv13, kernel_size=3, output_size=13, in_channels=96, out_channels=64, name="deconv_14", strides=[1, 2, 2, 1])  # (13, 13, 64)
    #         conv14 = tf.concat([conv7, conv14], axis=-1)                                                                                        # (13, 13, 128)
    #         conv15 = deconv2d(conv14, kernel_size=3, output_size=13, in_channels=128, out_channels=64, name="deconv_15")                       # (13, 13, 64)
    #
    #         conv16 = deconv2d(conv15, kernel_size=3, output_size=26, in_channels=64, out_channels=32, name="deconv_16", strides=[1, 2, 2, 1])  # (26, 26, 32)
    #         conv16 = tf.concat([conv5, conv16], axis=-1)                                                                                        # (26, 26, 64)
    #         conv17 = deconv2d(conv16, kernel_size=3, output_size=26, in_channels=64, out_channels=32, name="deconv_17")                        # (26, 26, 32)
    #
    #         conv18 = deconv2d(conv17, kernel_size=3, output_size=51, in_channels=32, out_channels=16, name="deconv_18", strides=[1, 2, 2, 1])  # (51, 51, 16)
    #         conv18 = tf.concat([conv3, conv18], axis=-1)                                                                                        # (51, 51, 32)
    #         conv19 = deconv2d(conv18, kernel_size=3, output_size=51, in_channels=32, out_channels=16, name="deconv_19")                        # (51, 51, 16)
    #
    #         conv20 = deconv2d(conv19, kernel_size=3, output_size=101, in_channels=16, out_channels=8, name="deconv_20", strides=[1, 2, 2, 1])  # (101, 101, 8)
    #         conv20 = tf.concat([conv1, conv20], axis=-1)                                                                                        # (101, 101, 16)
    #         conv21 = deconv2d(conv20, kernel_size=3, output_size=101, in_channels=16, out_channels=8, name="deconv_21")                        # (101, 101, 8)
    #
    #         logits = deconv2d(conv21, kernel_size=3, output_size=101, in_channels=8, out_channels=2, name="logits", strides=[1, 1, 1, 1], activation_is_relu=False)      # (101, 101, 2)
    #
    #         #logits = tf.nn.softmax(logits)
    #         self._infer_labels = logits[:, :, :, 0]
    #         #self._infer_labels = tf.round(logits[:, :, :, 0])
    #
    #     with tf.variable_scope("loss"):
    #         gt_labels = tf.reshape(self._ground_truth[:, :, :, 0], shape=[self._batch_size, -1])
    #         infer_logits = tf.reshape(logits[:, :, :, 0], shape=[self._batch_size, -1])
    #         intersection = tf.reduce_sum(tf.multiply(infer_logits, gt_labels), 1)
    #
    #         epsilon = tf.constant(value=1e-4)
    #         match_rate_batch = (2 * intersection + epsilon) / (tf.reduce_sum(tf.multiply(infer_logits, infer_logits), 1) +
    #                                                            tf.reduce_sum(tf.multiply(gt_labels, gt_labels), 1) + epsilon)
    #         match_rate = tf.reduce_mean(match_rate_batch)
    #         self._loss = 1.0 - match_rate
    #
    #         # add summary operations
    #         tf.summary.scalar('loss', self._loss)
    #
    #     with tf.variable_scope("optimization"):
    #         train_op = tf.train.AdamOptimizer(self._learning_rate)
    #         self._optimizer = train_op.minimize(self._loss)
    #         self._summary_op = tf.summary.merge_all()

    def build_model(self):
        """
        build Unet model
        :return:
        """

        def get_variable(name, shape):
            return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

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
                activation=tf.nn.relu,
                name=name
            )
            return upconv

        with tf.variable_scope("UNet"):
            # net down
            image = self._input_data
            conv1_1 = conv2d(name='conv1_1', inputs=image, filter_shape=[3, 3, 1, 16], strides=[1, 1])
            conv1_2 = conv2d(name='conv1_2', inputs=conv1_1, filter_shape=[3, 3, 16, 16], strides=[1, 1])
            pool_1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv2_1 = conv2d(name='conv2_1', inputs=pool_1, filter_shape=[3, 3, 16, 48], strides=[1, 1])
            conv2_2 = conv2d(name='conv2_2', inputs=conv2_1, filter_shape=[3, 3, 48, 48], strides=[1, 1])
            pool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv3_1 = conv2d(name='conv3_1', inputs=pool_2, filter_shape=[3, 3, 48, 96], strides=[1, 1])
            conv3_2 = conv2d(name='conv3_2', inputs=conv3_1, filter_shape=[3, 3, 96, 96], strides=[1, 1])
            pool_3 = tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            conv4_1 = conv2d(name='conv4_1', inputs=pool_3, filter_shape=[3, 3, 96, 128], strides=[1, 1])
            conv4_2 = conv2d(name='conv4_2', inputs=conv4_1, filter_shape=[3, 3, 128, 128], strides=[1, 1])
            pool_4 = tf.nn.max_pool(conv4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # bottom
            conv5_1 = conv2d(name='conv5_1', inputs=pool_4, filter_shape=[3, 3, 128, 192], strides=[1, 1])
            conv5_2 = conv2d(name='conv5_2', inputs=conv5_1, filter_shape=[3, 3, 192, 192], strides=[1, 1])

            # net up
            upconv6_1 = upconv2d(name='upconv6_1', inputs=conv5_2, filter_shape=[3, 3, 192, 128], strides=[2, 2])
            concat6_1 = tf.concat([upconv6_1, conv4_2], axis=3)
            conv6_2 = conv2d(name='conv6_2', inputs=concat6_1, filter_shape=[3, 3, 256, 128], strides=[1, 1])
            conv6_3 = conv2d(name='conv6_3', inputs=conv6_2, filter_shape=[3, 3, 128, 128], strides=[1, 1])

            upconv7_1 = upconv2d(name='upconv7_1', inputs=conv6_3, filter_shape=[2, 2, 128, 96], strides=[2, 2])
            concat7_1 = tf.concat([upconv7_1, conv3_2], axis=3)
            conv7_2 = conv2d(name='conv7_2', inputs=concat7_1, filter_shape=[3, 3, 192, 96], strides=[1, 1])
            conv7_3 = conv2d(name='conv7_3', inputs=conv7_2, filter_shape=[3, 3, 96, 96], strides=[1, 1])

            upconv8_1 = upconv2d(name='upconv8_1', inputs=conv7_3, filter_shape=[2, 2, 96, 48], strides=[2, 2])
            concat8_1 = tf.concat([upconv8_1, conv2_2], axis=3)
            conv8_2 = conv2d(name='conv8_2', inputs=concat8_1, filter_shape=[3, 3, 96, 48], strides=[1, 1])
            conv8_3 = conv2d(name='conv8_3', inputs=conv8_2, filter_shape=[3, 3, 48, 48], strides=[1, 1])

            upconv9_1 = upconv2d(name='upconv9_1', inputs=conv8_3, filter_shape=[2, 2, 48, 16], strides=[2, 2])
            concat9_1 = tf.concat([upconv9_1, conv1_2], axis=3)
            conv9_2 = conv2d(name='conv9_2', inputs=concat9_1, filter_shape=[3, 3, 32, 16], strides=[1, 1])
            conv9_3 = conv2d(name='conv9_3', inputs=conv9_2, filter_shape=[3, 3, 16, 16], strides=[1, 1])

            logits = conv2d(name='conv9_4', inputs=conv9_3, filter_shape=[3, 3, 16, 2], strides=[1, 1], activation=tf.nn.sigmoid)

            self._infer_labels = logits[:, :, :, 0]
            #self._infer_labels = tf.round(logits[:, :, :, 0])

        # with tf.variable_scope('loss'):
        #     gt_labels = tf.reshape(self._ground_truth[:, :, :, 0], shape=[self._batch_size, -1])
        #     infer_logits = tf.reshape(logits[:, :, :, 0], shape=[self._batch_size, -1])
        #     infer_labels = tf.reshape(tf.to_int64(logits[:, :, :, 0] > 0.5), shape=[self._batch_size, -1])
        #     self._loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(gt_labels, infer_logits))
        #     score, up_opt = tf.metrics.mean_iou(gt_labels, infer_labels, 2)
        #     score = tf.identity(score)
        #
        # with tf.variable_scope("optimization"):
        #     train_op = tf.train.AdamOptimizer(self._learning_rate)
        #     self._optimizer = train_op.minimize(self._loss)
        #     self._summary_op = tf.summary.merge_all()

        with tf.variable_scope("loss"):
            gt_labels = tf.reshape(self._ground_truth[:, :, :, 0], shape=[self._batch_size, -1])
            infer_probs = tf.reshape(logits[:, :, :, 0], shape=[self._batch_size, -1])
            infer_labels = tf.to_float(tf.reshape(tf.to_int64(logits[:, :, :, 0] > 0.5), shape=[self._batch_size, -1]))

            # figure out IOU
            intersection = tf.reduce_sum(tf.multiply(infer_labels, gt_labels), 1)
            epsilon = tf.constant(value=1e-3)
            match_rate_batch = (2 * intersection + epsilon) / (tf.reduce_sum(tf.multiply(infer_labels, infer_labels), 1) +
                                                               tf.reduce_sum(tf.multiply(gt_labels, gt_labels), 1) + epsilon)
            self._iou_score = tf.reduce_mean(match_rate_batch)
            # self._loss = 1.0 - self._iou_score
            self._loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(gt_labels, infer_probs))

            # add summary operations
            tf.summary.scalar('loss', self._loss)

        with tf.variable_scope("optimization"):
            train_op = tf.train.AdamOptimizer(self._learning_rate)
            self._optimizer = train_op.minimize(self._loss)
            self._summary_op = tf.summary.merge_all()


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
