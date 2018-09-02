#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time    : 2018-09-01 15:01
# @Author  : Gaowei Xu
# @Email   : gaowxu@hotmail.com
# @File    : rle.py

import numpy as np


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
        bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
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

