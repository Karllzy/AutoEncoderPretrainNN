#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18

参考: https://juejin.im/post/5acfef976fb9a028db5918b5
"""

import numpy as np


def prp_2_oh_array(arr):
    """
    概率矩阵转换为OH矩阵
    arr = np.array([[0.1, 0.5, 0.4], [0.2, 0.1, 0.6]])
    :param arr: 概率矩阵
    :return: OH矩阵
    """
    arr_size = arr.shape[1]  # 类别数
    arr_max = np.argmax(arr, axis=1)  # 最大值位置
    oh_arr = np.eye(arr_size)[arr_max]  # OH矩阵
    return oh_arr


def feature_cc(feature, y_test):
    self.cc = np.zeros((feature.shape[0], feature.shape[1]), dtype=np.float32)
    for i in range(feature.shape[1]):
        self.cc[:, i] = abs(np.corrcoef(feature[:, i], y_test[:, 0])[0, 1])

    cc_max = np.max(self.cc)
    cc_min = np.min(self.cc)
    self.cc =(self.cc - cc_min) / (cc_max - cc_min)
    self.cc = np.sqrt(self.cc)
    print(self.cc.shape)