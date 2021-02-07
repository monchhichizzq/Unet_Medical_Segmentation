# -*- coding: utf-8 -*-
# @Time    : 2021/2/7 23:50
# @Author  : Zeqi@@
# @FileName: metrics.py
# @Software: PyCharm



import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def dice_loss_with_CE(beta=1, smooth=1e-5):
    def _dice_loss_with_CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[..., :-1] * K.log(y_pred)
        CE_loss = K.mean(K.sum(CE_loss, axis=-1))

        tp = K.sum(y_true[..., :-1] * y_pred, axis=[0, 1, 2])
        fp = K.sum(y_pred, axis=[0, 1, 2]) - tp
        fn = K.sum(y_true[..., :-1], axis=[0, 1, 2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score
        # dice_loss = tf.Print(dice_loss, [dice_loss, CE_loss])
        return CE_loss + dice_loss

    return _dice_loss_with_CE


def CE():
    def _CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[..., :-1] * K.log(y_pred)
        CE_loss = K.mean(K.sum(CE_loss, axis=-1))
        # dice_loss = tf.Print(CE_loss, [CE_loss])
        return CE_loss

    return _CE


def Iou_score(smooth=1e-5, threshold=0.5):
    def _Iou_score(y_true, y_pred):
        # score calculation
        y_pred = K.greater(y_pred, threshold)
        y_pred = K.cast(y_pred, K.floatx())

        intersection = K.sum(y_true[..., :-1] * y_pred, axis=[0, 1, 2])
        union = K.sum(y_true[..., :-1] + y_pred, axis=[0, 1, 2]) - intersection

        score = (intersection + smooth) / (union + smooth)
        return score

    return _Iou_score


def f_score(beta=1, smooth=1e-5, threhold=0.5):
    def _f_score(y_true, y_pred):
        y_pred = K.greater(y_pred, threhold)
        y_pred = K.cast(y_pred, K.floatx())

        tp = K.sum(y_true[..., :-1] * y_pred, axis=[0, 1, 2])
        fp = K.sum(y_pred, axis=[0, 1, 2]) - tp
        fn = K.sum(y_true[..., :-1], axis=[0, 1, 2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        return score

    return _f_score

