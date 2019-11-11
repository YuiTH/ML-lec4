# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:10:03 2019

@author: Lenovo
"""

from bi_perceptron_sgd import bi_perceptron
from bi_logistic_reg_sgd import bi_log_reg
from multi_label_perceptron import mul_perceptron
from readFile import get2ClassData, get3ClassData
from plot import plot_steps
import numpy as np
from bi_logistic_reg_sgd import logistic_reg_predict_z
from preprocess import preprocess
import time

x2, y2 = get2ClassData()
x3, y3 = get3ClassData()
epochs = 200
lr = 0.001
x, y = x3, y3
num_class = int(np.max(y))+1
#bi_log_reg(x, y, epochs, lr)


def logistic_plot():
    starttime = time.time()
    (total_acc, total_loss, w_list, b_list, acc_best_epoch, loss_best_epoch) = bi_log_reg(
        x2, y2, epochs, lr)  # best acc = 0.825000, epoch = 94
    endtime = time.time()
    print("use", (endtime - starttime), "sec")
    xx2 = preprocess(x2)
    plot_steps(total_acc, total_loss, xx2, y2, 2, w_list, b_list, "logi")


def bi_perceptron_plot():
    starttime = time.time()
    (total_acc, total_loss, w_list, b_list, acc_best_epoch, loss_best_epoch) = bi_perceptron(
        x2, y2, epochs, lr)  # best acc = 0.825000, epoch = 68
    xx2 = preprocess(x2)
    endtime = time.time()
    print("use", (endtime - starttime), "sec")
    plot_steps(total_acc, total_loss, xx2, y2, 2, w_list, b_list, "per")


def multi_label_perceptron_plot():
    starttime = time.time()

    (total_acc, total_loss, w_list, b_list, acc_best_epoch, loss_best_epoch) = mul_perceptron(
        x3, y3, epochs, lr)  # best acc = 0.966667, epoch = 56
    endtime = time.time()
    print("use", (endtime - starttime), "sec")

    xx3 = preprocess(x3)
    plot_steps(total_acc, total_loss, xx3, y3, 3, w_list, b_list, "per")


if __name__ == "__main__":
    # logistic_plot()
    # bi_perceptron_plot()
    multi_label_perceptron_plot()
