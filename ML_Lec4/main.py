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

x2, y2 = get2ClassData()
x3, y3 = get3ClassData()
epochs = 500
lr = 0.001
x, y = x2, y2
num_class = int(np.max(y))+1
#bi_log_reg(x, y, epochs, lr)

(total_acc, total_loss, w_list, b_list, acc_best_epoch, loss_best_epoch) = bi_log_reg(x2, y2, epochs, lr)  # best acc = 0.825000, epoch = 94
# (total_acc, total_loss, w_list, b_list, acc_best_epoch, loss_best_epoch) = bi_perceptron(x2, y2, epochs, lr) # best acc = 0.825000, epoch = 68


# (total_acc, total_loss, w_list, b_list, acc_best_epoch, loss_best_epoch) = mul_perceptron(x, y, epochs, lr)  #best acc = 0.966667, epoch = 56
plot_steps(total_acc, total_loss, x, y, num_class, w_list, b_list)
# x=preprocess(x)
# for i in range(len(w_list)):
#     print(i)
#     print(logistic_reg_predict_z(w_list[i],x[:40],b_list[i]))
#     print("——")
#     print(logistic_reg_predict_z(w_list[i],x[40:],b_list[i]))
