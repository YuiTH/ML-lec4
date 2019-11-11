# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:32:09 2019

@author: Lenovo
"""

from readFile import get3ClassData
import numpy as np
import matplotlib.pyplot as plt
from bi_logistic_reg_sgd import logistic_reg_predict
from preprocess import preprocess


# x, y = get3ClassData()
# x0, y0 = x[0:50], y[0:50]
# x1, y1 = x[50:100], y[50:100]
# x2, y2 = x[100:150], y[100:150]

# plt.scatter(x0[:,0], x0[:,1], c='green')
# plt.scatter(x1[:,0], x1[:,1], c='blue')
# plt.scatter(x2[:,0], x2[:,1], c='red')


def plot_step(total_acc, total_loss, x, y, num_class, w_list, b_list,pred_fun):
    if len(w_list) == 0:
        return
    # plt.figure(1)
    plt.ion()
    plt.cla()
    plt.subplot(221)
    plt.title('Acc')
    plt.scatter(range(0, len(total_acc), 5),
                total_acc[::5], s=9,color='blue')  # acc plot
    plt.plot(range(len(total_acc)), total_acc, color='blue')
    plt.subplot(222)
    plt.title('Loss')

    plt.scatter(range(0, len(total_loss), 5),
                total_loss[::5], s=9,color='red')  # loss plot
    plt.plot(range(len(total_loss)), total_loss, color='red')
    plt.subplot(223)
    # plot_decision_boundary(logistic_reg_predict, x, w_list[-1], b_list[-1], y)
    plot_decision_boundary(pred_fun, x, w_list[-1], b_list[-1], y)

    # plt.plot([3,4],[4,5])

    # for i in range(num_class):
    #     xx = x[y == i]
    #     plt.scatter(xx[:, 0], xx[:, 1], s=5)

    # plt.scatter(x[:,0],x[:,1],s=5)
    plt.pause(0.005)
    # plt.ioff()
    plt.show()


def plot_steps(total_acc, total_loss, x, y, num_class, w_list, b_list,pred_fun):
    x = preprocess(x)
    if pred_fun == "per":
        f = predict_multi_perception
    elif pred_fun == "logi":
        f = logistic_reg_predict

    for i in range(len(total_acc)):
        plot_step(total_acc[:i],
        total_loss[:i],
        x, y, num_class,
        w_list[:i], b_list[:i],f)
    plt.ioff()
    plt.show()



def plot_decision_boundary(pred_func, X, w, b, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = pred_func(w, np.c_[xx.ravel(), yy.ravel()], b)
    Z = Z.reshape(xx.shape)
    # print(Z)


    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral,s=5)


def predict_multi_perception(w, x, b):  # (N, 3)
    z = x@w+b
    pred_index = z.argmax(axis=1)

    if z.shape[1] == 3:
        return pred_index
    return z > 0