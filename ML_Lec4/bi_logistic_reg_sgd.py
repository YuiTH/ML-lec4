# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:55:45 2019
# it works
@author: admin
"""

from readFile import get2ClassData, get3ClassData
import numpy as np
from preprocess import preprocess

# global

print("Example of binary logsitic reg ")


def compute_total_loss(predict_total, y):  # (N) (N)
    # loss
    loss_pos = (np.log(predict_total) * y).sum()
    loss_neg = (np.log(1 - predict_total) * (1 - y)).sum()
    loss = -1 * (loss_pos + loss_neg)
    # acc
    pred = np.array([1 if(i>=0.5) else 0 for i in predict_total])
    right_count = (y==pred).sum()
    acc = (1.0*right_count) / y.shape[0]
    
    return loss, acc


def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z))) 

def dsigmoid(z):
    return (z>0) * (sigmoid(z) * (1 - sigmoid(z)))


def relu(z):
    return (z>0) * z

def drelu(z):
    return (z>0) * 1

def act(z):
    return sigmoid(z)

def dact(z):
    return dsigmoid(z)

def logistic_reg_predict(w,x,b):
    return sigmoid(x@w+b) > 0.5

def logistic_reg_predict_z(w, x, b):
    return np.mean(sigmoid(x@w+b))

def bi_log_reg(x, y, epochs, lr):
    x = preprocess(x)   # preprocess

    total_loss = []
    total_acc = []
    w_list = []
    b_list = []
    acc_best_val,loss_best_val = 0, float('inf')
    acc_best_epoch, loss_best_epoch = 0, 0
    x_dim, y_dim = x.shape[1], 1
    w = np.random.standard_normal([x_dim, y_dim])  # (2,1) mean=0, stdev=1
    b = np.ones(y_dim)   # (1) init b
    # loss before training
    w_list.append(w.copy())
    b_list.append(b.copy())
    predict_total = act(x.dot(w) + b)
    loss, acc = compute_total_loss(predict_total, y)

    print("before training, loss = %f, acc = %f" % (loss, acc))
    total_loss.append(loss)
    total_acc.append(acc)

    for epoch in range(epochs):
        # shuffle
        shuffle_index = np.random.permutation(x.shape[0])
        x_shuffle = x[shuffle_index, :]
        y_shuffle = y[shuffle_index]
        # update param for each x_k
        for k, x_k in enumerate(x_shuffle):
            # transfer 2 matrix
            x_k = x_k.reshape(-1, len(x_k))  # (1,2)
            y_k = y_shuffle[k]
            # forward
            pred_k = act(x_k.dot(w) + b)  #(1,1)
    #        loss_k = -1.0 * ((y_k * np.log(pred_k) + (1 - y_k) * np.log(1 - pred_k)))  # (1,3)
            # backward
            dw = x_k.T.dot(pred_k - y_k)   # (2,1)= (2,1)*(1)
            db = (pred_k - y_k).sum(axis=0)            # (1)=(1)
            # update
            w -= lr * dw
            b -= lr * db


        w_list.append(w.copy())
        b_list.append(b.copy())
        # compute loss for the epoch
        predict_total = act(x_shuffle.dot(w) + b)  # (N)
        loss, acc = compute_total_loss(predict_total, y_shuffle)
        total_loss.append(loss)
        total_acc.append(acc)

        if loss < loss_best_val:
            loss_best_val = loss
            loss_best_epoch = epoch

        if acc > acc_best_val:
            acc_best_val = acc
            acc_best_epoch = epoch


        if (epoch+1) % 10 == 0:
            print("epoch %d, loss = %f, acc = %f" % (epoch+1, loss, acc))

    print("best acc = %f, epoch = %d" % (acc_best_val, acc_best_epoch))
    print('best loss = %f, epoch = %d ' % (loss_best_val, loss_best_epoch))

    return (total_acc, total_loss, w_list, b_list, acc_best_epoch, loss_best_epoch)



#
# def test():
#     epochs = 500
#     lr = 0.001
#
#     total_loss = []
#     total_acc = []
#     w_list = []
#     b_list = []
#     acc_best_val,loss_best_val = 0, float('inf')
#     acc_best_epoch, loss_best_epoch = 0, 0
#
#     np.random.seed(0)
#     x, y = get2ClassData()
#     x = preprocess(x)   # preprocess
#
#     x_dim, y_dim = x.shape[1], 1
#     w = np.random.standard_normal([x_dim, y_dim])  # (2,1) mean=0, stdev=1
#     b = np.ones(y_dim)   # (1) init b
#
#     # loss before training
#     predict_total = act(x.dot(w) + b)
#     loss, acc = compute_total_loss(predict_total, y)
#
#     print("before training, loss = %f, acc = %f" % (loss, acc))
#     total_loss.append(loss)
#     total_acc.append(acc)
#
#     for epoch in range(epochs):
#         # shuffle
#         shuffle_index = np.random.permutation(x.shape[0])
#         x_shuffle = x[shuffle_index, :]
#         y_shuffle = y[shuffle_index]
#         # update param for each x_k
#         for k, x_k in enumerate(x_shuffle):
#             # transfer 2 matrix
#             x_k = x_k.reshape(-1, len(x_k))  # (1,2)
#             y_k = y_shuffle[k]
#             # forward
#             pred_k = act(x_k.dot(w) + b)  #(1,1)
#     #        loss_k = -1.0 * ((y_k * np.log(pred_k) + (1 - y_k) * np.log(1 - pred_k)))  # (1,3)
#             # backward
#             dw = x_k.T.dot(pred_k - y_k)   # (2,1)= (2,1)*(1)
#             db = (pred_k - y_k).sum(axis=0)            # (1)=(1)
#             # update
#             w -= lr * dw
#             b -= lr * db
#
#
#         w_list.append(w.copy())
#         b_list.append(b.copy())
#         # compute loss for the epoch
#         predict_total = act(x_shuffle.dot(w) + b)  # (N)
#         loss, acc = compute_total_loss(predict_total, y_shuffle)
#
#         if loss < loss_best_val:
#             loss_best_val = loss
#             loss_best_epoch = epoch
#
#         if acc > acc_best_val:
#             acc_best_val = acc
#             acc_best_epoch = epoch
#
#         total_loss.append(loss)
#         total_acc.append(acc)
#         if (epoch+1) % 10 == 0:
#             print("epoch %d, loss = %f, acc = %f" % (epoch+1, loss, acc))
#
#     print("best acc = %f, epoch = %d" % (acc_best_val, acc_best_epoch))
#     print('best loss = %f, epoch = %d ' % (loss_best_val, loss_best_epoch))
#
#
