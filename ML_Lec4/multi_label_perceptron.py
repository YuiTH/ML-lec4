# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from readFile import get2ClassData, get3ClassData
from preprocess import preprocess
import numpy as np



def transform_one_hot(labels):
  n_labels = np.max(labels) + 1
  one_hot = np.eye(n_labels)[labels.astype(int)]
  return one_hot


def compute_loss(z_total, y):  # (N, 3)
    y_index = y.argmax(axis=1)  # (N)
    pred_index = z_total.argmax(axis=1)  #　(N)
    right_count = (y_index==pred_index).sum()
    acc = (1.0*right_count) / y.shape[0]
    z_pred = np.array([z_total[i,j] for i, j in enumerate(pred_index)])
    z_true = np.array([z_total[i,j] for i,j in enumerate(y_index)])
    loss = (z_pred - z_true).sum()
    return loss, acc


def activate(z):  #(f_out)
    z_pred_index = np.argmax(z)  # (1)
    pred = np.zeros((1, z.shape)) #(f_out)
    pred[z_pred_index] = 1
    return pred


def mul_perceptron(x, y, epochs, lr):
    x = preprocess(x)   # preprocess  # (N,2)
    y = transform_one_hot(y.astype(int))   #(N, 3)

    print("Example of multi-label perceptron ")
    total_loss = []
    total_acc = []
    w_list = []
    b_list = []
    acc_best_val,loss_best_val = 0, float('inf')
    acc_best_epoch, loss_best_epoch = 0, 0
    
    x_dim, y_dim = x.shape[1], y.shape[1]
    w = np.random.standard_normal([x_dim, y_dim])  # (2,3) mean=0, stdev=1
    b = np.ones(y_dim)  # init b
    
    w_list.append(w.copy())
    b_list.append(b.copy())
    
    
    z_total = x.dot(w) + b
    # loss before training
    loss, acc = compute_loss(z_total, y)
    print("before training, loss = %f, acc = %f" % (loss, acc))
    total_loss.append(loss)
    total_acc.append(acc)
    # 进入epoch
    for epoch in range(epochs):
        shuffle_index = np.random.permutation(x.shape[0])
        x_shuffle = x[shuffle_index, :]
        y_shuffle = y[shuffle_index, :]
        # update param for each x_k
        for k, x_k in enumerate(x_shuffle):   # step
            x_k = x_k.reshape(-1, len(x_k))  # (1,2)
            pred_k = np.argmax(x_k.dot(w) + b)  # (1)
            y_k = y_shuffle[k].argmax()  # (1)
            
    #        y_k = y_shuffle[k]
    #        x_k = x_k.reshape(-1, len(x_k))  # (1,2)
    #        z_k = (x_k.dot(w) + b)   # (1,3)
    #        pred_k = activate(z_k.squeeze()).reshape(-1, len(y_k))  # (1,3)
    #        y_k = y_shuffle[k].reshape(-1, len(y_k))  # (1,3)
    #        
    #        dw = (z_k - y_k)
    
            if pred_k != y_k:  # predict wrong, and update param
                w[:, pred_k] = w[:, pred_k] - lr * x_k
                b[pred_k] = b[pred_k] - lr * 1
                w[:,y_k] = w[:,y_k] + lr * x_k 
                b[y_k] = b[y_k] + lr * 1
    
            else:  # predict right
                pass
        # loss of the epoch
        w_list.append(w.copy())
        b_list.append(b.copy())
        predict_total = x_shuffle.dot(w) + b
        loss, acc = compute_loss(predict_total, y_shuffle)
        if loss < loss_best_val:
            loss_best_val = loss
            loss_best_epoch = epoch
        
        if acc > acc_best_val:
            acc_best_val = acc
            acc_best_epoch = epoch
        
        total_loss.append(loss)
        total_acc.append(acc)
        if (epoch+1) % 10 == 0:
            print("epoch %d, loss = %f, acc = %f" % (epoch+1, loss, acc))
            
    print("best acc = %f, epoch = %d" % (acc_best_val, acc_best_epoch))
    print('best loss = %f, epoch = %d ' % (loss_best_val, loss_best_epoch))
    
    return (total_acc, total_loss, w_list, b_list, acc_best_epoch, loss_best_epoch)

    
    


def test():
    # global
    epochs = 500
    lr = 0.001
    print("Example of multi-label perceptron ")
    total_loss = []
    total_acc = []
    w_list = []
    b_list = []
    acc_best_val,loss_best_val = 0, float('inf')
    acc_best_epoch, loss_best_epoch = 0, 0
    
    
    np.random.seed(0)
    x, y = get2ClassData()
    x = preprocess(x)   # preprocess  # (N,2)
    y = transform_one_hot(y.astype(int))   #(N, 3)
    x_dim, y_dim = x.shape[1], y.shape[1]
    w = np.random.standard_normal([x_dim, y_dim])  # (2,3) mean=0, stdev=1
    b = np.ones(y_dim)  # init b
    
    w_list.append(w.copy())
    b_list.append(b.copy())
    
    
    z_total = x.dot(w) + b
    
    # loss before training
    loss, acc = compute_loss(z_total, y)
    print("before training, loss = %f, acc = %f" % (loss, acc))
    total_loss.append(loss)
    total_acc.append(acc)
    # 进入epoch
    for epoch in range(epochs):
        shuffle_index = np.random.permutation(x.shape[0])
        x_shuffle = x[shuffle_index, :]
        y_shuffle = y[shuffle_index, :]
        # update param for each x_k
        for k, x_k in enumerate(x_shuffle):   # step
            x_k = x_k.reshape(-1, len(x_k))  # (1,2)
            pred_k = np.argmax(x_k.dot(w) + b)  # (1)
            y_k = y_shuffle[k].argmax()  # (1)
            
    #        y_k = y_shuffle[k]
    #        x_k = x_k.reshape(-1, len(x_k))  # (1,2)
    #        z_k = (x_k.dot(w) + b)   # (1,3)
    #        pred_k = activate(z_k.squeeze()).reshape(-1, len(y_k))  # (1,3)
    #        y_k = y_shuffle[k].reshape(-1, len(y_k))  # (1,3)
    #        
    #        dw = (z_k - y_k)
    
            if pred_k != y_k:  # predict wrong, and update param
                w[:, pred_k] = w[:, pred_k] - lr * x_k
                b[pred_k] = b[pred_k] - lr * 1
                w[:,y_k] = w[:,y_k] + lr * x_k 
                b[y_k] = b[y_k] + lr * 1
    
            else:  # predict right
                pass
        # loss of the epoch
        w_list.append(w.copy())
        b_list.append(b.copy())
        predict_total = x_shuffle.dot(w) + b
        loss, acc = compute_loss(predict_total, y_shuffle)
        if loss < loss_best_val:
            loss_best_val = loss
            loss_best_epoch = epoch
        
        if acc > acc_best_val:
            acc_best_val = acc
            acc_best_epoch = epoch
        
        total_loss.append(loss)
        total_acc.append(acc)
        if (epoch+1) % 10 == 0:
            print("epoch %d, loss = %f, acc = %f" % (epoch+1, loss, acc))
            
    print("best acc = %f, epoch = %d" % (acc_best_val, acc_best_epoch))
    print('best loss = %f, epoch = %d ' % (loss_best_val, loss_best_epoch))
        



# /test
#w[:,2]
#right_count = (y_index==predict_index.sum())
# temp1, temp2 = np.random.shuffle()
# /test
