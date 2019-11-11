# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:08:17 2019

@author: admin
"""


from readFile import get2ClassData, get3ClassData
from preprocess import preprocess
import numpy as np




def signal(z):
    return 1 if(z>=0) else 0


def compute_loss(z_total, y):
    pred_total = [signal(z) for z in z_total]
    right_count = (pred_total == y).sum()
    acc = (1.0*right_count) / y.shape[0]
    loss = ((pred_total - y) * z_total).sum()
    return loss, acc





def bi_perceptron(x, y, epochs, lr):
    print("Example of multi-label perceptron ")

    x = preprocess(x)   # preprocess

    total_loss = []
    total_acc = []
    w_list = []
    b_list = []
    acc_best_val,loss_best_val = 0, float('inf')
    acc_best_epoch, loss_best_epoch = 0, 0

    x_dim, y_dim = x.shape[1], 1
    w = np.random.standard_normal([x_dim, y_dim])  # (2,1) mean=0, stdev=1
    b = np.ones(y_dim)  # (1)
    
    w_list.append(w.copy())
    b_list.append(b.copy())
    z_total = x.dot(w) + b
    loss, acc = compute_loss(z_total, y)
    print("before training, loss = %f, acc = %f" % (loss, acc))
    total_loss.append(loss)
    total_acc.append(acc)
    
    # 进入epoch
    for epoch in range(epochs):
        shuffle_index = np.random.permutation(x.shape[0])   #(N)
        x_shuffle = x[shuffle_index, :]  # (N,2)
        y_shuffle = y[shuffle_index]  
        # update param for each x_k
        for k, x_k in enumerate(x_shuffle):   # step
            x_k = x_k.reshape(-1, len(x_k))  # (1,2)       
            y_k = y_shuffle[k]
            pred_k = signal(x_k.dot(w) + b)
            dw = (pred_k - y_k) * x_k.T   # (2,1)
            db = pred_k - y_k
            w -= lr * dw             # (2,1)
            b -= lr * db            # (1)
            
        # loss of the epoch
        w_list.append(w.copy())
        b_list.append(b.copy())
        z_total = x_shuffle.dot(w) + b
        loss, acc = compute_loss(z_total, y_shuffle)
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
    x = preprocess(x)   # preprocess
    x_dim, y_dim = x.shape[1], 1
    w = np.random.standard_normal([x_dim, y_dim])  # (2,1) mean=0, stdev=1
    b = np.ones(y_dim)  # (1)
    
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
        shuffle_index = np.random.permutation(x.shape[0])   #(N)
        x_shuffle = x[shuffle_index, :]  # (N,2)
        y_shuffle = y[shuffle_index]  
        # update param for each x_k
        for k, x_k in enumerate(x_shuffle):   # step
            x_k = x_k.reshape(-1, len(x_k))  # (1,2)       
            y_k = y_shuffle[k]
            pred_k = signal(x_k.dot(w) + b)
            dw = (pred_k - y_k) * x_k.T   # (2,1)
            db = pred_k - y_k
            w -= lr * dw             # (2,1)
            b -= lr * db            # (1)
            
        # loss of the epoch
        w_list.append(w.copy())
        b_list.append(b.copy())
        z_total = x_shuffle.dot(w) + b
        loss, acc = compute_loss(z_total, y_shuffle)
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
