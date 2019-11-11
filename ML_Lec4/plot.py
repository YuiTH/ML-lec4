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

def plot_step(total_acc,total_loss,x,y,num_class,w_list,b_list):

    plt.figure(1)
    plt.ion()
    plt.cla()
    plt.subplot(221)
    plt.scatter(range(0,len(total_acc),10),total_acc[::10],color='blue')  # acc plot
    plt.plot(range(len(total_acc)),total_acc,color='blue')
    plt.subplot(222)
    plt.scatter(range(0,len(total_loss),10),total_loss[::10],color='red')  # loss plot
    plt.plot(range(len(total_loss)),total_loss,color='red')
    plt.subplot(223)
    # plt.plot([3,4],[4,5])

    for i in range(num_class):
        xx = x[y==i]
        plt.scatter(xx[:,0],xx[:,1],s=5)

    # plt.scatter(x[:,0],x[:,1],s=5)
    plt.pause(0.005)
    # plt.ioff()
    plt.show()


def plot_steps(total_acc, total_loss, x, y, num_class, w_list, b_list):
    x=preprocess(x)
	# for i in range(len(total_acc)):
    #     plot_step(total_acc[:i],
    #     total_loss[:i], 
    #     x, y, num_class,
    #     w_list[:i], b_list[:i])
    plt.ioff()
    plt.show()
    print(logistic_reg_predict(w_list[-1],x,b_list[0]))
    
