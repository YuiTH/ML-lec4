# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:06:56 2019

@author: Lenovo
"""

# normalization
from readFile import get3ClassData
import numpy as np
import matplotlib.pyplot as plt


def preprocess(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_normal = (x - x_mean) / x_std
    return x_normal
    
    
#x, y = get3ClassData()
#x_mean = np.mean(x, axis=0)
#x_std = np.std(x, axis=0)
#x_normal = (x - x_mean) / x_std
#
#x = x_normal
#x0, y0 = x[0:50], y[0:50]
#x1, y1 = x[50:100], y[50:100]
#x2, y2 = x[100:150], y[100:150]
#
#plt.scatter(x0[:,0], x0[:,1], c='green')
#plt.scatter(x1[:,0], x1[:,1], c='blue')
#plt.scatter(x2[:,0], x2[:,1], c='red')







