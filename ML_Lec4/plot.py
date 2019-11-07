# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:32:09 2019

@author: Lenovo
"""

from readFile import get3ClassData
import numpy as np
import matplotlib.pyplot as plt

x, y = get3ClassData()
x0, y0 = x[0:50], y[0:50]
x1, y1 = x[50:100], y[50:100]
x2, y2 = x[100:150], y[100:150]

plt.scatter(x0[:,0], x0[:,1], c='green')
plt.scatter(x1[:,0], x1[:,1], c='blue')
plt.scatter(x2[:,0], x2[:,1], c='red')


