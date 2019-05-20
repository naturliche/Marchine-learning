# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:14:31 2019

@author: natur
"""

import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


x = [1,2,2,3,3,4,5,6,6,6,8,10]
x = np.reshape(x, newshape = (12,1))
y = [-890,-1411,-1560,-2220,-2091,-2878,-3537,-3268,-3920,-4163,-5471,-5157]
y = np.reshape(y, newshape = (12,1))

#调用
model = LinearRegression()
#训练模型
model.fit(x,y)
#计算R**2
print(model.score(x,y))
#计算y_hat 预测值
y_hat = model.predict(x)
#计算两个参数
a = model.intercept_
b = model.coef_
print(a)
print(b)

plt.scatter(x,y)
plt.plot(x,y_hat)