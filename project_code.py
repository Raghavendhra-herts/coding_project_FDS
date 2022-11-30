# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 19:40:41 2022

@author: Raghavendhra Rao
"""

import numpy as np
#import matplotlib_inline
from sklearn.linear_model import LinearRegression
import pandas as pd

read_data = np.loadtxt("inputdata7.csv", delimiter="," , dtype=str)
#print(read_data)

# read the data into Python numpy array(s)
read_data = np.loadtxt("inputdata7.csv", delimiter="," , dtype=str, skiprows=1)
# print("\n \n", read_data)

# plot the data as a two-dimensional scatter plot,
x , y = np.transpose(read_data)
#plt.scatter(x, y)
#plt.xlim(0, 15)

# create a linear regression model based on the data
print(x.reshape(-1,1))