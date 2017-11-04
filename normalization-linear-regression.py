# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:40:45 2017

@author: RITIKA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fileData = pd.read_csv('ex1data1.txt')

# set X (training data) and y (target variable)
cols = fileData.shape[1]  
X2 = fileData.iloc[:,0:cols-1]  
y2 = fileData.iloc[:,cols-1:cols]
X2.insert(0, 'Ones', 1)
#print(cols)
#print(X2)
#print(y2)
z = X2.T.dot(X2)

inv_mat = np.linalg.inv(z)
## ANS DFF To lp
theta = (inv_mat.dot(X2.T)).dot(y2)

print(theta)

print(np.array([1,3.5]).dot(theta))
print(np.array([1, 7]).dot(theta))
# feature scaling
# not reqd in vectoriztaion data = (fileData - fileData.mean())/fileData.std()

[[-4.21150401]
 [ 1.21354725]]
[ 0.03591138]
[ 4.28332677]



