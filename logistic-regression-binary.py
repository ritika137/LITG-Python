# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:39:30 2017

@author: RITIKA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures


fileData = pd.read_csv('ex2data1.txt', names = ['Exam 1', 'Exam 2', 'Admitted'])

fileData.insert(0, 'Ones',1)

print(fileData.head())


m = fileData.shape[0]
features = fileData.shape[1]

data = np.loadtxt('ex2data1.txt', delimiter=',')
print('Dimensions: ',data.shape)
print(data[1:6,:])

X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]
y = np.c_[data[:,2]]

print('SHAPES X : ', X.shape)
print('SHAPES Y : ', y.shape)


## plotting the graph
# defining positives and negativs first
positive = fileData[fileData['Admitted'].isin([1])]
negative = fileData[fileData['Admitted'].isin([0])]

#print(positive)
#print(negative)

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')  
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')  
ax.legend()   ## what is this doing?
ax.set_xlabel('Exam 1 Score')  
ax.set_ylabel('Exam 2 Score')  

#defining the sigmoid function
def sigmoid(z):
    return(1 / (1 + np.exp(-z)))
    
def cost_fn(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    
    print('SHAPES CF : ')
    print(h.shape)
    print(X.shape)
    print(y.shape)
    
    J = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))
    
    print('J size : ',J.shape)
               
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])


def gradient(theta, X, y):
    
    print('SHAPES :')
    print(theta.shape)
    print(X.shape)
    print(y.shape)
    
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    
    grad =(1/m)*X.T.dot(h-y)

    return(grad.flatten())


initial_theta = np.zeros(X.shape[1])
cost = cost_fn(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost: \n', cost)
print('Grad: \n', grad)

res = minimize(cost_fn, initial_theta, args=(X,y), method=None, jac=gradient, options={'maxiter':400})
print(res)

def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))
    
print(sigmoid(np.array([1, 45, 85]).dot(res.x.T)))

p = predict(res.x, X) 
print('P = ',p)
print('Y = ', Y.ravel())
print('Train accuracy {}%'.format(100*sum(p == Y.ravel())/p.size))
        
     
