# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:39:30 2017

@author: RITIKA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


fileData = pd.read_csv('ex2data1.txt', names = ['Exam 1', 'Exam 2', 'Admitted'])

fileData.insert(0, 'Ones',1)

print(fileData.head())


m = fileData.shape[0]
features = fileData.shape[1]

X = fileData.iloc[:,0:features-1]
Y = fileData.iloc[:,features-1:features]

X = np.matrix(X, dtype=np.float64)
Y = np.matrix(Y,dtype=np.float64).flatten()
theta = np.zeros((features-1, 1))
theta = np.matrix(theta)
print(theta)

#print(features)
#print(X)
#print(Y)
#print(m)


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
def sigmoid_array(x):                                        
   return 1 / (1 + np.exp(-x))

def hypothesis_fn(theta, x):
    temp = x.dot(theta)
    s = sigmoid_array(temp)
    #print(s)
    return s

s = np.array([1, 9, 10, 4, 0.6, 0])




#print(sigmoid_array(s))
## cost fn is different in Logistic regression!

## modify cist_fn according to gradient descent i.e. when Y=1 sum differenty etc!
def cost_fn(X, Y, theta):
    theta = np.matrix(theta)
    hypo = hypothesis_fn(theta, X)
    #print(Y.shape) bot taken transpose since Y's shape is already transposed
    a = -Y
    #print(a.shape)
    b = (1-Y)
    #print(b.shape)
    c = np.log(hypo)
    #print(c.shape)
    d = np.log(1-hypo)
    #print(d.shape)
    cost = a.dot(c) - b.dot(d)
    cost = np.sum(cost)/m
    return cost   

def sigmoid(z):  
    return 1 / (1 + np.exp(-z))


def gradient_descent(X, Y, theta, iterations, alpha):    
    
    gd = -14.00
    for iter in range(iterations):
        hypo_fn = hypothesis_fn(theta, X)
        #print(hypo_fn.shape)
        loss = hypo_fn - Y.T
        gd = (X.T.dot(loss))/m
        theta = theta - alpha*gd
        #ct = cost_fn(X, Y, theta)
        #print(ct)
        #cost_hist[iter] = ct
    
    return gd,theta

print(cost_fn(X,Y, theta))

#(t,c) = gradient_descent(X,Y,theta, 10, 0.01)

print('s')

(g, t) = gradient_descent(X,Y,theta, 1, 0.01)

print(g)


#print(cost_fn(X,Y, t))

num1 = np.arange(-10, 10, step=1)
num2 = np.arange(-10, 10, step=1)

fn = -0.00501746*num1 + 0.21526477*num2 + 0.13214539

fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(num1, fn, 'r')  



num1 = np.arange(-0.01, 0.01, step=0.0001)
num2 = np.arange(-0.01, 0.01, step=0.0001)

fn = -10 -656.44274057*num1 -662.21998088*num2


fig,ax = plt.subplots(figsize=(12,8))  
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')  
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')  
ax.legend()   ## what is this doing?
ax.plot(num1, fn, 'y')  
ax.set_xlabel('Exam 1 Score')  
ax.set_ylabel('Exam 2 Score')  

print('ANS')
test_theta = [-24, 0.2, 0.2]
test_theta = np.matrix(test_theta)
print(cost_fn(X, Y, test_theta.T))

(g, t) = gradient_descent(X,Y,test_theta.T, 2000, 0.01)

print(cost_fn(X, Y, g))
        
     