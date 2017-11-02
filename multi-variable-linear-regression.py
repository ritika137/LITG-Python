# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 03:34:46 2017

@author: RITIKA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### GENERALISE FOR N FEATURESSS

fileData = pd.read_csv('ex1data2.txt', names = ['Size', 'Bedrooms', 'Price'])

print(fileData.head())

fileData = (fileData - fileData.mean())/fileData.std()

print(fileData.head())

n = 2;

X_data = pd.DataFrame(fileData.Size);

#mean_size = X_data['Size'].mean()

### WHY IS THIS GIVING WRONG???
###std_dev = X_data['Size'].std()
###print('STD DEV 1')
###print(std_dev)



#a = np.array(X_data)
#std_dev = a.std()
#print(std_dev)

## SEE WHY PLACIMG ONES IN THE FIRST COL RATHER THAN LAST MAKES A DIFFERENCE??

X_data['Bedrooms'] = pd.DataFrame(fileData.Bedrooms);

X_data.insert(0, 'Ones', 1)

mean_bedrooms = X_data['Bedrooms'].mean()

print(X_data)
Y_data = pd.DataFrame(fileData.Price)

data_size = len(X_data)

plt.figure(figsize=(10,8))
plt.plot(X_data, Y_data, 'kx')
plt.xlabel('Size and bedrooms')
plt.ylabel('Price of the house')

iterations = 700;
alpha = 0.01


# converting to numpy arrays for easier execution
X = np.array(X_data, dtype=np.float64)
Y = np.array(Y_data, dtype=np.float64).flatten()
theta = np.array([0, 0, 0]) 

#normalization of the sizes of the house...

for i in range(data_size):
   X[i][0] = (X[i][0]) - mean_size
   X[i][0] = X[i][0]/std_dev
   
plt.figure(figsize=(10,8))
plt.plot(X, Y, 'kx')
plt.xlabel('Size and bedrooms')
plt.ylabel('Price of the house')

def cost_fn(X, Y, theta):
    J = np.sum(((X.dot(theta) - Y )**2))/(2*data_size)
    return J


def gradient_descent(X, Y, theta):
    
    cost_hist = [0]*iterations
    for iter in range(iterations):
        hypothesis_fn = X.dot(theta)
        loss = hypothesis_fn -Y
        gd = X.T.dot(loss)/(data_size)
        theta = theta - alpha*gd
        cost = cost_fn(X, Y, theta)
        #print(cost)
        cost_hist[iter] = cost
    
    return theta, cost_hist

(t, c) = gradient_descent(X,Y,theta)

print(t)
print('COST:: ')
print(cost_fn(X, Y, t))

## Prediction
##print(np.array([3.5, 1]).dot(t))
##print(np.array([7, 1]).dot(t))


fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(np.arange(iterations), c, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')  



### HOW TO DRAW THE GRAPH TO DRAINING DATAAA????
plt.figure(figsize=(10,10))
plt.plot(X, Y, 'kx')
#plt.plot(best_fit_x, best_fit_y, '-')
#plt.axis([0,25,-5,25])
#plt.xlabel('Population of City in 10,000s')
#plt.ylabel('Profit in $10,000s')
#plt.title('Profit vs. Population with Linear Regression Line')
    
    

