# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# for single feature: Linear regression:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fileData = pd.read_csv('ex1data1.txt', names = ['population', 'profit'])

print(fileData.head())

## Split population and profit into X and y
X_df = pd.DataFrame(fileData.population)  ### how is this working with data??? -_- shouldn't this be fileData??
y_df = pd.DataFrame(fileData.profit)
X_df.insert(0, 'Ones',1)

print(X_df)
## Length, or number of observations, in our data
m = len(X_df)

plt.figure(figsize=(10,8))  # size of graph
plt.plot(X_df, y_df, 'kx')  # What is kx here??
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')


plt.figure(figsize=(10,8))
plt.plot(X_df, y_df, 'k.')
plt.plot([5, 22], [6,6], '-')  # how are different colors coming?
plt.plot([5, 22], [0,20], '-')
plt.plot([5, 15], [-5,25], '-')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')


### Hpow are we deicidng this??????
iterations = 1500
alpha = 0.01

## Add a columns of 1s as intercept to X (col appended in the end)


print(X_df)

## Transform to Numpy arrays for easier matrix math and start theta at 0
X = np.array(X_df)
y = np.array(y_df).flatten()
theta = np.array([0, 0])  # Here theta is necessary 0 or can be any value?????

print(X)
print(y)

def cost_function(X, y, theta):
    """
    cost_function(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """
    ## number of training examples
    m = len(y) 
    
    ## Calculate the cost with the given parameters
    ## X.dot(theta) is equal to matrix multiplicaton of X*theta
    J = np.sum((X.dot(theta)-y)**2)/2/m
    
    return J


cost_function(X, y, theta)

## does .dot take care of dimensions ewhile multiplying 2 vectors/matrices??


def gradient_descent(X, y, theta, alpha, iterations):
    """
    gradient_descent Performs gradient descent to learn theta
    theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha
    """
    cost_history = [0] * iterations
    
    for iteration in range(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis-y
        gradient = X.T.dot(loss)/m 
        theta = theta - alpha*gradient
        cost = cost_function(X, y, theta)
        cost_history[iteration] = cost

    return theta, cost_history


(t, c) = gradient_descent(X,y,theta,alpha, iterations)

print(t)
print(cost_function(X, y, t))
## Prediction
print(np.array([1,3.5]).dot(t))
print(np.array([1, 7]).dot(t))


## Plotting the best fit line
best_fit_x = np.linspace(0, 25, 20)
best_fit_y = [t[0] + t[1]*xx for xx in best_fit_x]


plt.figure(figsize=(10,6))
plt.plot(X_df.population, y_df, '.')
plt.plot(best_fit_x, best_fit_y, '-')
plt.axis([0,25,-5,25])
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Profit vs. Population with Linear Regression Line')


