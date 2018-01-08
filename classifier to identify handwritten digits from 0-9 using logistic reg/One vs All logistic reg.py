# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 00:32:32 2017

@author: RITIKA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import optimize
import scipy.io as sio #Used to load the OCTAVE *.mat files


#LOADING DATA
fileData = sio.loadmat('ex3data1.mat')

'''There are 5000 training examples in ex3data1.mat, where each training
example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is
represented by a oating point number indicating the grayscale intensity at
that location. The 20 by 20 grid of pixels is unrolled into a 400-dimensional
vector. Each of these training examples becomes a single row in our data
matrix X. This gives us a 5000 by 400 matrix X where every row is a training
example for a handwritten digit image'''

X = np.array(fileData.get('X'))
Y = np.array(fileData.get('y'))



m,n = X.shape

z = np.ones((m,1))

#print('N : ', n)
X = np.append(X, z, axis = 1)
print(X.shape)

''' digit 0 is represented as 10'''

print(Y[1729])

visualizeX = X[1729, 0:400].reshape(20,20);

#print(visualizeX)
#plt.imshow(visualizeX)

def sigmoid(x):
    return 1/ ( 1+ np.exp(-x))

def hypothesis(X, theta):
    mat = X.dot(theta)
    mat = sigmoid(mat)
    return mat

def cost_function(t, X, Y, m, n, Lambda):
    ## unflatten theta as fmin_cg uses a flattened version. Convert to an appropriate sized matrix
    theta = t.reshape((-1, 1))
    h = hypothesis(X, theta)
    #print('SHAPE H' , h.shape)
    logh = np.log(h)
    log1h = np.log(1-h)
    
    J = (-Y.T.dot(logh)) - ((1-Y).T.dot(log1h))
    #print('SHAPE J1' , J.shape)
    r = J[0]
    
    if np.isnan(r):  ######IMP STEP ELSE ERROR!!!
        return np.inf
    
    regTheta =  theta.T.dot( theta ) * Lambda / (2*m)
    cost = (1./m)*J + regTheta
    
    return cost

def gradient(t, X, Y, m, n, Lambda):
    ## unflatten theta
    theta = t.reshape((-1, 1))
    h = hypothesis(X, theta)
    z = h - Y
    #print('SHAPE Z' , z.shape)
    grad = X.T.dot(z)
    #print('SHAPE GRADD' , grad.shape)
    regTheta = theta[1:]*(Lambda/m) #shape: (400,1)
    grad[1:] = grad[1:] + regTheta
    #print('SHAPE' , grad.shape)
    return grad.ravel()  ###IMP AS FMIN_CG TAKES 1D VECTOR VALUES

def optimize_theta(th,X,Y,m , n, myLambda):
    result = optimize.fmin_cg(cost_function, x0=th, fprime=gradient, args=(X, Y ,m ,n,myLambda),
                              maxiter=100,disp=True,full_output=True)
    return result[0], result[1]


def buildForAll(X, Y, num_labels, n, m, myLambda):
    
    th =  np.zeros((n+1, 1))
    THETA = np.zeros((num_labels, n+1))
    
    for i in range(num_labels):
        iclass = i if i else 10
        print ("Optimizing for handwritten number %d..." %i)
        logic_Y = np.array([1 if x == iclass else 0 for x in Y]).reshape((X.shape[0], 1)) # building y for ith class
        #print('SHAPE LOGIC Y' , logic_Y.shape)
        itheta, imincost = optimize_theta(th,X,logic_Y,m , n , myLambda)
        #print('itheta : SHAPE ' ,itheta.shape)
        THETA[i,:] = itheta # set for ith class!!!
    return THETA


def predict_all(X, all_theta):  
    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # compute sigmoid for all the rows. (hypothesis for all values of X and theta for all classes from 0-9)
    h = sigmoid(X * all_theta.T) #(5000x401 and 10x401)
    # result -> 5000x10 i.e. sigmoid value for all the 10 classes for each test case.

    # create array of the index with the maximum sigmoid value
    h_max = np.argmax(h, axis=1) # +1 not required here because of build_for_all fn, we are already adjusting the 0 thingy
    print('SHAPE H_MAX : ', h_max.shape)
    return h_max

num_labels = 10 ## number of classes that is 0-9
myLambda = 0.1
ThetaAns = buildForAll(X, Y, num_labels, n, m, myLambda) 


y_pred = predict_all(X, ThetaAns)  
print(y_pred)
print(Y)
# since 0 is represented as 10 map back to 0.
# zip(a, b) basically pairs all items in a and b arrays
correct = [1 if ((a == b) or (a==0 and b==10)) else 0 for (a, b) in zip(y_pred,Y)]  
# calculating the number of times correct is one vs over all for the accuracy
accuracy = (sum(map(int, correct)) / float(len(correct)))  
print ('accuracy = {0}%'.format(accuracy * 100))