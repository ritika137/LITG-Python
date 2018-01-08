# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 01:30:26 2017

@author: RITIKA
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy.optimize import scipy
import scipy.io as sio #Used to load the OCTAVE *.mat files


#LOADING DATA
fileData = sio.loadmat('ex3data1.mat')
weights = sio.loadmat('ex3weights')

X = np.array(fileData.get('X'))
Y = np.array(fileData.get('y'))

Theta1 = np.array(weights.get('Theta1'))
Theta2 = np.array(weights.get('Theta2'))

m , n = X.shape
z = np.ones((m,1))

#print('N : ', n)
#X = np.append(X, z, axis = 1)



print('X : ', X[0])

input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;   


def sigmoid(x):
    return 1/ ( 1+ np.exp(-x))

    
def feedForward(ThetaWeights, activationLayer):
    
    #print('AAA : ',activationLayer)
    z = activationLayer.dot(ThetaWeights.T)
    act = sigmoid(z)
    return act

def output_layer(Theta1, Theta2, X):
    
    #add column
    output = np.zeros(X.shape[0])
    a1 = np.insert(X, 0, 1, axis=1)
    
    act2 = feedForward(Theta1, a1) # first hidden layer
    print('ACT 1 :', act2.shape)
    
    a2 = np.insert(act2, 0, 1, axis=1)
    a3 = feedForward(Theta2, a2) # output layer
    print('ACT 2 :', a3)
    output = np.argmax(a3, axis=1)+1 ###WHY IS +1 required here?? 0-based indexing in python!!!!!!
    return output
    
y_pred = output_layer(Theta1, Theta2, X)  
y_pred = y_pred.reshape((-1,1))
print(y_pred)
print(Y)
correct = [1 if (a == b) else 0 for (a, b) in zip(y_pred,Y)]  
# calculating the number of times correct is one vs over all for the accuracy
accuracy = (sum(map(int, correct)) / float(len(correct)))  
print ('accuracy = {0}%'.format(accuracy * 100))
    