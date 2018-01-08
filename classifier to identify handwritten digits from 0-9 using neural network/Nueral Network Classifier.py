# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:24:14 2018

@author: RITIKA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy.optimize import scipy
import scipy.io as sio #Used to load the OCTAVE *.mat files


#LOADING DATA
fileData = sio.loadmat('ex4data1.mat')
weights = sio.loadmat('ex4weights')

X = np.array(fileData.get('X'))
Y = np.array(fileData.get('y'))

Theta1 = np.array(weights.get('Theta1'))
Theta2 = np.array(weights.get('Theta2'))

m , n = X.shape
z = np.ones((m,1))

print('N : ', Y)

input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;   



def flattenParams(Theta1, Theta2):
    return np.concatenate((Theta1.flatten(), Theta2.flatten()))

def unflattenParams(flattened_T1andT2, num_labels, hidden_layer, input_layer):
    #assert flattened_XandTheta.shape[0] == int(num_movies*num_features+num_users*num_features)
    reTheta1 = flattened_T1andT2[:int(input_layer*hidden_layer)].reshape((hidden_layer,input_layer))
    reTheta2 = flattened_T1andT2[int(input_layer*hidden_layer):].reshape((hidden_layer,num_labels))
    
    return reTheta1, reTheta2

nn_params = flattenParams(Theta1, Theta2)

def sigmoid(x):
    return 1/ ( 1+ np.exp(-x))

    
def feedForward(ThetaWeights, activationLayer):
    
    z = activationLayer.dot(ThetaWeights.T)
    act = sigmoid(z)
    return act

def output_layer(Theta1, Theta2, X):
    
    #add column
    output = np.zeros(X.shape[0])
    a1 = np.insert(X, 0, 1, axis=1)
    act2 = feedForward(Theta1, a1) # first hidden layer
    a2 = np.insert(act2, 0, 1, axis=1)
    a3 = feedForward(Theta2, a2) # output layer
    print('A 3 ', a3.shape)
    output = np.argmax(a3, axis=1)+1 ###WHY IS +1 required here?? 0-based indexing in python!!!!!!
    return a3, output
    

def NN_costFunction(Y_mat ,a3, num_labels, hidden_layer, input_layer, Lambda, m):
    #Theta1, Theta2 = unflattenParams(Theta, num_labels, hidden_layer, input_layer)
    logh = np.log(a3)
    log1h = np.log(1-a3)
    
    J = (np.multiply(-Y_mat,logh)) - (np.multiply((1-Y_mat),log1h))
    #print('SHAPE J1' , J.shape)
    cost = (1./m)*np.sum(np.sum(J))
    print('COST : ', m)
    
    if np.isnan(cost):  ######IMP STEP ELSE ERROR!!!
        return np.inf
    
    #regTheta =  Theta.T.dot( Theta ) * Lambda / (2*m)
    #cost = (1./m)*cost + regTheta
    
    return cost











Lambda = 0

I = np.eye(num_labels)
logic_Y = np.array([(x-1) if x!=10 else 9 for x in Y])
Y_matrix = I[logic_Y , :]  ## wn't work as 0 represents 10
print(Y_matrix)


A3, y_pred = output_layer(Theta1, Theta2, X)  
y_pred = y_pred.reshape((-1,1))

costNN = NN_costFunction(Y_matrix ,A3, num_labels, hidden_layer_size, input_layer_size, Lambda, m)

print('COST NN : ', costNN)

#print(y_pred)
#print(Y)
correct = [1 if (a == b) else 0 for (a, b) in zip(y_pred,Y)]  
# calculating the number of times correct is one vs over all for the accuracy
accuracy = (sum(map(int, correct)) / float(len(correct)))  
print ('accuracy = {0}%'.format(accuracy * 100))