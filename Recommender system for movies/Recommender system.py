# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:17:54 2017

@author: RITIKA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy.optimize import scipy
import scipy.io as sio #Used to load the OCTAVE *.mat files
import scipy.optimize as optimize#Use for fmincg


#LOADING DATA
fileData = sio.loadmat('ex8_movies.mat')
movieParams = sio.loadmat('ex8_movieParams.mat')
#movieId = pd.read_csv('movie_ids.txt', delimiter = '\n') #not working becuase of encoding!
movieId = pd.read_csv('movie_ids.txt', sep='\n', encoding='latin-1')


#val = "1 Toy Story (1995)"

#np.insert(movieIdtoName, 0, val)

# Reading to np arrays
movieIdtoName = np.array(movieId)
Y = np.array(fileData.get('Y'))
R = np.array(fileData.get('R'))

#X = np.array(movieParams.get('X'))
#Theta = np.array(movieParams.get('Theta'))

num_users = np.array(movieParams.get('num_users'))
num_movies = np.array(movieParams.get('num_movies'))
num_features = np.array(movieParams.get('num_features'))

num_movies, num_users = Y.shape



Y = np.multiply(Y, R)
print(num_features)



#print(fileData)
#print('masked Y : ', masked_ratings)
#print('movie : ', movieParams)
#print('MEAN : ', mean_rating,' ', mean_rating.shape)


def flattenParams(X, Theta):
    return np.concatenate((X.flatten(), Theta.flatten()))

'''def unflattenParams(flattenX_Theta, n_movies, n_users, n_features):
    assert flattenX_Theta.shape[0] == int(n_movies*n_features + n_users*n_features)
    reX = flattenX_Theta[:int(n_movies*n_features)].reshape(n_movies, n_features)
    reTheta = flattenX_Theta[int(n_movies*n_features):].reshape(n_users, n_features)
    return reX, reTheta'''
def unflattenParams(flattened_XandTheta, mynm, mynu, mynf):
    #assert flattened_XandTheta.shape[0] == int(num_movies*num_features+num_users*num_features)
    
    reX = flattened_XandTheta[:int(mynm*mynf)].reshape((mynm,mynf))
    reTheta = flattened_XandTheta[int(mynm*mynf):].reshape((mynu,mynf))
    
    return reX, reTheta




###########COST FUNCTION ############

def cost_function(myparams, Y, R, n_users, n_movies, n_features, Lambda=0.):
    X,Theta = unflattenParams(myparams,n_movies, n_users, n_features)
    # X,Y, Theta are elemenent wise multiplied matrix where all will be 0 when R(i,j) = 0   
    c = X.dot(Theta.T)
    #print('c : ', c.shape)
    
    co = np.multiply(R,c); # Taking only those entries where R(i,j)=1
    
    cost = 0.5*(np.sum(np.square(co-Y)))
    
    #regularization terms
    regX = (Lambda/2.)*(np.sum(np.square(X)))
    regTheta = (Lambda/2.)*(np.sum(np.square(Theta)))
    
    cost += (regX + regTheta)
    #print('COST : ', cost)
    return cost

def gradient_vectorized(myparams, Y, R, n_users, n_movies, n_features, Lambda=0.):
    
    #computing cost
    
    X,Theta = unflattenParams(myparams, n_movies, n_users, n_features)
    
    regX = Lambda*X
    regTheta = (Lambda)*(Theta)    
    
    c = X.dot(Theta.T)
    c = np.multiply(c,R)
    c-=Y
    
    gradX = c.dot(Theta) + regX
    #print('gradX : ', gradX.shape)
    
    gradTheta = c.T.dot(X) + regTheta
    #print('gradTheta : ', gradTheta.shape)    
    #gradX = np.multiply(R,gradX) + regX # Taking only those entries where R(i,j)=1
    #gradientX = np.sum(gradX)    
    #gradTheta = np.multiply(R,gradTheta) + regTheta # Taking only those entries where R(i,j)=1
    #gradientTheta = np.sum(gradTheta)    
    return flattenParams(gradX, gradTheta)
'''
#checking gradient computation:
def checkGradient(myparams, myY, myR, mynu, mynm, mynf, mylambda = 0.):
    
    print ('Numerical Gradient \t cofiGrad \t\t Difference')
    
    # Compute a numerical gradient with an epsilon perturbation vector
    myeps = 0.0001
    nparams = len(myparams)
    epsvec = np.zeros(nparams)
    # These are my implemented gradient solutions
    mygrads = gradient_vectorized(myparams,myY,myR,mynu,mynm,mynf,mylambda)

    # Choose 10 random elements of my combined (X, Theta) param vector
    # and compute the numerical gradient for each... print to screen
    # the numerical gradient next to the my cofiGradient to inspect
    
    for i in range(10):
        idx = np.random.randint(0,nparams)
        epsvec[idx] = myeps
        loss1 = cost_function(myparams-epsvec,myY,myR,mynu,mynm,mynf,mylambda)
        loss2 = cost_function(myparams+epsvec,myY,myR,mynu,mynm,mynf,mylambda)
        mygrad = (loss2 - loss1) / (2*myeps)
        epsvec[idx] = 0
        print ('%0.15f \t %0.15f \t %0.15f' %(mygrad, mygrads[idx],mygrad - mygrads[idx]))

# For now, reduce the data set size so that this runs faster
nu = 4
nm = 5
nf = 3
X = X[:nm,:nf]
Theta = Theta[:nu,:nf]
Y = Y[:nm,:nu]
R = R[:nm,:nu]

print( "Checking gradient with lambda = 0...")
checkGradient(flattenParams(X,Theta),Y,R,nu,nm,nf)
print ("\nChecking gradient with lambda = 1.5...")
checkGradient(flattenParams(X,Theta),Y,R,nu,nm,nf,mylambda = 1.5)

 WHAT IS HAPPENING HEREEE????
def gradient(X,Y,R,Theta,num_movies,num_users, num_features, Lambda):
    
    J = 0
    Xgrad = zeros(size(X))
    Thetagrad = zeros(size(Theta))
    J =  cost_function(X, Y, R, Theta, number_of_movies, number_of_users, num_features, Lambda)
    
    for i in range(0,num_movies):
        idx =  np.where(R[i,:]==1) # indices of a the terms where R(i,j) = 1
        Thetatemp = Theta[idx, :] #set of users which have rated the ith movie
        Ytemp = Y[i, idx] #set of users which have rated the ith movie
        regX = (Lambda)*(np.sum(X[i,:])
        #z = X[i, :]
        c = X[i, :]*(Thetatemp.T)
        Xgrad[i, :] = c.dot(Thetatemp) + regX
    print('Xgrad : ', Xgrad.shape)
    
    for j in range(0,num_users):
        idx = np.where(R[:,j]==1)
        Xtemp = X[idx,:]
        Ytemp = [idx,j]
        regTheta = (Lambda)*(np.sum(Theta[j,:])) 
        Thetagrad[j,:] = ((Xtemp.dot(Thetat[j,:].T)-Ytemp)).dot(Xtemp) + regTheta
        
    print('gradTheta : ', Thetagrad.shape)
    
    return gradientX, gradientTheta
'''

# defining my ratings!!

my_ratings = np.zeros((1682, 1))

my_ratings[0]   = 4
my_ratings[97]  = 2
my_ratings[6]   = 3
my_ratings[11]  = 5
my_ratings[53]  = 4
my_ratings[63]  = 5
my_ratings[65]  = 3
my_ratings[68]  = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

# So, this file has the list of movies and their respective index in the Y vector
# Let's make a list of strings to reference later
movies = []
with open('movie_ids.txt') as f:
    for line in f:
        movies.append(' '.join(line.strip('\n').split(' ')[1:]))

#print(movies)
        

#adding new data to train on the matrix Y and what all have we rated into the matrix R
add_row = my_ratings>0
Y = np.hstack((Y, my_ratings))
R = np.hstack((R, add_row))

num_movies, num_users = Y.shape


#### THIS MEAN COMPUTATION IS WRONG ONLY!!! HENCE WRONG ANSWER!!!
def mean_normalisation(Y, R):
    #Masking matrix. Another way id to do element wise multiplication of R and Y, And the 0 entries means no rating!
    #masked_ratings = np.ma.masked_array(Y, mask=R==0)

    # After masking now calculate mean
    #mean_rating = masked_ratings.mean(axis=1) ## mean rating/ average rating of each of the movie!
    mean_rating = np.sum(Y,axis=1)/np.sum(R,axis=1)
    mean_rating = mean_rating.reshape((Y.shape[0],1))
    
    print('SHAPE : ',mean_rating.shape)
    Ynorm = Y - mean_rating
    return Ynorm, mean_rating

#Ynorm, Ymean = mean_normalisation(Y,R)
Ymean = np.zeros((num_movies, 1))  
Ynorm = np.zeros((num_movies, num_users))

for i in range(num_movies):  
    idx = np.where(R[i,:] == 1)[0]
    Ymean[i] = Y[i,idx].mean()
    Ynorm[i,idx] = Y[i,idx] - Ymean[i]


num_features = 10

'''doing this gives error to minimize whyyy?? 
Warning: Desired error not necessarily achieved due to precision loss. in fmincg why?'''
#Y = Ynorm; '

X = np.random.rand(num_movies, num_features)
Theta = np.random.rand(num_users, num_features)

#flatten these parameters as the advanced optimization needs it to be flattened
myflat = flattenParams(X, Theta)
myLambda = 10.0


# Training the actual model with fmin_cg
result = optimize.fmin_cg(cost_function, x0=myflat, fprime=gradient_vectorized, args=(Ynorm,R,num_users,num_movies,
         num_features,myLambda),maxiter=50,disp=True,full_output=True)

# Reshape the trained output into sensible "X" and "Theta" matrices
resX, resTheta = unflattenParams(result[0], num_movies, num_users, num_features)

prediction_matrix = resX.dot(resTheta.T)

print('SHAPE PRED : ',prediction_matrix.shape)

# last user's predictions (since I put my predictions at the end of the Y matrix, not the front)
# Add back in the mean movie ratings
my_predictions = prediction_matrix[:,-1] + Ymean.flatten()

#print(Ymean.flatten())

# Sort my predictions from highest to lowest
pred_idxs_sorted = np.argsort(my_predictions)
pred_idxs_sorted[:] = pred_idxs_sorted[::-1]

print ("Top recommendations for you:")
for i in range(10):
    print ('Predicting rating %0.1f for movie %s.' %(my_predictions[pred_idxs_sorted[i]],movies[pred_idxs_sorted[i]]))
    
print ("\nOriginal ratings provided:")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print ('Rated %d for movie %s.' % (my_ratings[i],movies[i]))
        
        
''' VARIATION!!! CALCULATE MINIMUM REVIEWS FOR EACH MOVIE AND MODIFY RESULTS ACCORSINGLY!!! '''






