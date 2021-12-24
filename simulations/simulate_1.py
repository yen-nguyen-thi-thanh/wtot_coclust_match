#!/usr/bin/env python
# coding: utf-8

#simulate data in the first simulation study.


import numpy as np
from numpy.random import multivariate_normal as mvn
from numpy import random as rd



def sample_data(mus, proportions, d = 3, N_x = 200, N_y = 200, 
                Nx_noise = 20, Ny_noise= 20,  var_cl = 0.1, var_noise = 30):
    
  

    #the variance = var_cl*identity(d)
    cov_cl = var_cl*np.eye(d)


    proportions_1 = proportions[0]
    cl_1 = len(proportions_1)
    proportions_2 = proportions[1] 
    cl_2 = len(proportions_2)

    components_ids_x = np.random.choice(cl_1, N_x, p = proportions_1)
    components_ids_y = np.random.choice(cl_2, N_y, p = proportions_2)

        
        
    X = []
    labels_x  = []

    for k in range(cl_1):
        N_k = np.sum(components_ids_x ==  k)
        X_k = mvn(mean = mus[k], cov = cov_cl, size = N_k)
        X.append(X_k)
        labels_x.append(k*np.ones(N_k))
    X = np.vstack(X) 
    labels_x = np.hstack(labels_x).astype(int)
    
    
    ### generate the second data set
    Y = []
    labels_y  = []

    for k in range(cl_2):
        N_k = np.sum(components_ids_y ==  k)
        Y_k = mvn(mean = -mus[k], cov = cov_cl, size = N_k)
        Y.append(Y_k)
        labels_y.append(k*np.ones(N_k))
        
    Y = np.vstack(Y) 
    labels_y = np.hstack(labels_y).astype(int)
    

    idx1 = np.arange(X.shape[0])
    rd.shuffle(idx1)
    X = X[idx1]
    labels_x = labels_x[idx1] 
    idx2 = np.arange(Y.shape[0])
    rd.shuffle(idx2)
    Y = Y[idx2]
    labels_y = labels_y[idx2]         
    return  X, Y, labels_x, labels_y


#example: 
mus = 
