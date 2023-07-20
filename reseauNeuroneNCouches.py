#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 17:39:26 2023

@author: barret
"""

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss


# 1. INIT
def initialisation(dimensions):
    parametres = {}
    
    C = len(dimensions) 
    
    for c in range(1, C):
        parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c-1])
        parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)
    
    return parametres



def forward_PROPAGATION(X, parametres):
    activations = {'A0':X}
    
    C = len(parametres)//2 #division entiere
    
    for c in range(1, C + 1):
        Z = parametres['W' + str(c)].dot(activations['A' + str(c-1)])
        activations['A' + str(c)] = 1/(1 + np.exp(-Z))
    
    return activations



def back_PROPAGATION(y, activations, parametres):
    m = y.shape[1]
    C = len(parametres) // 2
    
    dZ = activations['A'+str(C)]-y
    gradients = {}
    
    for c in reversed(range(1,C+1)):
        gradients['dW'+str(c)] = 1/m * np.dot(dZ, activations['A' + str(c-1)].T)
        gradients['db'+str(c)] = 1/m * np.sum(dZ,axis=1,keepdims=True)
        if c > 1:
            dZ = np.dot(parametres['W'+str(c)].T, dZ) * activations['A' + str(c-1)]*(1-activations['A'+str(c-1)])
    
    
    return gradients


def update(gradients, parametres, learning_rate):
    C = len(parametres) // 2
    
    for c in range(1,C+1):
        parametres['W'+str(c)] = parametres['W'+str(c)] - learning_rate*gradients['dW'+str(c)]
        parametres['b'+str(c)] = parametres['b'+str(c)] - learning_rate*gradients['db'+str(c)]
    
    return parametres




def predict(X , parametres):
    activations = forward_PROPAGATION(X, parametres)
    C = len(parametres) // 2
    AC = activations['A'+str(C)]
    return AC >= 0.5


def log_loss(y, A):
    m = len(y)
    epsilon = 1.e-15
    return -1 / m * (np.sum(y*np.log(A+epsilon) + (1-y)*np.log(1-A+epsilon)))









# PGM PRINCIPAL
def deep_neural_network(X,y,hidden_layers={32,32,32},learning_rate=0.1,n_iter=1000):
    np.random.seed(0)
    
    dimensions = list(hidden_layers)
    dimensions.insert(0,X.shape[0])
    dimensions.append(y.shape[0])
    
    # INIT.
    parametres = initialisation(dimensions)
    
    train_loss = []
    train_acc = []
    training_history = np.zeros((int(n_iter), 2))
    
    for i in tqdm(range(n_iter)):
        
        activations = forward_PROPAGATION(X, parametres)
        gradients = back_PROPAGATION(y, activations, parametres)
        parametres = update(gradients, parametres, learning_rate)
        
        if i%10 == 0:
            C = len(parametres) // 2            
            Af = activations['A' + str(C)]            
            train_loss.append(log_loss(y, activations['A'+str(C)]))
            training_history[i, 0] = (log_loss(y.flatten(), Af.flatten()))
            y_pred = predict(X, parametres)
            
            current_accuracy = accuracy_score(y.flatten(), y_pred.flatten())
            training_history[i, 1] = (accuracy_score(y.flatten(), y_pred.flatten()))
            train_acc.append(current_accuracy)
            
        
        # VISUALISATION
#        plt.figure(figsize=(7,2))
#    
#        plt.subplot(1,2,1)
#        plt.plot(train_loss,label="train_loss")
#        plt.legend()
#        plt.subplot(1,2,2)
#        plt.plot(train_acc,label="train_acc")
#        plt.legend()
#        
#        plt.grid()    
#        plt.show()
#    fig,ax = plt.subplots(nrows=1, ncols=3, figsize(18,4))
#    ax[0].plot(train_loss, label='train_loss')
#    ax[0].legend()
#    ax[1].plot(train_acc, label='train_acc')
#    ax[1].legend()
##    visualisation(X,y,parametres,ax)
#    plt.show()
            
    
    # Plot courbe d'apprentissage
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='train acc')
    plt.legend()
    plt.show()
    return parametres
    




# TESTS
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X = X.T
y = y.reshape((1, y.shape[0])) # ATTENTION !!!!
plt.scatter(X[0, :], X[1, :], cmap='summer')
plt.show()


#print('tests initialisation')
#    
#dimensions = [2,32,32,1]
#parametres = initialisation(dimensions)
#for key,val in parametres.items():
#    print(key,val.shape)
#    
#print('tests forward_PROPAGATION')
#    
#activations = forward_PROPAGATION(X, parametres)
#for key,val in activations.items():
#    print(key, val.shape)
#    
#print('tests back_PROPAGATION')  
#grad = back_PROPAGATION(y, activations, parametres) 
#for key,val in grad.items():
#    print(key,val.shape)
    
    
deep_neural_network(X,y,hidden_layers={32,32,32},learning_rate=0.1,n_iter=500)


 
    
    