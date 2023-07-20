#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:43:46 2023

@author: barret
"""
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score #, log_loss





def initialisation(n0,n1,n2):
    
    W1 = np.random.randn(n1,n0)
    b1 = np.random.rand(n1,1)
    W2 = np.random.randn(n2,n1)
    b2 = np.random.rand(n2,1)
    
    parametres = {
            "W1":W1,
            "b1":b1,
            "W2":W2,
            "b2":b2
            } 
    
    return parametres




def forward_propagation(X, parametres):
    
    W1 = parametres["W1"]
    b1 = parametres["b1"]
    W2 = parametres["W2"]
    b2 = parametres["b2"]
    
    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    
    activations = {
            "A1":A1,
            "A2":A2            
            }
    
    return activations





def back_propagation(X, y, parametres, activations):
    
    A1 = activations["A1"]
    A2 = activations["A2"]
    W2 = parametres["W2"]
    
    m = y.shape[1]
    
    dZ2 = A2 - y 
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2 , axis = 1, keepdims = True) #keepdims permet de garder un tableau a 2 dimensions
    
    dZ1 = np.dot(W2.T , dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1 , axis = 1, keepdims = True)
    
    gradients = {
            "dW1":dW1,
            "db1":db1,
            "dW2":dW2,
            "db2":db2
            }
    
    return gradients




def update(gradients, parametres, learning_rate):
    
    W1 = parametres["W1"]
    b1 = parametres["b1"]
    W2 = parametres["W2"]
    b2 = parametres["b2"]
    
    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parametres = {
            "W1":W1,
            "b1":b1,
            "W2":W2,
            "b2":b2
            }
    
    return parametres





def predict(X , parametres):
    activations = forward_propagation(X, parametres)
    A2 = activations['A2']
    return A2 >= 0.5




def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def log_loss(y, A):
    m = len(y)
    epsilon = 1.e-15
    return -1 / m * (np.sum(y*np.log(A+epsilon) + (1-y)*np.log(1-A+epsilon)))






def neural_network(X, y, n1=32, learning_rate=0.1, n_iter=1000):
    
    # initialisation parametres
    n0 = X.shape[0]
    n2 = y.shape[0]
    np.random.seed(0)
    parametres = initialisation(n0, n1, n2) 

    train_loss = []
    train_acc = []
    history = []
    
    #gradient descent
    for i in tqdm(range(n_iter)):
        
        activations = forward_propagation(X, parametres)
        A2 = activations['A2']
        
        #  plot courbe d'apprentissage
        train_loss.append(log_loss(y.flatten(),A2.flatten()))
        y_pred = predict(X, parametres)
        train_acc.append(accuracy_score(y.flatten(), y_pred.flatten()))

        history.append([parametres.copy(), train_loss, train_acc, i])
        
        # mise a jour
        gradients = back_propagation(X, y, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)
           
            
    plt.figure(figsize=(7,2))
    
    plt.subplot(1,2,1)
    plt.plot(train_loss,label="train_loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_acc,label="train_acc")
    plt.legend()
    
    plt.grid()    
    plt.show()
    
    return parametres