#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:10:14 2023

@author: barret
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import accuracy_score
import seaborn as se
from tqdm import tqdm



def initialisation(X):
    W = np.random.randn(X.shape[1],1)
    b = np.random.randn(1)
    return (W,b)

def model(X,W,b):
    Z = X.dot(W) + b
    A = 1/(1 + np.exp(-Z))
    return A

def log_loss(A,y):
    m = len(y)
    epsilon = 1.e-15
    return -1/m * (np.sum(y*np.log(A+epsilon) + (1-y)*np.log(1-A+epsilon)))

def gradients(A,X,y):
    m = len(y)
    dW = 1/m*(np.dot(X.T , A-y))
    db = 1/m*(np.sum(A - y))
    return (dW,db)

def update(W,b,dW,db,learning_rate):
    W = W - learning_rate*dW
    b = b - learning_rate*b
    return (W,b)

def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5



# PROGRAMME GENERAL
def artificial_network(X_train, y_train, X_test, y_test, learning_rate=0.01, n_iter=1000):
    # initialisation W,b
    W,b = initialisation(X_train)
    
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    

    # boucle d'apprentissage
    for i in tqdm(range(n_iter)):
        
        # activations
        A = model(X_train, W, b)


        if i%10 == 0:
            # TRAIN
            # calcul du cout
            train_loss.append(log_loss(A, y_train))            
            # calcul de l'accuracy
            y_pred = predict(X_train, W, b)
            train_acc.append(accuracy_score(y_train, y_pred))
            
            # TEST
            # calcul du cout
            A_test = model(X_test, W, b)
            test_loss.append(log_loss(A_test, y_test))            
            # calcul de l'accuracy
            y_pred = predict(X_test, W, b)
            test_acc.append(accuracy_score(y_test, y_pred))

            
        # mise a jour
        dW,db = gradients(A, X_train, y_train)
        W,b = update(W, b, dW, db, learning_rate)
       
        
        
    # Affichage
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss,label='train_loss')
    plt.plot(test_loss,label='test_loss')
    plt.legend()
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(train_acc,label='train_acc')
    plt.plot(test_acc,label='test_acc')
    plt.legend()
    plt.grid()    
    plt.show()
    
    return (W,b)











def artificial_network2(X,y,learning_rate=0.01,n_iter=100):
    j = 0
    W,b = initialisation(X)
    W[0],W[1] = -7.5,-7.5
    Loss = []
    N = n_iter//10
    history = np.zeros((N,5))
    for i in range(n_iter):
        A = model(X,W,b)
        Loss.append(log_loss(A,y))
        dW,db = gradients(A,X,y)
        W,b = update(W,b,dW,db,learning_rate)
        
        if i%10 == 0:
            history[j,0]=W[0]
            history[j,1]=W[1]
            history[j,2]=b
            history[j,3]=i
            history[j,4]=log_loss(y,A)
            j += 1
       
    plt.plot(Loss)
    plt.show()
    return history,b
        
    



