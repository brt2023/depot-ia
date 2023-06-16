#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:14:22 2023

@author: barret
"""

import h5py
from utilities import *
import utilities as ut
import artificial_network as an

X_train, y_train, X_test, y_test = ut.load_data()



print(an.np.unique(y_train,return_counts=True))


# AFFICHAGE des PHOTOS (9)
an.plt.figure(figsize=(16,8))
for i in range(1,10):
    an.plt.subplot(4,5,i)
    an.plt.imshow(X_train[i],cmap='gray')
    an.plt.title(y_train[i])
    an.plt.tight_layout()
an.plt.show()



X_train_reshape = X_train.reshape(X_train.shape[0] , -1)/X_train.max()
X_test_reshape = X_test.reshape(X_test.shape[0] , -1)/X_train.max() #eh oui


#W,b = an.artificial_network(X_train_reshape, y_train)
W,b = an.artificial_network(X_train_reshape, y_train, X_test_reshape, y_test)








