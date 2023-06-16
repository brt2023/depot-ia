#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 09:58:18 2023

@author: barret
"""

#import artificial_network as an

#lim = 10
#h = 100
#w1 = np.linspace(-lim,lim,h)
#w2 = w1
#w11,w22 = np.meshgrid(w1,w2)
#
#
#w_final = np.concatenate([w11.ravel() , w22.ravel()]).T
#
#w11_t = np.concatenate([np.ravel(w11),np.ravel(w22)])
#w11_r = np.ravel(w11)
#
#w1122 = np.dot(w11,w22)
#plt.contour(w11,w22,w1122)


import plotly.graph_objects as go
#fig = (go.figure(data=[go.Surface(z=1, x=w11, y = w22, opacity=1)]))
#fig.show()

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import artificial_network as an
X,y = make_blobs(n_samples=100,n_features=2,centers=2,random_state=0)

X[:,1] = X[:,1]*10
y = y.reshape(y.shape[0], 1)
an.plt.scatter(X[:,0],X[:,1],c='y',cmap='summer')
an.plt.show()
history, b = an.artificial_network2(X, y, learning_rate=0.1, n_iter=1000)



