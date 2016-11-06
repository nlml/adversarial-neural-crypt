# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 09:52:44 2016

@author: liam schoneveld

Implementation of model described in 'Learning to Protect Communications with 
Adversarial Neural Cryptography' (Martín Abadi & David G. Andersen, 2016, 
https://arxiv.org/abs/1610.06918)
"""

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from layers import ConvLayer, HiddenLayer, get_all_params
from lasagne.updates import adam

import cPickle
(results_bob, results_eve) = cPickle.load(open('run' + str(1) + '.dump', 'rb'))
# Plot the results
n = 2002
plt.plot([np.min(results_bob[i:i+n]) for i in np.arange(0, len(results_bob), n)])
plt.plot([np.min(results_eve[i:i+n]) for i in np.arange(0, len(results_eve), n)])
plt.legend(['bob', 'eve'])
plt.show()
#%%
# Plot the results
plt.plot([i for i in results_bob])
plt.plot([i for i in results_eve])
plt.legend(['bob', 'eve'])
plt.show()