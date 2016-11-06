# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 09:52:44 2016

@author: liam

Shows a basic model that creates an embedding layer from integer (categorical)
inputs in Theano. Then below, the same thing in Keras!
"""

import theano
import theano.tensor as T
import numpy as np
        
def get_activation(inp, act_fn, name):
    if act_fn == 'tanh':
        return T.tanh(inp)
    elif act_fn == 'relu':
        return T.nnet.relu(inp)
    elif act_fn == 'sigmoid':
        return T.nnet.sigmoid(inp)
    else:
        print 'Note: no valid activation specified for ' + name
        return inp

# Function used to get the theano tensor from the class if class was passed to
# layer instead of raw tensor
def get_source(source):
    if 'Layer' in source.__class__.__name__:
        return source.output
    return source
    
# Function to get Glorot-initialised W shared matrix
def get_weights(in_dim, out_dim, name):
    W_val = np.asarray(\
        np.random.uniform(low=-np.sqrt(6. / (in_dim + out_dim)), 
                          high=np.sqrt(6. / (in_dim + out_dim)),
                          size=(in_dim, out_dim)), dtype=theano.config.floatX)
    return theano.shared(value=W_val, name=name, borrow=True)

# Function to get bias shared variable
def get_bias(d, name):
    b_values = np.zeros((d,), dtype=theano.config.floatX)
    b = theano.shared(value=b_values, name=name, borrow=True)
    return b

# Function to extract all the params from a list of layers
def get_all_params(layers):
    out = []
    for l in layers:
        for p in l.params:
            out.append(p)
    return out

class ConvLayer(object):

    def __init__(self, source, filter_shape, image_shape, stride,
                 act_fn, border_mode='full', name='conv'):
        """
        Create a convolutional layer
        This is adapted from the deeplearning.net Theano tutorial
        
        :source: previous layer or tensor

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        """

        assert image_shape[1] == filter_shape[1]
                             
        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.stride = stride
        self.border_mode = border_mode
        self.name = name
        self.act_fn = act_fn
        
        self.parent = source
        self.source = get_source(source)

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width"
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                np.random.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True,
            name=name + '_W'
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name=name + '_b')
        # convolve input feature maps with filters
        conv_out = T.nnet.conv2d(
            input=self.source,
            filters=self.W,
            filter_shape=self.filter_shape,
            input_shape=self.image_shape,
            border_mode=self.border_mode,
            subsample=self.stride
        )
        
        # Calc output
        self.output_pre_activ = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        # Activate it
        self.output = get_activation(self.output_pre_activ,
                                     act_fn=self.act_fn,
                                     name=self.name)

        self.params = [self.W, self.b]

class HiddenLayer():
    def __init__(self, source, input_size, hidden_size, name, act_fn):
        self.parent = source
        self.source = get_source(source)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.name = name
        self.act_fn = act_fn
        # Get weights and bias
        self.W = get_weights(self.input_size, self.hidden_size, 'W_' + name)
        self.b = get_bias(self.hidden_size, 'b_' + name)
        # Calc output
        self.output_pre_activ = T.dot(self.source, self.W) + \
                                self.b.dimshuffle('x', 0)
        # Activate it
        self.output = get_activation(self.output_pre_activ,
                                     act_fn=self.act_fn,
                                     name=self.name)
        self.params = [self.W, self.b]