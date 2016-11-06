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

# Parameters
batch_size = 512
msg_len = 16
key_len = 16
comm_len = 16

# Set this flag to exclude convolutional layers from the networks
skip_conv = False

# Function to generate n random messages and keys
def gen_data(n=batch_size, msg_len=msg_len, key_len=key_len):
    return (np.random.randint(0, 2, size=(n, msg_len))*2-1).\
                astype(theano.config.floatX),\
           (np.random.randint(0, 2, size=(n, key_len))*2-1).\
                astype(theano.config.floatX)

# Function to assess a batch by eye (see what the errors look like)
def assess(pred_fn, n=batch_size, msg_len=msg_len, key_len=key_len):
    msg_in_val, key_val = gen_data(n, msg_len, key_len)
    return np.round(np.abs(msg_in_val[0:n] - \
           pred_fn(msg_in_val[0:n], key_val[0:n])), 0)

# Function to get the error over just one batch
def err_over_samples(err_fn, n=batch_size):
    msg_in_val, key_val = gen_data(n)
    return err_fn(msg_in_val[0:n], key_val[0:n])

class StandardConvSetup():
    '''
    Standard convolutional layers setup used by Alice, Bob and Eve.
    Input should be 4d tensor of shape (batch_size, 1, msg_len + key_len, 1)
    Output is 4d tensor of shape (batch_size, 1, msg_len, 1)
    '''
    def __init__(self, reshaped_input, name='unnamed'):
        
        self.name = name
        self.conv_layer1 = ConvLayer(reshaped_input,
                                     filter_shape=(2, 1, 4, 1), #num outs, num ins, size
                                     image_shape=(None, 1, None, 1),
                                     stride=(1,1),
                                     name=self.name + '_conv1',
                                     border_mode=(2,0),
                                     act_fn='relu')
        
        self.conv_layer2 = ConvLayer(self.conv_layer1, 
                                     filter_shape=(4, 2, 2, 1),
                                     image_shape=(None, 2, None, 1),
                                     stride=(2,1),
                                     name=self.name + '_conv2',
                                     border_mode=(0,0),
                                     act_fn='relu')
        
        self.conv_layer3 = ConvLayer(self.conv_layer2, 
                                     filter_shape=(4, 4, 1, 1),
                                     image_shape=(None, 4, None, 1),
                                     stride=(1,1),
                                     name=self.name + '_conv3',
                                     border_mode=(0,0),
                                     act_fn='relu')
        
        self.conv_layer4 = ConvLayer(self.conv_layer3, 
                                     filter_shape=(1, 4, 1, 1),
                                     image_shape=(None, 4, None, 1),
                                     stride=(1,1),
                                     name=self.name + '_conv4',
                                     border_mode=(0,0),
                                     act_fn='tanh')
        
        self.output = self.conv_layer4.output
        self.layers = [self.conv_layer1, self.conv_layer2, 
                       self.conv_layer3, self.conv_layer4]
        self.params = []
        for l in self.layers:
            self.params += l.params
            
# Tensor variables for the message and key
msg_in = T.matrix('msg_in')
key = T.matrix('key')

# Alice's input is the concatenation of the message and the key
alice_in = T.concatenate([msg_in, key], axis=1)

# Alice's hidden layer
alice_hid = HiddenLayer(alice_in,
                        input_size=msg_len + key_len,
                        hidden_size=msg_len + key_len,
                        name='alice_to_hid',
                        act_fn='relu')
if skip_conv:
    alice_conv = HiddenLayer(alice_hid,
                             input_size=msg_len + key_len,
                             hidden_size=msg_len,
                             name='alice_hid_to_comm',
                             act_fn='tanh')
    alice_comm = alice_conv.output
else:
    # Reshape the output of Alice's hidden layer for convolution
    alice_conv_in = alice_hid.output.reshape((batch_size, 1, msg_len + key_len, 1))
    # Alice's convolutional layers
    alice_conv = StandardConvSetup(alice_conv_in, 'alice')
    # Get the output communication
    alice_comm = alice_conv.output.reshape((batch_size, msg_len))

# Bob's input is the concatenation of Alice's communication and the key
bob_in = T.concatenate([alice_comm, key], axis=1)
# He decrypts using a hidden layer and a conv net as per Alice
bob_hid = HiddenLayer(bob_in, 
                      input_size=comm_len + key_len,
                      hidden_size=comm_len + key_len,
                      name='bob_to_hid',
                      act_fn='relu')
if skip_conv:
    bob_conv = HiddenLayer(bob_hid,
                           input_size=comm_len + key_len,
                           hidden_size=msg_len,
                           name='bob_hid_to_msg',
                           act_fn='tanh')
    bob_msg = bob_conv.output
else:
    bob_conv_in = bob_hid.output.reshape((batch_size, 1, comm_len + key_len, 1))
    bob_conv = StandardConvSetup(bob_conv_in, 'bob')
    bob_msg = bob_conv.output.reshape((batch_size, msg_len))

# Eve see's Alice's communication to Bob, but not the key
# She gets an extra hidden layer to try and learn to decrypt the message
eve_hid1 = HiddenLayer(alice_comm, 
                       input_size=comm_len,
                       hidden_size=comm_len + key_len,
                       name='eve_to_hid1',
                       act_fn='relu')
                          
eve_hid2 = HiddenLayer(eve_hid1, 
                       input_size=comm_len + key_len,
                       hidden_size=comm_len + key_len,
                       name='eve_to_hid2',
                       act_fn='relu')

if skip_conv:
    eve_conv = HiddenLayer(eve_hid2,
                           input_size=comm_len + key_len,
                           hidden_size=msg_len,
                           name='eve_hid_to_msg',
                           act_fn='tanh')
    eve_msg = eve_conv.output
else:
    eve_conv_in = eve_hid2.output.reshape((batch_size, 1, comm_len + key_len, 1))
    eve_conv = StandardConvSetup(eve_conv_in, 'eve')
    eve_msg = eve_conv.output.reshape((batch_size, msg_len))

# Eve's loss function is the L1 norm between true and recovered msg
decrypt_err_eve = T.mean(T.abs_(msg_in - eve_msg))

# Bob's loss function is the L1 norm between true and recovered
decrypt_err_bob = T.mean(T.abs_(msg_in - bob_msg))
# plus (N/2 - decrypt_err_eve) ** 2 / (N / 2) ** 2
# --> Bob wants Eve to do only as good as random guessing
loss_bob = decrypt_err_bob + (1. - decrypt_err_eve) ** 2.


# Get all the parameters for Bob and Alice, make updates, train and pred funcs
params   = {'bob' : get_all_params([bob_conv, bob_hid, 
                                    alice_conv, alice_hid])}
updates  = {'bob' : adam(loss_bob, params['bob'])}
err_fn   = {'bob' : theano.function(inputs=[msg_in, key],
                                    outputs=decrypt_err_bob)}
train_fn = {'bob' : theano.function(inputs=[msg_in, key],
                                    outputs=loss_bob,
                                    updates=updates['bob'])}
pred_fn  = {'bob' : theano.function(inputs=[msg_in, key], outputs=bob_msg)}

# Get all the parameters for Eve, make updates, train and pred funcs
params['eve']   = get_all_params([eve_hid1, eve_hid2, eve_conv])
updates['eve']  = adam(decrypt_err_eve, params['eve'])
err_fn['eve']   = theano.function(inputs=[msg_in, key], 
                                  outputs=decrypt_err_eve)
train_fn['eve'] = theano.function(inputs=[msg_in, key], 
                                  outputs=decrypt_err_eve,
                                  updates=updates['eve'])
pred_fn['eve']  = theano.function(inputs=[msg_in, key], outputs=eve_msg)

# Function for training either Bob+Alice or Eve for some time
def train(bob_or_eve, results, max_iters, print_every, es=0., es_limit=100):
    count = 0
    for i in range(max_iters):
        # Generate some data
        msg_in_val, key_val = gen_data()
        # Train on this batch and get loss
        loss = train_fn[bob_or_eve](msg_in_val, key_val)
        # Store absolute decryption error of the model on this batch
        results = np.hstack((results, 
                             err_fn[bob_or_eve](msg_in_val, key_val).sum()))
        # Print loss now and then
        if i % print_every == 0:
            print 'training loss:', loss
        # Early stopping if we see a low-enough decryption error enough times
        if es and loss < es:
            count += 1
            if count > es_limit:
                break
    return np.hstack((results, np.repeat(results[-1], max_iters - i - 1)))

# Initialise some empty results arrays
results_bob, results_eve = [], []
adversarial_iterations = 60

# Perform adversarial training
for i in range(adversarial_iterations):
    n = 2000
    print_every = 100
    print 'training bob and alice, run:', i+1
    results_bob = train('bob', results_bob, n, print_every, es=0.01)
    print 'training eve, run:', i+1
    results_eve = train('eve', results_eve, n, print_every, es=0.01)

# Plot the results
plt.plot([np.min(results_bob[i:i+n]) for i in np.arange(0, 
          len(results_bob), n)])
plt.plot([np.min(results_eve[i:i+n]) for i in np.arange(0, 
          len(results_eve), n)])
plt.legend(['bob', 'eve'])
plt.xlabel('adversarial iteration')
plt.ylabel('lowest decryption error achieved')
plt.show()