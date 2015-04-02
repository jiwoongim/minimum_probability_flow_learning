''' Version 1.000

 Code provided by Daniel Jiwoong Im

 Permission is granted for anyone to copy, use, modify, or distribute this
 program and accompanying programs and documents for any purpose, provided
 this copyright notice is retained and prominently displayed, along with
 a note saying that the original programs are available from our
 web page.
 The programs and documents are distributed without any warranty, express or
 implied.  As the programs were written for research purposes only, they have
 not been tested to the degree that would be advisable in any important
 application.  All use of these programs is entirely at the user's own risk.'''

'''Demo of Minimum Probability Flow learning method with one-bit flip 
connectivities on Restricted Boltzmann Machines.
For more information, see :http://arxiv.org/abs/1412.6617
'''


import numpy as np
import time, pickle, sys, math
import theano
import theano.tensor as T
import os


'''Optimizes based on Minimum Probabaility Flow learning algorithm'''
class MPF_optimizer(object):

    def __init__(self, hyper_params):

        self.batch_sz, self.epsilon, self.lam = hyper_params


    '''Optimizing by mini-batch gradient descent'''
    def mpf_MBGD(self, model, train_data, valid_data, reg_type='l2'):

        NT = 1000 #batchsize for test data
        X = T.fmatrix('X'); i = T.iscalar('i'); lr = T.fscalar('lr'); Y = T.fmatrix('Y');

        if model.mpf_type == '1bit':
            #Defining the cost function for training data
            cost = model.bitflip_cost(X)
            if reg_type == 'l2':
                cost += self.lam * model.weight_decay()
            elif reg_type =='l1':
                cost += self.lam * model.L1_norm()  

            #Defining the cost function for test data
            cost_test = model.bitflip_cost(X, size=NT)
        else:
            #Defining the cost function for training data
            cost, YY = model.factored_mpf_cost(X,Y)
            if reg_type == 'l2':
                cost += self.lam * model.weight_decay()
            elif reg_type =='l1':
                cost += self.lam * model.L1_norm() 
            
            #Defining the cost function for test data
            cost_test,YY_test = model.factored_mpf_cost(X,Y, size=NT)

        gparams = T.grad(cost, model.params)

        #Update Gradient 
        update_grads = []
        for param, gparam in zip(model.params, gparams):

            new_param = param - lr *gparam 
            update_grads.append((param, new_param))       

        if model.mpf_type == '1bit':
            train_update = theano.function([i, theano.Param(lr,default=self.epsilon)],\
                                    outputs=cost, updates=update_grads, \
                            givens={X:train_data[i*self.batch_sz:(i+1)*self.batch_sz]})

            get_valid_cost = theano.function([i], outputs=cost_test, \
                            givens={X:valid_data[i*NT:(i+1)*NT]})
        else:
            train_update = theano.function([i, Y,\
                theano.Param(lr,default=self.epsilon)], outputs=cost, updates=update_grads, \
                        givens={X:train_data[i*self.batch_sz:(i+1)*self.batch_sz]})

            get_valid_cost = theano.function([i,Y], outputs=cost_test, \
                        givens={X:valid_data[i*self.batch_sz:(i+1)*self.batch_sz]})


        return train_update, get_valid_cost

