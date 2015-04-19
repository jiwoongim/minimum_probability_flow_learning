''' Version 1.000

 Code provided by Daniel Jiwoong Im
 http://www.cs.uoguelph.edu/~imj

 Permission is granted for anyone to copy, use, modify, or distribute this
 program and accompanying programs and documents for any purpose, provided
 this copyright notice is retained and prominently displayed, along with
 a note saying that the original programs are available from our
 web page.
 The programs and documents are distributed without any warranty, express or
 implied.  As the programs were written for research purposes only, they have
 not been tested to the degree that would be advisable in any important
 application.  All use of these programs is entirely at the user's own risk.'''

'''For more information, see :http://arxiv.org/abs/1412.6617'''

import numpy as np
import time, pickle, sys, math
import theano
import theano.tensor as T
import os
import theano.sandbox.rng_mrg as RNG_MRG

from theano.tensor.shared_randomstreams import RandomStreams
from utils import *

'''Restricted Boltzmann Machine based on Minimum Probability Flow Learning method'''

class RBM_MPF(object):

    def __init__(self, input=None, n_visible=784, n_hidden=500, \
        W=None, hbias=None, vbias=None, numpy_rng=None,
        theano_rng=None, enhanced_grad_flag=False, batch_sz=100, mpf_type='1bit'):

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.enhanced_grad_flag = enhanced_grad_flag 
        self.batch_sz = batch_sz

        if numpy_rng is None:
            # create a number generator
            numpy_rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RNG_MRG.MRG_RandomStreams(numpy_rng.randint(2 ** 30))
            #theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        num_vishid = n_visible * n_hidden

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.mpf_type = mpf_type
        self._init_params(numpy_rng, n_hidden, n_visible, mpf_type)
        self.theano_rng = theano_rng


    '''Initialize the parameters of MPF'''
    def _init_params(self, numpy_rng, n_hidden, n_visible, mpf_type):

        # W is initialized with `initial_W` which is uniformely
        # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
        # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
        # converted using asarray to dtype theano.config.floatX so
        # that the code is runable on GPU
        initial_W = np.asarray(numpy_rng.uniform(
                  low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                  high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                  size=(n_visible, n_hidden)),
                  dtype=theano.config.floatX)

        # theano shared variables for weights and biases
        self.W = theano.shared(value=initial_W, name='W', borrow=True)

        # create shared variable for hidden units bias
        self.hbias = theano.shared(value=np.zeros(n_hidden,
                                                dtype=theano.config.floatX),
                              name='hbias', borrow=True)

        # create shared variable for visible units bias
        self.vbias = theano.shared(value=np.zeros(n_visible,
                                                dtype=theano.config.floatX),
                              name='vbias', borrow=True)
        self.params = [self.W, self.hbias, self.vbias]   

        if mpf_type != '1bit':
            self.oldW = theano.shared(initial_W, borrow=True)
            self.oldhb = theano.shared(np.zeros((self.n_hidden,),\
                                            dtype=theano.config.floatX), borrow=True)
            self.oldvb = theano.shared(np.zeros((self.n_visible,),\
                                            dtype=theano.config.floatX), borrow=True)
            self.old_params = [self.oldW, self.oldhb, self.oldvb]


    ''' Function to compute the free energy '''   
    def free_energy(self, v_sample, W, hbias, vbias):
        wx_b = T.dot(v_sample, W) + hbias
        vbias_term = T.dot(v_sample, vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=-1)
        return -hidden_term - vbias_term

    '''L2 Norm of the Weights'''
    def weight_decay(self):
		return (self.W ** 2).sum()

    '''L1 Norm of the Weights'''
    def L1_norm(self):
		return abs(self.W.sum())


    '''Cost function for Minimum Probability Flow (MPF). 
    User can specify the type of MPF learning by setting mpf_type 
    between one bit flip versus factored MPF when creating the instance of
    RBM.
    Singe bit flip only requires input X. But factored MPF require
    input X and sample Y from previous time-step'''
    def cost(self, X, Y=None, size=None):
        if size is None:
            size = self.batch_sz

        if self.mpf_type=='1bit':
            return self.bitflip_cost(X, size=size)
        elif self.mpf_type=='factored_mpf':
            return self.factored_mpf_cost(X, Y, size=size)

    '''Cost function for Single bit filp'''
    def bitflip_cost(self, X, size=None):

        if size is None:
            size = self.batch_sz

        Y = X.reshape((size, 1, self.n_visible), 3)\
            * T.ones((1, self.n_visible, 1)) #tile out data vectors (repeat each one D times)
        YY = (Y + T.eye(self.n_visible).reshape((1, self.n_visible, self.n_visible), 3))%2 # flip each bit once 
       
        free_y = self.free_energy(YY, self.W, self.hbias, self.vbias)
        free_x = self.free_energy(X, self.W, self.hbias, self.vbias).dimshuffle(0, 'x')
        Z = T.exp(0.5 * (free_x - free_y))
        K = T.sum(Z) / size 
        K.name = 'K'
        
        return T.cast(K, 'float32')

    '''Cost function for factored MPF'''
    def factored_mpf_cost(self, X, YY, size=None):
        if size is None:
            size = self.batch_sz

        free_y = self.free_energy(YY, self.W, self.hbias, self.vbias)
        free_x = self.free_energy(X, self.W, self.hbias, self.vbias)

        old_free_y = self.free_energy(YY, self.oldW, self.oldhb, self.oldvb)
        old_free_x = self.free_energy(X, self.oldW, self.oldhb, self.oldvb)

        ZD = T.sum(T.exp(0.5 * (free_x - old_free_x))) / (size)
        ZM = T.sum(T.exp(0.5 * (old_free_y - free_y))) / (size) 

        K = ZD*ZM
        K.name = 'K'
        
        return T.cast(K, 'float32'), YY


    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units
        
        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)
        
        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]


    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]


    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units
        
        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)
        
        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]


    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
        starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
        starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]


    '''Returns samples'''   
    def get_samples(self, X, step=1):

        pre_sigmoid_ph, ph_mean, ph_samples = self.sample_h_given_v(X)

        chain_start = ph_samples 
        [pre_sigmoid_nvs, nv_means, nv_samples,
         pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self.gibbs_hvh,
                    outputs_info=[None, None, None, None, None, chain_start],
                    n_steps=step)

        v1_means = nv_means[-1]
        v1_sample = nv_samples[-1]
        h1_sample = nh_samples[-1]
        return v1_sample, v1_means, h1_sample, updates



