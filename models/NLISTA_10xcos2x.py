#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink
from models.LISTA_base import LISTA_base

class NLISTA_10xcos2x (LISTA_base):

    """
    Implementation of LISTA model proposed by LeCun in 2010.
    """

    def __init__(self, A, T, lam, untied, coord, scope):
        """
        :A      : Numpy ndarray. Dictionary/Sensing matrix.
        :T      : Integer. Number of layers (depth) of this LISTA model.
        :lam    : Float. The initial weight of l1 loss term in LASSO.
        :untied : Boolean. Flag of whether weights are shared within layers.
        :scope  : String. Scope name of the model.
        """
        self._A   = A.astype (np.float32)
        self._T   = T
        self._lam = lam
        self._M   = self._A.shape [0]
        self._N   = self._A.shape [1]

        self._scale = 1.001 * np.linalg.norm (A, ord=2)**2
        self._theta = (self._lam / self._scale).astype(np.float32)
        if coord:
            self._theta = np.ones ((self._N, 1), dtype=np.float32) * self._theta
            
#         self._theta = tf.zeros(1)
        
        self._untied = untied
        self._coord  = coord
        self._scope  = scope

        """ Set up layers."""
        self.setup_layers()


    def setup_layers(self):
        """
        Implementation of LISTA model proposed by LeCun in 2010.

        :prob: Problem setting.
        :T: Number of layers in LISTA.
        :returns:
            :layers: List of tuples ( name, xh_, var_list )
                :name: description of layers.
                :xh: estimation of sparse code at current layer.
                :var_list: list of variables to be trained seperately.

        """
        Bs_     = []
        theta_1_s_ = []
        theta_2_s_ = []

        B = (np.transpose (self._A)).astype (np.float32)
#         B = tf.random_normal (shape=(self._N, self._N), stddev=1,dtype=tf.float32)

        with tf.variable_scope (self._scope, reuse=False) as vs:
            # constant
            self._kA_ = tf.constant (value=self._A, dtype=tf.float32)
        

            for t in range (self._T):
                theta_1_s_.append (tf.get_variable (name="theta_1_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=tf.ones(1)))
                theta_2_s_.append (tf.get_variable (name="theta_2_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=(np.power(0.4,10*t+1)).astype (np.float32) ))
                # theta_2_s_.append (tf.get_variable (name="theta_2_%d"%(t+1),
                #                                  dtype=tf.float32,
                #                                  initializer= self._theta ))
                                                 
                if self._untied: # untied model
                    Bs_.append (tf.get_variable (name="B_%d"%(t+1), dtype=tf.float32,initializer=B))
                    
                    
        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list (zip (Bs_, theta_1_s_, theta_2_s_))
        
        
    def inference (self, y_, x0_=None):
        xhs_  = [] # collection of the regressed sparse codes

        if x0_ is None:
            batch_size = tf.shape (y_) [-1]
            xh_ = tf.zeros (shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
        xhs_.append (xh_)

        with tf.variable_scope (self._scope, reuse=True) as vs:
            for t in range (self._T):
                B_, theta_1, theta_2 = self.vars_in_layer [t]    
                W_ = self._A
                
                residual = y_ - 10 *tf.matmul(W_,xh_) - tf.cos(2 * tf.matmul(W_,xh_))
                grad_nonlinear = 10 - 2* tf.sin(2 * tf.matmul(W_,xh_))
                
                g_res = grad_nonlinear * residual    
                g_res = g_res*tf.pow(tf.maximum(1*tf.sqrt(tf.reduce_sum(tf.square (g_res),axis=0,keepdims=True)), 1), -1)
        
                xh_ = shrink (xh_+ theta_1 * tf.matmul(B_ ,g_res), theta_2)
                xhs_.append (xh_)

        return xhs_


