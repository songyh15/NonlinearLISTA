#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os , sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!

# timing
import time
from datetime import timedelta

from config import get_config
import utils.prob as problem
import utils.data as data
import utils.train as train

import numpy as np
import tensorflow as tf
try :
    from PIL import Image
    from sklearn.feature_extraction.image \
            import extract_patches_2d, reconstruct_from_patches_2d
except Exception as e :
    pass


def setup_model (config , **kwargs) :
    untiedf = 'u' if config.untied else 't'
    coordf  = 'c' if config.coord  else 's'
        
    """LISTA"""
    if config.net == 'LISTA' :
        config.model = ("LISTA_T{T}_lam{lam}_{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.LISTA import LISTA
        model = LISTA (kwargs ['A'], T=config.T, lam=config.lam,
                       untied=config.untied, coord=config.coord,
                       scope=config.scope)
        
    """NLISTA_2xcosx"""
    if config.net == 'NLISTA_2xcosx' :
        config.model = ("NLISTA_2xcosx_T{T}_lam{lam}_{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.NLISTA_2xcosx import NLISTA_2xcosx
        model = NLISTA_2xcosx (kwargs ['A'], T=config.T, lam=config.lam,
                       untied=config.untied, coord=config.coord,
                       scope=config.scope)
    
    """NLISTA_10xcos2x"""
    if config.net == 'NLISTA_10xcos2x' :
        config.model = ("NLISTA_10xcos2x_T{T}_lam{lam}_{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.NLISTA_10xcos2x import NLISTA_10xcos2x
        model = NLISTA_10xcos2x (kwargs ['A'], T=config.T, lam=config.lam,
                       untied=config.untied, coord=config.coord,
                       scope=config.scope)

    """NLISTA_10xcos3x"""
    if config.net == 'NLISTA_10xcos3x' :
        config.model = ("NLISTA_10xcos3x_T{T}_lam{lam}_{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.NLISTA_10xcos3x import NLISTA_10xcos3x
        model = NLISTA_10xcos3x (kwargs ['A'], T=config.T, lam=config.lam,
                       untied=config.untied, coord=config.coord,
                       scope=config.scope)

    """NLISTA_10xcos4x"""
    if config.net == 'NLISTA_10xcos4x' :
        config.model = ("NLISTA_10xcos4x_T{T}_lam{lam}_{untiedf}_{coordf}_{exp_id}"
                        .format (T=config.T, lam=config.lam, untiedf=untiedf,
                                 coordf=coordf, exp_id=config.exp_id))
        from models.NLISTA_10xcos4x import NLISTA_10xcos4x
        model = NLISTA_10xcos4x (kwargs ['A'], T=config.T, lam=config.lam,
                       untied=config.untied, coord=config.coord,
                       scope=config.scope)

    config.modelfn = os.path.join (config.expbase, config.model)
    config.resfn   = os.path.join (config.resbase, config.model)
    print ("model disc:", config.model)

    return model


############################################################
######################   Training    #######################
############################################################

def run_train (config) :
    if config.task_type == 'sc':
        run_sc_train (config)

def run_sc_train (config) :
    """Load problem."""
    if not os.path.exists (config.probfn):
        raise ValueError ("Problem file not found.")
    else:
        p = problem.load_problem (config.probfn)

    """Set up model."""
    model = setup_model (config, A=p.A)

    """Set up input."""
    config.SNR = np.inf if config.SNR == 'inf' else float (config.SNR)
    y_, x_, y_val_, x_val_ = (
        train.setup_input_sc (
            config.test, p, config.tbs, config.vbs, config.fixval,
            config.supp_prob, config.SNR, config.magdist,config.function, **config.distargs))

    """Set up training."""
    stages = train.setup_sc_training (
            model, y_, x_, y_val_, x_val_, None,
            config.init_lr, config.decay_rate, config.lr_decay)


    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer ())

        # start timer
        start = time.time ()

        # train model
        model.do_training (sess, stages, config.modelfn, config.scope,
                           config.val_step, config.maxit, config.better_wait)
            
        # end timer
        end = time.time ()
        elapsed = end - start
        print ("elapsed time of training = " + str (timedelta (seconds=elapsed)))

############################################################
######################   Testing    ########################
############################################################

def run_test (config):
    if config.task_type == 'sc':
        run_sc_test (config)

def run_sc_test (config) :
    """
    Test model.
    """

    """Load problem."""
    if not os.path.exists (config.probfn):
        raise ValueError ("Problem file not found.")
    else:
        p = problem.load_problem (config.probfn)

    """Load testing data."""
    xt = np.load (config.xtest)
    nmse_denom = np.sum (np.square (xt))

    """Set up input for testing."""
    config.SNR = np.inf if config.SNR == 'inf' else float (config.SNR)
    input_, label_ = (train.setup_input_sc (config.test, p, xt.shape [1], None, False,
                              config.supp_prob, config.SNR,
                              config.magdist,config.function, **config.distargs))
    
    """Set up model."""
    model = setup_model (config , A=p.A)
    xhs_ = model.inference (input_, None)        

    """Create session and initialize the graph."""
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer ())
        # load model
        model.load_trainable_variables (sess , config.modelfn)

        supp_gt = xt != 0
        
      # test model
        lay = 0
        for xh_ in xhs_ :
            xh = sess.run (xh_ , feed_dict={label_:xt})
            # nmse:
            loss = np.sum (np.square (xh - xt))
            nmse_dB = 10.0 * np.log10 (loss / nmse_denom)
            print ('layer '+str(lay)+': '+str(nmse_dB))
            lay += 1
    # end of test

############################################################
#######################    Main    #########################
############################################################

def main ():
    # parse configuration
    config, _ = get_config()
    # set visible GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    if config.test:
        run_test (config)
    else:
        run_train (config)
    # end of main

if __name__ == "__main__":
    main ()

