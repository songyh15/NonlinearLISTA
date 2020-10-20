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


def imread_CS_py(im_fn, patch_size, stride):
    im_org = np.array (Image.open (im_fn), dtype='float32')
    H, W   = im_org.shape
    num_rpatch = (H - patch_size + stride - 1) // stride + 1
    num_cpatch = (W - patch_size + stride - 1) // stride + 1
    H_pad = patch_size + (num_rpatch - 1) * stride
    W_pad = patch_size + (num_cpatch - 1) * stride
    im_pad = np.zeros ((H_pad, W_pad), dtype=np.float32)
    im_pad [:H, :W] = im_org

    return im_org, H, W, im_pad, H_pad, W_pad


def img2col_py(im_pad, patch_size, stride):
    [H, W] = im_pad.shape
    num_rpatch = (H - patch_size) / stride + 1
    num_cpatch = (W - patch_size) / stride + 1
    num_patches = int (num_rpatch * num_cpatch)
    img_col = np.zeros ([patch_size**2, num_patches])
    count = 0
    for x in range(0, H-patch_size+1, stride):
        for y in range(0, W-patch_size+1, stride):
            img_col[:, count] = im_pad[x:x+patch_size, y:y+patch_size].reshape([-1])
            count = count + 1
    return img_col


def col2im_CS_py(X_col, patch_size, stride, H, W, H_pad, W_pad):
    X0_rec = np.zeros ((H_pad, W_pad))
    counts = np.zeros ((H_pad, W_pad))
    k = 0
    for x in range(0, H_pad-patch_size+1, stride):
        for y in range(0, W_pad-patch_size+1, stride):
            X0_rec[x:x+patch_size, y:y+patch_size] += X_col[:,k].\
                    reshape([patch_size, patch_size])
            counts[x:x+patch_size, y:y+patch_size] += 1
            k = k + 1
    X0_rec /= counts
    X_rec = X0_rec[:H, :W]
    return X_rec


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
    # xt = xt[:,0:15*64]
    nmse_denom = np.sum (np.square (xt))

    """Set up input for testing."""
    config.SNR = np.inf if config.SNR == 'inf' else float (config.SNR)
    # input_, label_ = (train.setup_input_sc (config.test, p, 64, None, False,
    #                           config.supp_prob, config.SNR,
    #                           config.magdist, **config.distargs))
    input_, label_ = (train.setup_input_sc (config.test, p, xt.shape [1], None, False,
                              config.supp_prob, config.SNR,
                              config.magdist,config.function, **config.distargs))
    
    """Set up model."""
    model = setup_model (config , A=p.A)
    xhs_ = model.inference (input_, None)
     
    # """ """
    # nmse_ = []
    # nmse_.append (tf.ones(1))
    # nmse_denom_  = tf.nn.l2_loss (label_)
    # for t in range (model._T):
    #     loss_ = tf.nn.l2_loss (xhs_ [t+1] - label_)
    #     nmse_.append( loss_ / nmse_denom_ )
        

    """Create session and initialize the graph."""
    tfconfig = tf.ConfigProto (allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session (config=tfconfig) as sess:
        # graph initialization
        sess.run (tf.global_variables_initializer ())
        # load model
        model.load_trainable_variables (sess , config.modelfn)

        #  print weights
        # print('.........theta........')
        # for t in range (model._T):
        #     B_, theta_1, theta_2 = model.vars_in_layer [t]          
        #     print(sess.run(theta_2))
        # print('.........beta........')
        # for t in range (model._T):
        #     B_, theta_1, theta_2 = model.vars_in_layer [t]          
        #     print(sess.run(theta_1))
        # print('.........WiAj........')
        # for t in range (model._T):
        #     B_, theta_1, theta_2 = model.vars_in_layer [t]      
        #     WT =   sess.run(B_)
        #     WTA = np.dot(WT,model._A)
        #     print(np.max(WTA-5*np.eye(WTA.shape[0])))
        # print('.........bataWiAj........')
        # for t in range (model._T):
        #     B_, theta_1, theta_2 = model.vars_in_layer [t]      
        #     WT = sess.run(B_)
        #     beta = sess.run(theta_1)
        #     WTA = np.dot(WT,model._A)
        #     print(beta[0]*np.max(WTA-5*np.eye(WTA.shape[0])))
               

        # ATA = np.dot(model._A.T,model._A)
        # print(ATA.shape)
        # print(np.max(ATA))
        # print(np.max(ATA-np.eye(ATA.shape[0])))
        # print(np.where((ATA - np.eye(ATA.shape[0])==np.max(ATA))))

        # print('.........theta........')
        # for t in range (model._T):
        #     B_, theta_1, theta_2 = model.vars_in_layer [t]          
        #     print(sess.run(theta_2))

        
        # x_test = sess.run(label_)
        # np.save( path, x_test )
        
        supp_gt = xt != 0
        # supp_gt = x_test != 0
    
        lnmse  = []
        lspar  = []
        lsperr = []
        lflspo = []
        lflsne = []
        
      # test model
        for xh_ in xhs_ :
            xh = sess.run (xh_ , feed_dict={label_:xt})
            # xh = sess.run (xh_ , feed_dict={label_:x_test})

            # xh = np.zeros(xt.shape)
            # for i in range(15):
            #     x_now = xt[:, i*64:(i+1)*64]
            #     xh[:,i*64:(i+1)*64] = sess.run (xh_ , feed_dict={label_:x_now})

    
            # nmse:
            loss = np.sum (np.square (xh - xt))
            nmse_dB = 10.0 * np.log10 (loss / nmse_denom)
            
            # nmse = sess.run(nmse_[i])
            # nmse_dB = 10.0 * np.log10 (nmse)

            print (nmse_dB)
            lnmse.append (nmse_dB)

            supp = xh != 0.0
            # intermediate sparsity
            spar = np.sum (supp , axis=0)
            lspar.append (spar)

            # support error
            sperr = np.logical_xor(supp, supp_gt)
            lsperr.append (np.sum (sperr , axis=0))

            # false positive
            flspo = np.logical_and (supp , np.logical_not (supp_gt))
            lflspo.append (np.sum (flspo , axis=0))

            # false negative
            flsne = np.logical_and (supp_gt , np.logical_not (supp))
            lflsne.append (np.sum (flsne , axis=0))

    res = dict (nmse=np.asarray  (lnmse),
                spar=np.asarray  (lspar),
                sperr=np.asarray (lsperr),
                flspo=np.asarray (lflspo),
                flsne=np.asarray (lflsne))

    # np.savez (config.resfn , **res)
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

