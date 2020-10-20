
import argparse
import numpy as np
import utils.prob as problem

def FUN_quad(q,t,K):
    # soft-thresholding: min_x 0.5*x^2-q*x+t|x|
    # x = np.maximum(q-t,np.zeros([K,1]))-np.maximum(-q-t,np.zeros([K,1]))
    x = np.sign(q) * np.maximum( np.abs(q) - t, 0.0 )
    return x

def SpaRSA(config):
    print('function is ' + config.function)
    print('sample number is ' + str(config.sample))

    # parameters
    N = 250 
    K = 500
    MaxIter = config.T # maximum number of iterations
    Sample=config.sample # number of repeatitions in the Monte Carlo simulations

    xt = np.load ('./data/x_test_p0.1_1.npy')
    if config.cond == 50:
        p = problem.load_problem ('./experiments/prob_k50.npz')
    elif config.cond == 0:
        p = problem.load_problem ('./experiments/prob.npz')
    else:
        RuntimeError('cond error')
    A = p.A
    mu = config.mu
    if config.function == 'linear':
        bt = np.dot(A,xt)
    elif config.function == '2xcosx':
        bt = 2*np.dot(A,xt) + np.cos(np.dot(A,xt))
    elif config.function == '10xcos2x':
        bt = 10*np.dot(A,xt) + np.cos(2*np.dot(A,xt))
    elif config.function == '10xcos3x':
        bt = 10*np.dot(A,xt) + np.cos(3*np.dot(A,xt))
    elif config.function == '10xcos4x':
        bt = 10*np.dot(A,xt) + np.cos(4*np.dot(A,xt))
    else:
        RuntimeError('there is no such function')

    if config.SNR == 'inf':
        print('SNR is inf')
    else:
        config.SNR =float(config.SNR)
        std_ = np.std(bt,axis=0) * np.power (10.0, -config.SNR/20.0)
        noise_ = np.random.normal(0, std_, bt.shape)
        bt += noise_
        print('SNR is '+str(config.SNR))

    objval_i=np.zeros([Sample,MaxIter+1])
    l2_loss_i=np.zeros([Sample,MaxIter+1])
    db_nmse_i=np.zeros([Sample,MaxIter+1])

    X_ori = np.zeros([K,Sample])
    X_pre = np.zeros([K,Sample,MaxIter+1])

    for s in range(Sample):
        # print('Sample '+str(s+1))

        # generating the parameter
        x_ori = xt[:,s]
        l2_x_ori = np.sum (np.square (x_ori))
        X_ori[:,s] = x_ori
        b = bt[:,s:s+1]

        # initialization

        x = np.zeros([K,1])
        Ax= np.dot(A,x)
        
        if config.function == 'linear':
            residual= Ax - b
        elif config.function == '2xcosx':
            residual= 2*Ax + np.cos(Ax) - b
        elif config.function == '10xcos2x':
            residual= 10*Ax + np.cos(2*Ax)-b
        elif config.function == '10xcos3x':
            residual= 10*Ax + np.cos(3*Ax)-b
        elif config.function == '10xcos4x':
            residual= 10*Ax + np.cos(4*Ax)-b

        val_f = 0.5*np.sum (np.square (residual))
        val_g = mu*np.linalg.norm(x,ord=1)
        objval_i[s,0]=val_f+val_g
        for t in range(1,MaxIter+1):
            Ax = np.dot(A,x)
            if config.function == 'linear':
                Gradient_sigma = np.ones([N,1])
            elif config.function == '2xcosx':
                Gradient_sigma = 2*np.ones([N,1])-np.sin(Ax)
            elif config.function == '10xcos2x':
                Gradient_sigma = 10*np.ones([N,1])-2*np.sin(2*Ax)
            elif config.function == '10xcos3x':
                Gradient_sigma = 10*np.ones([N,1])-3*np.sin(3*Ax)
            elif config.function == '10xcos4x':
                Gradient_sigma = 10*np.ones([N,1])-4*np.sin(4*Ax) 
            
            Gradient_f = np.dot(A.T, residual*Gradient_sigma)
            
            # compute best-response
            if t == 1:
                L=1       
            else:
                delta = x - x_old
                g = Gradient_f - Gradient_f_old
                if np.dot(delta.T, delta):
                    L = np.dot(delta.T, g) / np.dot(delta.T, delta)
                else:
                    L = 1
            if L==0:
                L = 1

            eta = 2
            x_old = x
            Gradient_f_old = Gradient_f
            while 1:
                x_new = FUN_quad(x-Gradient_f/L, mu/L, K)
                Ax_new = np.dot(A, x_new)
                
                if config.function == 'linear':
                    residual_new = Ax_new - b
                elif config.function == '2xcosx':
                    residual_new = 2*Ax_new + np.cos(Ax_new)-b
                elif config.function == '10xcos2x':
                    residual_new = 10*Ax_new + np.cos(2*Ax_new)-b
                elif config.function == '10xcos3x':
                    residual_new = 10*Ax_new + np.cos(3*Ax_new)-b
                elif config.function == '10xcos4x':
                    residual_new = 10*Ax_new + np.cos(4*Ax_new)-b

                val_f_new = 0.5*np.sum (np.square (residual_new))
                val_g_new = mu * np.linalg.norm(x_new,ord=1)

                if  val_f_new+val_g_new <= np.max(objval_i[s,np.maximum(t-1,0):t]-1e-5*L/2*np.sum(np.square (x_new-x))):
                    x = x_new
                    Ax = Ax_new 
                    val_f=val_f_new 
                    val_g=val_g_new 
                    residual=residual_new
                    break
                else:
                    L = L*eta

            X_pre[:,s,t] = x[:,0]
            objval_i[s,t] = val_f+val_g
            l2_loss_i[s,t] = np.sum(np.square (x[:,0] - x_ori))
            db_nmse_i[s,t] = 10 * np.log10(l2_loss_i[s,t] / l2_x_ori)

    l2_demo = np.sum (np.square (X_ori))
    l2_loss = np.zeros([MaxIter+1,1])
    db_nmse = np.zeros([MaxIter+1,1])

    print('mu = '+str(mu))
    print('db_nmse ')
    for t in range(MaxIter+1):
        l2_loss[t] = np.sum (np.square (X_pre[:,:,t] - X_ori))
        db_nmse[t] = 10 * np.log10(l2_loss[t] / l2_demo)
        print(str(db_nmse[t][0]))

def FPCA(config):
    print('function is ' + config.function)
    print('sample number is ' + str(config.sample))

    # parameters
    N = 250 
    K = 500
    MaxIter = config.T # maximum number of iterations
    Sample=config.sample # number of repeatitions in the Monte Carlo simulations

    xt = np.load ('./data/x_test_p0.1_1.npy')
    if config.cond == 50:
        p = problem.load_problem ('./experiments/prob_k50.npz')
    elif config.cond == 0:
        p = problem.load_problem ('./experiments/prob.npz')
    else:
        RuntimeError('cond error')
    A = p.A
    mu_ori = config.mu
    if config.function == 'linear':
        bt = np.dot(A,xt)
    elif config.function == '2xcosx':
        bt = 2*np.dot(A,xt) + np.cos(np.dot(A,xt))
    elif config.function == '10xcos2x':
        bt = 10*np.dot(A,xt) + np.cos(2*np.dot(A,xt))
    elif config.function == '10xcos3x':
        bt = 10*np.dot(A,xt) + np.cos(3*np.dot(A,xt))
    elif config.function == '10xcos4x':
        bt = 10*np.dot(A,xt) + np.cos(4*np.dot(A,xt))
    else:
        RuntimeError('there is no such function')

    if config.SNR == 'inf':
        print('SNR is inf')
    else:
        config.SNR =float(config.SNR)
        std_ = np.std(bt,axis=0) * np.power (10.0, -config.SNR/20.0)
        noise_ = np.random.normal(0, std_, bt.shape)
        bt += noise_
        print('SNR is '+str(config.SNR))

    objval_i=np.zeros([Sample,MaxIter+1])
    l2_loss_i=np.zeros([Sample,MaxIter+1])
    db_nmse_i=np.zeros([Sample,MaxIter+1])

    X_ori = np.zeros([K,Sample])
    X_pre = np.zeros([K,Sample,MaxIter+1])

    result = []

    decay_rate = config.decay_rate
    print('decay_rate = '+str(decay_rate))
    for s in range(Sample):
        mu = mu_ori
        # print('Sample '+str(s+1))

        # generating the parameter
        x_ori = xt[:,s]
        l2_x_ori = np.sum (np.square (x_ori))
        X_ori[:,s] = x_ori
        b = bt[:,s:s+1]

        # initialization
        x = np.zeros([K,1])
        Ax= np.dot(A,x)
        
        if config.function == 'linear':
            residual= Ax - b
        elif config.function == '2xcosx':
            residual= 2*Ax+np.cos(Ax) - b
        elif config.function == '10xcos2x':
            residual= 10*Ax + np.cos(2*Ax)-b
        elif config.function == '10xcos3x':
            residual= 10*Ax + np.cos(3*Ax)-b
        elif config.function == '10xcos4x':
            residual= 10*Ax + np.cos(4*Ax)-b

        val_f = 0.5*np.sum (np.square (residual))
        val_g = mu*np.linalg.norm(x,ord=1)
        objval_i[s,0]=val_f+val_g

        mu_thre = 0.05
        for t in range(1,MaxIter+1):
            if (t>1) and (np.linalg.norm(x-x_old,ord=2)< mu_thre):
                mu *= decay_rate
                mu_thre *= decay_rate


            Ax = np.dot(A,x)
            if config.function == 'linear':
                Gradient_sigma = np.ones([N,1])
            elif config.function == '2xcosx':
                Gradient_sigma = 2*np.ones([N,1])-np.sin(Ax)
            elif config.function == '10xcos2x':
                Gradient_sigma = 10*np.ones([N,1])-2*np.sin(2*Ax)
            elif config.function == '10xcos3x':
                Gradient_sigma = 10*np.ones([N,1])-3*np.sin(3*Ax)
            elif config.function == '10xcos4x':
                Gradient_sigma = 10*np.ones([N,1])-4*np.sin(4*Ax) 
            
            Gradient_f = np.dot(A.T, residual*Gradient_sigma)
            
            # compute best-response
            if t == 1:
                L=1       
            else:
                delta = x - x_old
                g = Gradient_f - Gradient_f_old
                if np.dot(delta.T, delta):
                    L = np.dot(delta.T, g) / np.dot(delta.T, delta)
                else:
                    L = 1
            if L==0:
                L = 1

            eta = 2
            x_old = x
            Gradient_f_old = Gradient_f
            while 1:
                x_new = FUN_quad(x-Gradient_f/L, mu/L, K)
                Ax_new = np.dot(A, x_new)
                
                # 不同非线性形式
                if config.function == 'linear':
                    residual_new = Ax_new - b
                elif config.function == '2xcosx':
                    residual_new = 2*Ax_new + np.cos(Ax_new)-b
                elif config.function == '10xcos2x':
                    residual_new = 10*Ax_new + np.cos(2*Ax_new)-b
                elif config.function == '10xcos3x':
                    residual_new = 10*Ax_new + np.cos(3*Ax_new)-b
                elif config.function == '10xcos4x':
                    residual_new = 10*Ax_new + np.cos(4*Ax_new)-b

                val_f_new = 0.5*np.sum (np.square (residual_new))
                val_g_new = mu * np.linalg.norm(x_new,ord=1)

                if  val_f_new+val_g_new <= np.max(objval_i[s,np.maximum(t-1-0,0):t]-1e-5*L/2*np.sum(np.square (x_new-x))):
                    x = x_new
                    Ax = Ax_new 
                    val_f=val_f_new 
                    val_g=val_g_new 
                    residual=residual_new
                    break
                else:
                    L=L*eta

            X_pre[:,s,t] = x[:,0]
            objval_i[s,t] = val_f+val_g
            l2_loss_i[s,t] = np.sum(np.square (x[:,0] - x_ori))
            db_nmse_i[s,t] = 10 * np.log10(l2_loss_i[s,t] / l2_x_ori)

    l2_demo = np.sum (np.square (X_ori))
    l2_loss = np.zeros([MaxIter+1,1])
    db_nmse = np.zeros([MaxIter+1,1])

    print('mu_final = '+str(mu))
    print('db_nmse ')
    for t in range(MaxIter+1):
        l2_loss[t] = np.sum (np.square (X_pre[:,:,t] - X_ori))
        db_nmse[t] = 10 * np.log10(l2_loss[t] / l2_demo)
        print(str(db_nmse[t][0]))

def STELA(config):
    print('function is ' + config.function)
    print('sample number is ' + str(config.sample))

    # parameters
    N = 250 
    K = 500
    MaxIter = config.T # maximum number of iterations
    Sample=config.sample # number of repeatitions in the Monte Carlo simulations

    xt = np.load ('./data/x_test_p0.1_1.npy')
    if config.cond == 50:
        p = problem.load_problem ('./experiments/prob_k50.npz')
    elif config.cond == 0:
        p = problem.load_problem ('./experiments/prob.npz')
    else:
        RuntimeError('cond error')
    A = p.A
    mu = config.mu
    if config.function == 'linear':
        bt = np.dot(A,xt)
    elif config.function == '2xcosx':
        bt = 2*np.dot(A,xt) + np.cos(np.dot(A,xt))
    elif config.function == '10xcos2x':
        bt = 10*np.dot(A,xt) + np.cos(2*np.dot(A,xt))
    elif config.function == '10xcos3x':
        bt = 10*np.dot(A,xt) + np.cos(3*np.dot(A,xt))
    elif config.function == '10xcos4x':
        bt = 10*np.dot(A,xt) + np.cos(4*np.dot(A,xt))
    else:
        RuntimeError('there is no such function')

    if config.SNR == 'inf':
        print('SNR is inf')
    else:
        config.SNR =float(config.SNR)
        std_ = np.std(bt,axis=0) * np.power (10.0, -config.SNR/20.0)
        noise_ = np.random.normal(0, std_, bt.shape)
        bt += noise_
        print('SNR is '+str(config.SNR))

    objval_i=np.zeros([Sample,MaxIter+1])
    l2_loss_i=np.zeros([Sample,MaxIter+1])
    db_nmse_i=np.zeros([Sample,MaxIter+1])

    X_ori = np.zeros([K,Sample])
    X_pre = np.zeros([K,Sample,MaxIter+1])

    for s in range(Sample):
        # generating the parameter
        x_ori = xt[:,s]
        l2_x_ori = np.sum (np.square (x_ori))
        X_ori[:,s] = x_ori
        b = bt[:,s:s+1]

        # initialization
        x = np.zeros([K,1])
        Ax= np.dot(A,x)
        
        if config.function == 'linear':
            residual= Ax - b
        elif config.function == '2xcosx':
            residual= 2*Ax+np.cos(Ax) - b
        elif config.function == '10xcos2x':
            residual= 10*Ax + np.cos(2*Ax)-b
        elif config.function == '10xcos3x':
            residual= 10*Ax + np.cos(3*Ax)-b
        elif config.function == '10xcos4x':
            residual= 10*Ax + np.cos(4*Ax)-b

        val_f = 0.5*np.sum (np.square (residual))
        val_g = mu*np.linalg.norm(x,ord=1)
        objval_i[s,0]=val_f+val_g

        for t in range(1,MaxIter+1):
            Ax = np.dot(A,x)

            if config.function == 'linear':
                Gradient_sigma = np.ones([N,1])
            elif config.function == '2xcosx':
                Gradient_sigma = 2*np.ones([N,1])-np.sin(Ax)
            elif config.function == '10xcos2x':
                Gradient_sigma = 10*np.ones([N,1])-2*np.sin(2*Ax)
            elif config.function == '10xcos3x':
                Gradient_sigma = 10*np.ones([N,1])-3*np.sin(3*Ax)
            elif config.function == '10xcos4x':
                Gradient_sigma = 10*np.ones([N,1])-4*np.sin(4*Ax)
            
            Gradient_f = np.dot(A.T, residual*Gradient_sigma)
            
                            # compute best-response
            if t == 1:
                L=1       
            else:
                delta = x - x_old
                g = Gradient_f - Gradient_f_old
                if np.dot(delta.T, delta):
                    L = np.dot(delta.T, g) / np.dot(delta.T, delta)
                else:
                    L = 1
            if L==0:
                L = 1

            x_old = x
            Gradient_f_old = Gradient_f

            # compute best-response
            Bx = FUN_quad(x-Gradient_f/L, mu/L, K)
            # compute stepsize
            x_dif = Bx-x
            Ax_dif = np.dot(A, x_dif)
            val_g_new = mu * np.linalg.norm(Bx,ord=1)
            # to compute the stepsize by successive line search
            descent = np.dot(Gradient_f.T,x_dif) + mu * np.linalg.norm(Bx,ord=1) - mu * np.linalg.norm(x,ord=1)        
            alpha = 0.01 
            beta = 0.5 
            stepsize = 1
            while 1:
                x_new = x + stepsize*x_dif
                Ax_new = Ax + stepsize*Ax_dif
                
                if config.function == 'linear':
                    residual_new = Ax_new - b
                elif config.function == '2xcosx':
                    residual_new = 2*Ax_new + np.cos(Ax_new)-b
                elif config.function == '10xcos2x':
                    residual_new = 10*Ax_new + np.cos(2*Ax_new)-b
                elif config.function == '10xcos3x':
                    residual_new = 10*Ax_new + np.cos(3*Ax_new)-b
                elif config.function == '10xcos4x':
                    residual_new = 10*Ax_new + np.cos(4*Ax_new)-b
                
                val_f_new = 0.5*np.sum (np.square (residual_new))
                if val_f_new - val_f + stepsize*(val_g_new-val_g) <= alpha*stepsize*descent:
                    val_f = val_f_new 
                    residual = residual_new 
                    Ax = Ax_new 
                    x = x_new 
                    val_g = mu * np.linalg.norm(x,ord=1) 
                    break
                else:
                    stepsize = stepsize*beta

            
            X_pre[:,s,t] = x[:,0]
            objval_i[s,t] = val_f+val_g
            l2_loss_i[s,t] = np.sum(np.square (x[:,0] - x_ori))
            db_nmse_i[s,t] = 10 * np.log10(l2_loss_i[s,t] / l2_x_ori)

    l2_demo = np.sum (np.square (X_ori))
    l2_loss = np.zeros([MaxIter+1,1])
    db_nmse = np.zeros([MaxIter+1,1])

    print('mu = '+str(mu))
    print('db_nmse ')
    for t in range(MaxIter+1):
        l2_loss[t] = np.sum (np.square (X_pre[:,:,t] - X_ori))
        db_nmse[t] = 10 * np.log10(l2_loss[t] / l2_demo)
        print(str(db_nmse[t][0]))

def FISTA(config):
    print('function is ' + config.function)
    print('sample number is ' + str(config.sample))

    # parameters
    N = 250 
    K = 500
    MaxIter = config.T # maximum number of iterations
    Sample=config.sample # number of repeatitions in the Monte Carlo simulations

    xt = np.load ('./data/x_test_p0.1_1.npy')
    if config.cond == 50:
        p = problem.load_problem ('./experiments/prob_k50.npz')
    elif config.cond == 0:
        p = problem.load_problem ('./experiments/prob.npz')
    else:
        RuntimeError('cond error')
    A = p.A
    mu = config.mu
    if config.function == 'linear':
        bt = np.dot(A,xt)
    elif config.function == '2xcosx':
        bt = 2*np.dot(A,xt) + np.cos(np.dot(A,xt))
    elif config.function == '10xcos2x':
        bt = 10*np.dot(A,xt) + np.cos(2*np.dot(A,xt))
    elif config.function == '10xcos3x':
        bt = 10*np.dot(A,xt) + np.cos(3*np.dot(A,xt))
    elif config.function == '10xcos4x':
        bt = 10*np.dot(A,xt) + np.cos(4*np.dot(A,xt))
    else:
        RuntimeError('there is no such function')

    if config.SNR == 'inf':
        print('SNR is inf')
    else:
        config.SNR =float(config.SNR)
        std_ = np.std(bt,axis=0) * np.power (10.0, -config.SNR/20.0)
        noise_ = np.random.normal(0, std_, bt.shape)
        bt += noise_
        print('SNR is '+str(config.SNR))

    objval_i=np.zeros([Sample,MaxIter+1])
    l2_loss_i=np.zeros([Sample,MaxIter+1])
    db_nmse_i=np.zeros([Sample,MaxIter+1])

    X_ori = np.zeros([K,Sample])
    X_pre = np.zeros([K,Sample,MaxIter+1])

    for s in range(Sample):
        # generating the parameters
        x_ori = xt[:,s]
        l2_x_ori = np.sum (np.square (x_ori))
        X_ori[:,s] = x_ori
        b = bt[:,s:s+1]

        # initialization
        x = np.zeros([K,1])
        Ax= np.dot(A,x)

        if config.function == 'linear':
            residual= Ax - b
        elif config.function == '2xcosx':
            residual= 2*Ax+np.cos(Ax) - b
        elif config.function == '10xcos2x':
            residual= 10*Ax + np.cos(2*Ax)-b
        elif config.function == '10xcos3x':
            residual= 10*Ax + np.cos(3*Ax)-b
        elif config.function == '10xcos4x':
            residual= 10*Ax + np.cos(4*Ax)-b

        val_f = 0.5*np.sum (np.square (residual))
        val_g = mu*np.linalg.norm(x,ord=1)
        objval_i[s,0]=val_f+val_g

        t_old = 1
        for t in range(1,MaxIter+1):
            Ax = np.dot(A,x)

            if config.function == 'linear':
                Gradient_sigma = np.ones([N,1])
            elif config.function == '2xcosx':
                Gradient_sigma = 2*np.ones([N,1])-np.sin(Ax)
            elif config.function == '10xcos2x':
                Gradient_sigma = 10*np.ones([N,1])-2*np.sin(2*Ax)
            elif config.function == '10xcos3x':
                Gradient_sigma = 10*np.ones([N,1])-3*np.sin(3*Ax)
            elif config.function == '10xcos4x':
                Gradient_sigma = 10*np.ones([N,1])-4*np.sin(4*Ax)
            
            Gradient_f = np.dot(A.T, residual*Gradient_sigma)
            
            # compute best-response
            if t == 1:
                L=1       
            else:
                delta = x - x_old
                g = Gradient_f - Gradient_f_old
                if np.dot(delta.T, delta):
                    L = np.dot(delta.T, g) / np.dot(delta.T, delta)
                else:
                    L = 1
            if L<=0:
                L = 1

            eta = 2
            x_old = x
            Gradient_f_old = Gradient_f
            if t >1:
                x = z

            while L<1e5:
                x_new = FUN_quad(x-Gradient_f/L, mu/L, K)
                Ax_new = np.dot(A, x_new)
                
                # 不同非线性形式
                if config.function == 'linear':
                    residual_new = Ax_new - b
                elif config.function == '2xcosx':
                    residual_new = 2*Ax_new + np.cos(Ax_new)-b
                elif config.function == '10xcos2x':
                    residual_new = 10*Ax_new + np.cos(2*Ax_new)-b
                elif config.function == '10xcos3x':
                    residual_new = 10*Ax_new + np.cos(3*Ax_new)-b
                elif config.function == '10xcos4x':
                    residual_new = 10*Ax_new + np.cos(4*Ax_new)-b

                val_f_new = 0.5*np.sum (np.square (residual_new))
                val_g_new = mu * np.linalg.norm(x_new,ord=1)
                if  val_f_new+val_g_new <= np.max(objval_i[s,np.maximum(t-1,0):t]-1e-5*L/2*np.sum(np.square (x_new-x))):
                    x = x_new
                    Ax = Ax_new 
                    val_f=val_f_new 
                    val_g=val_g_new 
                    residual=residual_new
                    break
                else:
                    L=L*eta

            X_pre[:,s,t] = x[:,0]
            objval_i[s,t] = val_f+val_g
            l2_loss_i[s,t] = np.sum(np.square (x[:,0] - x_ori))
            db_nmse_i[s,t] = 10 * np.log10(l2_loss_i[s,t] / l2_x_ori)

            t_new = 0.5*np.power(1+(1+4*t_old*t_old),0.5)
            z = x + (t_old-1)/t_new*(x-x_old)
            t_old = t_new

    l2_demo = np.sum (np.square (X_ori))
    l2_loss = np.zeros([MaxIter+1,1])
    db_nmse = np.zeros([MaxIter+1,1])

    print('mu = '+str(mu))
    print('db_nmse ')
    for t in range(MaxIter+1):
        l2_loss[t] = np.sum (np.square (X_pre[:,:,t] - X_ori))
        db_nmse[t] = 10 * np.log10(l2_loss[t] / l2_demo)
        print(str(db_nmse[t][0]))

def main (config):
    print('model is '+ config.model)
    if config.model == 'SpaRSA':
        SpaRSA(config)
    elif config.model == 'FPCA':
        FPCA(config)
    elif config.model == 'STELA':
        STELA(config)
    elif config.model == 'FISTA':
        FISTA(config)
    else:
        RuntimeError('there is no such model')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument ('-model', '--model', type=str, help='Model name.')
    parser.add_argument ('-s', '--sample', type=int, default=1000, help='Sample number.')
    parser.add_argument ('-T', '--T', type=int, default=16, help='Iteration number.')
    parser.add_argument ('-f', '--function', type=str, help='Function.')
    parser.add_argument ('-mu', '--mu', type=float, default=0.5, help='regularization.')
    parser.add_argument ('-SNR', '--SNR', type=str, default='inf')
    parser.add_argument ('-cond', '--cond', type=int, help='Condition number.')
    parser.add_argument ('-dr', '--decay_rate', type=float,default=0.5, help='Decay rate.')
    config, _ = parser.parse_known_args ()
    main (config)