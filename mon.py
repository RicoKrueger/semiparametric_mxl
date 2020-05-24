from numba import jit
import time
import numpy as np
from scipy.stats import invwishart
from math import floor

###
#Convenience
###

@jit
def catrnd(p):
    cp = p.cumsum(axis = 1)
    N = p.shape[0]
    q = np.zeros((N,), dtype = 'int64')
    for n in np.arange(N):
        q[n] = np.searchsorted(cp[n,:], np.random.rand())         
    return q

def compsize(q,K):
    u, c = np.unique(q, return_counts = True)
    qN = np.zeros((K,), dtype = 'int64')
    qN[u] = c
    return qN

@jit
def monrnd(mu, Sigma, q):
    N = q.shape[0]
    R = mu.shape[1]
    x = np.zeros((N,R))
    ch = np.linalg.cholesky(Sigma)
    for n in np.arange(N): 
        x[n,:] = mu[q[n],:] + ch[q[n],:,:] @ np.random.randn(R,)
    return x

###
#Probabilities
###

@jit 
def mvnpdf_norm(x, mu, Sigma):
    xS = (x - mu).T
    f = np.exp(-0.5 * (xS * np.linalg.solve(Sigma, xS)).sum(axis = 0))
    k = mu.shape[0]
    con = np.sqrt((2 * np.pi)**k * np.linalg.det(Sigma))
    return f / con

@jit    
def mvnlpdf_norm(x, mu, Sigma):
    xS = (x - mu).T
    f = -0.5 * (xS * np.linalg.solve(Sigma, xS)).sum(axis = 0)
    k = mu.shape[0]
    con = (k / 2) * np.log(2 * np.pi) + 0.5 * np.sum(np.log(np.diag(Sigma)))
    return f - con

@jit
def monlpdf_q(x, mu, Sigma, q):
    N = x.shape[0]
    f = np.zeros((N,))
    for n in np.arange(N): 
        f[n] = mvnlpdf_norm(x[n,:], mu[q[n],:], Sigma[q[n],:,:])
    return f

@jit
def monpdf(x, mu, Sigma, pi):
    K = pi.shape[0]
    f = 0
    for k in np.arange(K): 
        f += pi[k] * mvnpdf_norm(x, mu[k,:], Sigma[k,:,:])
    return f

###
#MCMC
###
    
def next_eta(alpha, qN):
    cqN = np.cumsum(qN[::-1])[::-1]
    eta = np.random.beta(1 + qN[:-1], alpha + cqN[1:])
    return eta
    
def next_pi(alpha, qN, K, s, method):
    if method == 'f':
        pi = np.random.dirichlet(alpha + qN)
    elif method == 'dp':
        eta = next_eta(alpha, qN)
        etaC = 1 - eta
        etaC[etaC < 1e-300] = 1e-300
        cumprodEtaC = np.cumprod(etaC)
        pi = np.ones((K,))
        pi[:-1] = eta
        pi[1:] *= cumprodEtaC
        pi /= pi.sum()
        try: 
            alphaU = np.random.gamma(s[0] + K - 1, 1 / (s[1] - np.log(etaC).sum()))
        except:
            alphaU = alpha
        if alphaU > 0.3: alpha = alphaU
    else:
        assert False, 'Method not supported!'
    return alpha, pi
    
def next_q(x, mu, Sigma, pi, nInd, K):
    phi = np.zeros((nInd, K))
    for k in np.arange(K):
        phi[:,k] = mvnpdf_norm(x, mu[k,:], Sigma[k,:,:])
    numer = pi.reshape((1, K)) * phi
    denom = numer.sum(axis = 1).reshape((nInd, 1))
    p = numer / denom
    q = catrnd(p)
    qN = compsize(q,K)
    return q, qN

def next_mu(x, Sigma, mu0, Si0Inv, nInd, R):
    SigmaInv = np.linalg.inv(Sigma)
    mu_SiInv = Si0Inv + nInd * SigmaInv
    mu_mu = np.linalg.solve(mu_SiInv, 
                            Si0Inv @ mu0 + SigmaInv @ np.sum(x, axis = 0))
    mu = mu_mu + np.linalg.solve(np.linalg.cholesky(mu_SiInv).T,
                                 np.random.randn(R,))
    return mu

def next_Sigma(x, mu, nu, iwDiagA, diagCov, nInd, R):
    xS = np.array(x.reshape((nInd, R)) - mu.reshape((1, R))).reshape((nInd, R))
    Sigma = np.array(invwishart.rvs(nu + nInd + R - 1, 2 * nu * np.diag(iwDiagA) + xS.T @ xS)).reshape((R, R))
    if diagCov: Sigma = np.diag(np.diag(Sigma))
    return Sigma

def next_iwDiagA(Sigma, nu, invASq, R):
    iwDiagA = np.random.gamma((nu + R) / 2, 1 / (invASq + nu * np.diag(np.linalg.inv(Sigma))))
    return iwDiagA

def next_g0_k(mu0, Si0, nu, invASq, R):
    mu_k = mu0 + np.linalg.cholesky(Si0) @ np.random.randn(R,)
    iwDiagA_k = np.random.gamma(1 / 2, 1 / invASq)
    Sigma_k = np.array(invwishart.rvs(nu + R - 1, 2 * nu * np.diag(iwDiagA_k)).reshape((R, R)))
    return mu_k, Sigma_k, iwDiagA_k  

def next_theta(
        x, mu, Sigma, iwDiagA, mu0, Si0, Si0Inv, 
        nu, invASq, diagCov, 
        q, qN, nInd, K, R):
    for k in np.arange(K):
        if qN[k] > 0:
            x_k = np.array(x[q == k,:])
            mu[k,:] = next_mu(x_k, Sigma[k,:,:], mu0, Si0Inv, qN[k], R)
            iwDiagA[k,:] = next_iwDiagA(Sigma[k,:,:], nu, invASq, R)
            Sigma[k,:,:] = next_Sigma(x_k, mu[k,:], nu, iwDiagA[k,:], diagCov, qN[k], R)
        else:
            mu[k,:], Sigma[k,:,:], iwDiagA[k,:] = next_g0_k(mu0, Si0, nu, invASq, R)
        
    return mu, Sigma, iwDiagA

def gibbs(
        mcmc_iterBurn, mcmc_iterSample, method,
        x, K,
        mu0, Si0, nu, A, diagCov, alpha, s):
    
    ###
    #Precomputations
    ###
    
    nInd, R = x.shape
    
    ###
    #Storage
    ###
    
    mu_store = np.zeros((mcmc_iterSample, K, R))
    Sigma_store = np.zeros((mcmc_iterSample, K, R, R))
    pi_store = np.zeros((mcmc_iterSample, K))
    x_star_store = np.zeros((mcmc_iterSample, nInd, R))
    
    ###
    #MCMC
    ###
    
    invASq = np.ones((R,)) / A**2
    Si0Inv = np.linalg.inv(Si0)
    
    mu = np.zeros((K, R))
    Sigma = np.zeros((K, R, R))
    iwDiagA = np.zeros((K, R))
    for k in np.arange(K):
        mu[k,:], Sigma[k,:,:], iwDiagA[k,:] = next_g0_k(mu0, Si0, nu, invASq, R)
    if method == 'f':
        pi = np.random.dirichlet(alpha * np.ones((K,)))
    elif method == 'dp':
        eta = np.random.beta(1, alpha, (K - 1,))
        etaC = 1 - eta
        cumprodEtaC = np.cumprod(etaC)
        pi = np.ones((K,))
        pi[:-1] = eta
        pi[1:] *= cumprodEtaC
        pi /= pi.sum()
    else:
        assert False, 'Method not supported!'
    q, qN = next_q(x, mu, Sigma, pi, nInd, K)
    
    j = -1
    for i in np.arange(mcmc_iterBurn + mcmc_iterSample):
        #Sample
        alpha, pi = next_pi(alpha, qN, K, s, method)
        q, qN = next_q(x, mu, Sigma, pi, nInd, K)
        mu, Sigma, iwDiagA = next_theta(
                x, mu, Sigma, iwDiagA, mu0, Si0, Si0Inv, 
                nu, invASq, diagCov, 
                q, qN, nInd, K, R)
        
        #Evaluate density on grid
        x_star = monrnd(mu, Sigma, 
                        np.random.choice(np.arange(K), size = nInd, replace = True, p = pi))
        
        #Store
        if (i + 1) > mcmc_iterBurn: 
            j += 1
            mu_store[j,:,:] = mu
            Sigma_store[j,:,:,:] = Sigma
            pi_store[j,:] = pi
            x_star_store[j,:,:] = x_star
            
        #Display progess
        if ((i + 1) % 100) == 0:
            print('Iteration: ' + str(i + 1))
            
    return mu_store, Sigma_store, pi_store, x_star_store

###
#Density simulation
###
    
#Bivariate
def monpdf_bi_sim(mu_store, Sigma_store, pi_store, r_x, r_y, step):
    g_xx, g_yy = np.meshgrid(np.arange(r_x[0], r_x[1], step),
                             np.arange(r_y[0], r_y[1], step))
    g_x = g_xx.reshape((-1,))
    g_y = g_yy.reshape((-1,))
    g_xy = np.stack((g_x, g_y), axis = 1)
    z = 0
    
    nDraws = mu_store.shape[0]
    for i in np.arange(nDraws):
        z += monpdf(g_xy, mu_store[i,:,:], Sigma_store[i,:,:,:], pi_store[i,:])
        
        #Display progess
        if ((i + 1) % 100) == 0:
            print('Iteration: ' + str(i + 1))
    z /= nDraws
    g_zz = z.reshape((g_xx.shape[0], g_xx.shape[1]))
    
    return g_xx, g_yy, g_zz
    
###
#If main: test
###
    
if __name__ == "__main__":   
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from skewnormal import skewnormallogistic_rnd
    
    ###
    #Generate data
    ###
    
    
    nInd = 1000
    R = 2
    
    DGP = 2
    
    if DGP == 1:
        #DGP 1:
        nComp = 2
        
        #pComp = np.array([0.2, 0.1, 0.3, 0.3, 0.1])
        pComp = np.array([0.25, 0.25, 0.5])
        
        mu = [np.array([  1.3,  1.3, 0.3]),
              np.array([  0.3,  1.3, 0.8]),
              np.array([  0.8,  0.8, 0.3])]
        
        sd = [np.array([0.08, 0.08, 0.10]),
              np.array([0.08, 0.08, 0.10]),
              np.array([0.20, 0.20, 0.10])]
        
        Corr = [np.array([[1.0, 0.6, 0.2],
                          [0.6, 1.0, 0.3],
                          [0.2, 0.3, 1.0]]),
                np.array([[1.0, 0.2, 0.4],
                          [0.2, 1.0, 0.3],
                          [0.4, 0.3, 1.0]]),]
        idxCorr = np.array([0, 0, 1])
        
        Si = [np.diag(sd[i]) @ Corr[idxCorr[i]] @ np.diag(sd[i]) for i in np.arange(nComp)]
        ch = [np.linalg.cholesky(Si[i]) for i in np.arange(nComp)]
        
        q = np.random.choice(np.arange(nComp), size = nInd, replace = True, p = pComp)
        x = np.zeros((nInd, R))
        for n in np.arange(nInd):
            x[n,:] = mu[q[n]] + ch[q[n]] @ np.random.randn(R, )
            
    elif DGP == 2:
        #DGP 2:
        x0 = skewnormallogistic_rnd(0, 1, 50, nInd)
        x1 = skewnormallogistic_rnd(0, 1, 50, nInd)
        x = np.stack((x0, x1), axis = 1)

    elif DGP == 3:
        #DGP 3:
        nComp = 3
        
        pComp = np.array([0.25, 0.25, 0.5])
        
        loc = [np.array([ 1, -2]),
               np.array([-2, -2]),
               np.array([ 1,  1])]
        scale = [np.array([1,  1]),
                 np.array([1,  1]),
                 np.array([1,  1])]
        alpha = [np.array([ 40,  80]),
                 np.array([ 70,  70]),
                 np.array([-50, -50])]
        
        q = np.random.choice(np.arange(nComp), size = nInd, replace = True, p = pComp)
        x = np.zeros((nInd, R))
        for n in np.arange(nInd):
            for r in np.arange(R):
                x[n,r] = skewnormallogistic_rnd(loc[q[n]][r], scale[q[n]][r], alpha[q[n]][r])       
    
    ###
    #MCMC
    ###
    
    K = 100
    alpha = 1
    s = (2, 2)
    
    mu0 = np.zeros((R,))
    Si0 = 1 * np.eye(R)
    nu = 2
    A = 1.04
    diagCov = False
    
    mcmc_iterBurn = 5000
    mcmc_iterSample = 5000
    method = 'dp'

    tic = time.time()
    
    mu_store, Sigma_store, pi_store, x_star_store = gibbs(
        mcmc_iterBurn, mcmc_iterSample, method,
        x, K,
        mu0, Si0, nu, A, diagCov, alpha, s)
    
    toc = time.time() - tic
    print(toc)  
    
    ###
    #Plot: univariate
    ###
    
    fig = plt.figure()
    num_bins = 6 * 6
    rng = [(-3, 4), (-3, 2)]
    for i in np.arange(R):    
        ax = plt.subplot(2, R, i + 1)
        
        #True
        data = x[:,i]
        n, bins, patches = ax.hist(data, num_bins, rng[i], density = 1)
        

        #Estimated
        data = x_star_store[:,:,i].reshape((-1,))
        n, bins, patches = ax.hist(data, num_bins, rng[i], density = 1)

        ax.grid()
    plt.show()
    
    ###
    #Plot: bivariate
    ###
    
    r_x = (-2.5, 2.5)
    r_y = (-2.5, 2.5)
    step = 0.05
    
    g_xx, g_yy, g_zz = monpdf_bi_sim(mu_store, Sigma_store, pi_store, r_x, r_y, step)
    
    fig = plt.figure() 
    ax = plt.subplot()
    ax = sns.kdeplot(x[:,0], x[:,1], 
                cmap= "Blues" , shade = True, shade_lowest = False, linestyles = "-")
    ax.contour(g_xx, g_yy, g_zz, 10)
    plt.xlim(r_x); plt.ylim(r_y)
    ax.set(title = 'DP-MON', xlabel = 'X1', ylabel = 'X2')
    plt.show()

    
    
    