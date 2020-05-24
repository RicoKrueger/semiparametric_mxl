#from joblib import Parallel, delayed
from numba import jit
import os
import sys
import time
import numpy as np
import scipy.sparse
from math import floor
import h5py

from mxl import corrcov, prepareData, mvnlpdf, transFix, transRnd
from mon import monrnd, monlpdf_q, next_mu, next_Sigma, next_iwDiagA, \
next_pi, next_q, next_theta, next_g0_k

###
#Convenience

@jit
def monrnd_no_mu(Sigma, q):
    N = q.shape[0]; R = Sigma.shape[1];
    x = np.zeros((N, R))
    ch = np.linalg.cholesky(Sigma)
    for n in np.arange(N): 
        x[n,:] = ch[q[n],:,:] @ np.random.randn(R,)
    return x

###
#Probabilities
###

def probMxl(
        paramFix, paramRnd, paramRnd2,
        xFix, xFix_transBool, xFix_trans, nFix,
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        xRnd2, xRnd2_transBool, xRnd2_trans, nRnd2,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs):

    vFix = 0; vRnd = 0; vRnd2 = 0;
    if nFix > 0:
        if xFix_transBool: paramFix = transFix(paramFix, xFix_trans)
        vFix = xFix @ paramFix    
    if nRnd > 0:
        if xRnd_transBool: paramRnd = transRnd(paramRnd, xRnd_trans)
        paramRndPerRow = np.repeat(paramRnd, rowsPerInd, axis = 0)
        vRnd = np.sum(xRnd * paramRndPerRow, axis = 1) 
    if nRnd2 > 0:
        if xRnd2_transBool: paramRnd2 = transRnd(paramRnd2, xRnd_trans)
        paramRnd2PerRow = np.repeat(paramRnd2, rowsPerInd, axis = 0)
        vRnd2 = np.sum(xRnd2 * paramRnd2PerRow, axis = 1)            
    v = vFix + vRnd + vRnd2
    """
    if xRnd_transBool: paramRnd = transRnd(paramRnd, xRnd_trans)
    paramRndPerRow = np.repeat(paramRnd, rowsPerInd, axis = 0)
    if xRnd2_transBool: paramRnd2 = transRnd(paramRnd2, xRnd2_trans)
    paramRnd2PerRow = np.repeat(paramRnd2, rowsPerInd, axis = 0)
    
    v = xRnd[:,0] * paramRndPerRow[:,0] \
        -paramRndPerRow[:,1] * (xRnd[:,1] + np.sum(xRnd2[:,2:] * paramRnd2PerRow[:,2:], axis = 1))
    """
    
    ev = np.exp(v)
    ev[ev > 1e+200] = 1e+200
    ev[ev < 1e-300] = 1e-300 
    nev = map_avail_to_obs.T @ ev + 1
    pChosen = 1 / nev
    lPChosen = np.log(pChosen)
    lPInd = map_obs_to_ind.T @ lPChosen
    return lPInd
                
###
#MCMC
###
    
def next_paramFix(
        paramFix, paramRnd, paramRnd2,
        lPInd,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        xRnd2, xRnd2_transBool, xRnd2_trans, nRnd2,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs,
        rhoF):
    paramFix_star = paramFix + np.sqrt(rhoF) * np.random.randn(nFix,)
    lPInd_star = probMxl(
        paramFix_star, paramRnd, paramRnd2,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        xRnd2, xRnd2_transBool, xRnd2_trans, nRnd2,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs)
    r = np.exp(np.sum(lPInd_star - lPInd, axis = 0))
    if np.random.rand() <= r:
        paramFix = np.array(paramFix_star)
        lPInd = np.array(lPInd_star)
    return paramFix, lPInd

def next_paramRnd(
        paramFix, paramRnd, paramRnd2,
        zeta, Omega, zeta2, Omega2,
        lPInd,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        xRnd2, xRnd2_transBool, xRnd2_trans, nRnd2, pi, q,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs,
        rho):
    lPhi, lPhi2 = 0, 0
    lPhi_star, lPhi2_star = 0, 0
    paramRnd_star, paramRnd2_star = 0, 0
    if nRnd:
        paramRnd_star = paramRnd + np.sqrt(rho) * (np.linalg.cholesky(Omega) @ np.random.randn(nRnd, nInd)).T 
        lPhi = mvnlpdf(paramRnd, zeta, Omega)
        lPhi_star = mvnlpdf(paramRnd_star, zeta, Omega)
    if nRnd2:
        paramRnd2_star = paramRnd2 + np.sqrt(rho) * monrnd_no_mu(Omega2, q)
        lPhi2 = monlpdf_q(paramRnd2, zeta2, Omega2, q)
        lPhi2_star = monlpdf_q(paramRnd2_star, zeta2, Omega2, q)
    
    lPInd_star = probMxl(
        paramFix, paramRnd_star, paramRnd2_star,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        xRnd2, xRnd2_transBool, xRnd2_trans, nRnd2,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs)

    r = np.exp(lPInd_star + lPhi_star + lPhi2_star - lPInd - lPhi - lPhi2)
    idxAccept = np.random.rand(nInd,) <= r

    if nRnd: paramRnd[idxAccept, :] = np.array(paramRnd_star[idxAccept, :])
    if nRnd2: paramRnd2[idxAccept, :] = np.array(paramRnd2_star[idxAccept, :])
    lPInd[idxAccept] = np.array(lPInd_star[idxAccept])

    acceptRate = np.mean(idxAccept)
    rho = rho - 0.001 * (acceptRate < 0.3) + 0.001 * (acceptRate > 0.3)
    return paramRnd, paramRnd2, lPInd, rho, acceptRate

def mcmcChain(
        chainID, seed,
        mcmc_iter, mcmc_iterBurn, mcmc_iterSampleThin, mcmc_iterMemThin, mcmc_thin, mcmc_disp,
        rhoF, rho,
        modelName,
        method,
        paramFix, zeta, Omega, mu0, Si0Inv, invASq, nu, diagCov,
        alpha, K, g0_mu0, g0_Si0, g0_Si0Inv, g0_nu, g0_invASq, g0_s, diagCov2, 
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd,
        xRnd2, xRnd2_transBool, xRnd2_trans, nRnd2,
        nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs):   
    
    np.random.seed(seed + chainID)
    
    ###
    #Precomputations
    ###
    
    paramRnd = np.zeros((0,0))
    iwDiagA = np.zeros((0,0))
    if nRnd > 0:
        paramRnd = zeta + (np.linalg.cholesky(Omega) @ np.random.randn(nRnd, nInd)).T
        iwDiagA = np.random.gamma(1 / 2, 1 / invASq)

    zeta2 = None
    Omega2 = None
    iwDiagA2 = None
    pi = None
    q = None
    paramRnd2 = None
    if nRnd2:
        zeta2 = np.zeros((K, nRnd2))
        Omega2 = np.zeros((K, nRnd2, nRnd2))
        iwDiagA2 = np.zeros((K, nRnd2))
        for k in np.arange(K):
            zeta2[k,:], Omega2[k,:,:], iwDiagA2[k,:] = \
            next_g0_k(g0_mu0, g0_Si0, g0_nu, g0_invASq, nRnd2)
            Omega2[k,:,:] = 0.1 * np.eye(nRnd2)
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
        paramRnd2 = g0_mu0 + 2 * (np.linalg.cholesky(g0_Si0) @ np.random.randn(nRnd2, nInd)).T
        q, qN = next_q(paramRnd2, zeta2, Omega2, pi, nInd, K)
        #q = np.random.choice(np.arange(K), nInd); qN = compsize(q, K)
        #paramRnd2 = monrnd(zeta2, Omega2, q)
        
    lPInd = probMxl(
            paramFix, paramRnd, paramRnd2,
            xFix, xFix_transBool, xFix_trans, nFix, 
            xRnd, xRnd_transBool, xRnd_trans, nRnd,
            xRnd2, xRnd2_transBool, xRnd2_trans, nRnd2,
            nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs)   
    
    ###
    #Storage
    ###
    
    fileName = modelName + '_draws_chain' + str(chainID + 1) + '.hdf5'
    if os.path.exists(fileName):
        os.remove(fileName) 
    file = h5py.File(fileName, "a")
    
    if nFix > 0:
        paramFix_store = file.create_dataset('paramFix_store', (mcmc_iterSampleThin, nFix), dtype='float64')
        
        paramFix_store_tmp = np.zeros((mcmc_iterMemThin, nFix))
        
    if nRnd > 0:
        paramRnd_store = file.create_dataset('paramRnd_store', (mcmc_iterSampleThin, nInd, nRnd), dtype='float64')
        zeta_store = file.create_dataset('zeta_store', (mcmc_iterSampleThin, nRnd), dtype='float64')
        Omega_store = file.create_dataset('Omega_store', (mcmc_iterSampleThin, nRnd, nRnd), dtype='float64')
        Corr_store = file.create_dataset('Corr_store', (mcmc_iterSampleThin, nRnd, nRnd), dtype='float64')
        sd_store = file.create_dataset('sd_store', (mcmc_iterSampleThin, nRnd), dtype='float64')
        
        paramRnd_store_tmp = np.zeros((mcmc_iterMemThin, nInd, nRnd))
        zeta_store_tmp = np.zeros((mcmc_iterMemThin, nRnd))
        Omega_store_tmp = np.zeros((mcmc_iterMemThin, nRnd, nRnd))
        Corr_store_tmp = np.zeros((mcmc_iterMemThin, nRnd, nRnd))
        sd_store_tmp = np.zeros((mcmc_iterMemThin, nRnd))
        
    if nRnd2 > 0:
        paramRnd2_store = file.create_dataset('paramRnd2_store', (mcmc_iterSampleThin, nInd, nRnd2), dtype='float64')
        zeta2_store = file.create_dataset('zeta2_store', (mcmc_iterSampleThin, K, nRnd2), dtype='float64')
        Omega2_store = file.create_dataset('Omega2_store', (mcmc_iterSampleThin, K, nRnd2, nRnd2), dtype='float64')
        pi_store = file.create_dataset('pi_store', (mcmc_iterSampleThin, K), dtype='float64')
        
        paramRnd2_store_tmp = np.zeros((mcmc_iterMemThin, nInd, nRnd2))
        zeta2_store_tmp = np.zeros((mcmc_iterMemThin, K, nRnd2))
        Omega2_store_tmp = np.zeros((mcmc_iterMemThin, K, nRnd2, nRnd2))
        pi_store_tmp = np.zeros((mcmc_iterMemThin, K))        
    
    ###
    #Sample
    ###
    
    j = -1
    ll = 0
    acceptRate = 0 
    sampleState = 'burn in'
    for i in np.arange(mcmc_iter):
        if nFix > 0:
            paramFix, lPInd = next_paramFix(
                    paramFix, paramRnd, paramRnd2,
                    lPInd,
                    xFix, xFix_transBool, xFix_trans, nFix, 
                    xRnd, xRnd_transBool, xRnd_trans, nRnd,
                    xRnd2, xRnd2_transBool, xRnd2_trans, nRnd2,
                    nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs,
                    rhoF)
            
        if nRnd or nRnd2:
            paramRnd, paramRnd2, lPInd, rho, acceptRateIter = next_paramRnd(
                    paramFix, paramRnd, paramRnd2,
                    zeta, Omega, zeta2, Omega2,
                    lPInd,
                    xFix, xFix_transBool, xFix_trans, nFix, 
                    xRnd, xRnd_transBool, xRnd_trans, nRnd,
                    xRnd2, xRnd2_transBool, xRnd2_trans, nRnd2, pi, q,
                    nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs,
                    rho)
            acceptRate += acceptRateIter            
            
        if nRnd > 0:
            zeta = next_mu(paramRnd, Omega, mu0, Si0Inv, nInd, nRnd)
            Omega = next_Sigma(paramRnd, zeta, nu, iwDiagA, diagCov, nInd, nRnd)
            iwDiagA = next_iwDiagA(Omega, nu, invASq, nRnd)
            
        if nRnd2 > 0:
            alpha, pi = next_pi(alpha, qN, K, g0_s, method)
            q, qN = next_q(paramRnd2, zeta2, Omega2, pi, nInd, K)
            zeta2, Omega2, iwDiagA2 = next_theta(
                paramRnd2, zeta2, Omega2, iwDiagA2, g0_mu0, g0_Si0, g0_Si0Inv, 
                g0_nu, g0_invASq, diagCov2, 
                q, qN, nInd, K, nRnd2)
        
        if ((i + 1) % mcmc_disp) == 0:
            if (i + 1) > mcmc_iterBurn:
                sampleState = 'sampling'
            acceptRate /= mcmc_disp
            print('Chain ' + str(chainID + 1) + '; iteration: ' + str(i + 1) + ' (' + sampleState + '); '
                  'avg. accept rate: ' + str(np.round(acceptRate, 3)))
            acceptRate = 0
            sys.stdout.flush()
            
        if (i + 1) > mcmc_iterBurn:   
            if ((i + 1) % mcmc_thin) == 0:
                j+=1
            
                if nFix > 0:
                    paramFix_store_tmp[j,:] = paramFix
            
                if nRnd > 0:
                    paramRnd_store_tmp[j,:,:] = paramRnd
                    zeta_store_tmp[j,:] = zeta
                    Omega_store_tmp[j,:,:] = Omega
                    Corr_store_tmp[j,:,:], sd_store_tmp[j,:,] = corrcov(Omega)
                    
                if nRnd2 > 0:
                    paramRnd2_store_tmp[j,:,:] = paramRnd2
                    zeta2_store_tmp[j,:,:] = zeta2
                    Omega2_store_tmp[j,:,:,:] = Omega2
                    pi_store_tmp[j,:] = pi        
                    
            if (j + 1) == mcmc_iterMemThin:
                l = ll; ll += mcmc_iterMemThin; sl = slice(l, ll)
                
                print('Storing chain ' + str(chainID + 1))
                sys.stdout.flush()
                
                if nFix > 0:
                    paramFix_store[sl,:] = paramFix_store_tmp
                    
                if nRnd > 0:
                    paramRnd_store[sl,:,:] = paramRnd_store_tmp
                    zeta_store[sl,:] = zeta_store_tmp
                    Omega_store[sl,:,:] = Omega_store_tmp
                    Corr_store[sl,:,:] = Corr_store_tmp
                    sd_store[sl,:,] = sd_store_tmp
                    
                if nRnd2 > 0:
                    paramRnd2_store[sl,:,:] = paramRnd2_store_tmp
                    zeta2_store[sl,:,:] = zeta2_store_tmp
                    Omega2_store[sl,:,:,:] = Omega2_store_tmp
                    pi_store[sl,:] = pi_store_tmp                 
                
                j = -1 

###
#Posterior analysis
###  

def extractPost(paramName, size, mcmc_nChain, mcmc_iterSampleThin, modelName):    
    postDraws = np.zeros((mcmc_nChain, mcmc_iterSampleThin, size[0], size[1]))
    for c in range(mcmc_nChain):
        file = h5py.File(modelName + '_draws_chain' + str(c + 1) + '.hdf5', 'r')
        postDraws[c,:,:,:] = np.array(file[paramName + '_store']).reshape((mcmc_iterSampleThin, size[0], size[1])) 
    return postDraws             

###
#Estimate
###
    
def estimate(
        mcmc_nChain, mcmc_iterBurn, mcmc_iterSample, mcmc_thin, mcmc_iterMem, mcmc_disp, 
        seed, simDraws,
        rhoF, rho,
        modelName, deleteDraws,
        method,
        mu0, Si0, A, nu, diagCov,
        alpha, K, g0_mu0, g0_Si0, g0_nu, g0_A, g0_s, diagCov2, 
        paramFix_inits, zeta_inits, Omega_inits,
        indID, obsID, altID, chosen,
        xFix, xRnd, xRnd2,
        xFix_trans, xRnd_trans, xRnd2_trans):
    ###
    #Prepare data
    ###
    
    nFix = xFix.shape[1]
    nRnd = xRnd.shape[1]
    nRnd2 = xRnd2.shape[1]
    
    xFix_transBool = np.sum(xFix_trans) > 0
    xRnd_transBool = np.sum(xRnd_trans) > 0
    xRnd2_transBool = np.sum(xRnd_trans) > 0
    
    xList = [xFix, xRnd, xRnd2]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     map_obs_to_ind, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xFix, xRnd, xRnd2 = xList[0], xList[1], xList[2]
    
    ### 
    #Posterior sampling
    ###
    
    mcmc_iter = mcmc_iterBurn + mcmc_iterSample
    mcmc_iterSampleThin = floor(mcmc_iterSample / mcmc_thin)
    mcmc_iterMemThin = floor(mcmc_iterMem / mcmc_thin)
    
    Si0Inv = np.linalg.inv(Si0)
    invASq = np.ones((nRnd,)) * A ** (-2)
    
    g0_Si0Inv = np.linalg.inv(g0_Si0)
    g0_invASq = np.ones((nRnd2,)) * g0_A ** (-2)
    
    paramFix = paramFix_inits
    zeta = zeta_inits
    Omega = Omega_inits
    
    tic = time.time()
    
    for c in range(mcmc_nChain):
        mcmcChain(
                c, seed,
                mcmc_iter, mcmc_iterBurn, mcmc_iterSampleThin, mcmc_iterMemThin, mcmc_thin, mcmc_disp,
                rhoF, rho,
                modelName,
                method,
                paramFix, zeta, Omega, mu0, Si0Inv, invASq, nu, diagCov,
                alpha, K, g0_mu0, g0_Si0, g0_Si0Inv, g0_nu, g0_invASq, g0_s, diagCov2, 
                xFix, xFix_transBool, xFix_trans, nFix, 
                xRnd, xRnd_transBool, xRnd_trans, nRnd,
                xRnd2, xRnd2_transBool, xRnd2_trans, nRnd2,
                nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs) 
    """
    Parallel(n_jobs = mcmc_nChain)(delayed(mcmcChain)(
                c, seed,
                mcmc_iter, mcmc_iterBurn, mcmc_iterSampleThin, mcmc_iterMemThin, mcmc_thin, mcmc_disp,
                rhoF, rho,
                modelName,
                method,
                paramFix, zeta, Omega, mu0, Si0Inv, invASq, nu, diagCov,
                alpha, K, g0_mu0, g0_Si0, g0_Si0Inv, g0_nu, g0_invASq, g0_s, diagCov2, 
                xFix, xFix_transBool, xFix_trans, nFix, 
                xRnd, xRnd_transBool, xRnd_trans, nRnd,
                xRnd2, xRnd2_transBool, xRnd2_trans, nRnd2,
                nInd, rowsPerInd, map_obs_to_ind, map_avail_to_obs) 
    for c in range(mcmc_nChain))
    """
    toc = time.time() - tic
    
    print(' ')
    print('Computation time [s]: ' + str(toc))
       
    """
    ###
    #Posterior analysis
    ###
    
    post_paramFix = None
    post_paramRnd = None
    post_paramRnd2 = None
    post_pi = None
    if nFix: post_paramFix = extractPost('paramFix', (nFix,1), mcmc_nChain, mcmc_iterSampleThin, modelName)
    if nRnd: post_paramRnd = extractPost('paramRnd', (nInd, nRnd), mcmc_nChain, mcmc_iterSampleThin, modelName)
    if nRnd2: 
        post_paramRnd2 = extractPost('paramRnd2', (nInd, nRnd2), mcmc_nChain, mcmc_iterSampleThin, modelName)
        post_pi = extractPost('pi', (K, 1), mcmc_nChain, mcmc_iterSampleThin, modelName)
    """
    
    ###
    #Delete draws
    ###
    
    if deleteDraws:
        for c in range(mcmc_nChain):
            os.remove(modelName + '_draws_chain' + str(c + 1) + '.hdf5') 
        
    ###
    #Save results
    ###
    
    results = {'modelName': modelName, 'seed': seed,
               'estimation_time': toc
               }
    
    return results

###
#Prediction: between
###
    
def pPredMxl(v, sim_map_avail_to_obs, D, chosenIdx, nonChosenIdx):
    ev = np.exp(v)
    ev[ev > 1e+200] = 1e+200
    ev[ev < 1e-300] = 1e-300 
    nev = sim_map_avail_to_obs.T @ ev + 1
    nnev = sim_map_avail_to_obs @ nev
    pChosen = 1 / nev
    pNonChosen = ev / nnev
    pPredChosen = pChosen.reshape((D, -1)).mean(axis = 0)
    pPredNonChosen = pNonChosen.reshape((D, -1)).mean(axis = 0)
    pPred = np.zeros((chosenIdx.shape[0] + nonChosenIdx.shape[0]))
    pPred[chosenIdx] = pPredChosen
    pPred[nonChosenIdx] = pPredNonChosen
    return pPred

    
def mcmcChainPredB(
        chainID, seed,
        mcmc_iterSampleThin, mcmc_disp, nTakes, nSim,
        modelName,
        xFix, xFix_transBool, xFix_trans, nFix, 
        sim_xRnd, xRnd_transBool, xRnd_trans, nRnd,
        sim_xRnd2, xRnd2_transBool, xRnd2_trans, nRnd2, 
        nInd, nObs, nRow,
        sim_rowsPerInd, sim_map_avail_to_obs, chosenIdx, nonChosenIdx):   
    
    np.random.seed(seed + chainID)
    
    ###
    #Retrieve draws
    ###
    
    fileName = modelName + '_draws_chain' + str(chainID + 1) + '.hdf5'
    file = h5py.File(fileName, "r")
    
    paramFix_store = None; 
    zeta_store = None; Omega_store = None;
    zeta2_store = None; Omega2_store = None; pi_store = None; nComp = None;
    if nFix: 
        paramFix_store = np.array(file['paramFix_store'])
    if nRnd:
        zeta_store = np.array(file['zeta_store'])
        Omega_store = np.array(file['Omega_store'])
    if nRnd2:
        zeta2_store = np.array(file['zeta2_store'])
        Omega2_store = np.array(file['Omega2_store'])
        pi_store = np.array(file['pi_store'])
        nComp = pi_store.shape[1]
    
    ###
    #Simulate
    ###

    pPred = np.zeros((mcmc_iterSampleThin, nRow + nObs))
    vFix = 0; vRnd = 0; vRnd2 = 0;
    
    for i in np.arange(mcmc_iterSampleThin):
        
        if nFix: 
            paramFix = paramFix_store[i,:]
            if xFix_transBool: paramFix = transFix(paramFix, xFix_trans)
            vFix = np.tile(xFix @ paramFix, (nSim,));
        if nRnd:
            zeta_tmp = zeta_store[i,:]
            ch_tmp = np.linalg.cholesky(Omega_store[i,:,:])
        if nRnd2:
            zeta2_tmp = zeta2_store[i,:,:]
            Omega2_tmp = Omega2_store[i,:,:,:]
            pi_tmp = pi_store[i,:]
        
        pPred_take = 0
        
        for t in np.arange(nTakes):
            if nRnd:
                paramRnd = zeta_tmp + (ch_tmp @ np.random.randn(nRnd, nInd * nSim)).T
                if xRnd_transBool: paramRnd = transRnd(paramRnd, xRnd_trans)
                paramRndPerRow = np.repeat(paramRnd, sim_rowsPerInd, axis = 0)
                vRnd = np.sum(sim_xRnd * paramRndPerRow, axis = 1)
            if nRnd2:
                q = np.random.choice(np.arange(nComp), size = nInd * nSim, replace = True, p = pi_tmp)
                paramRnd2 = monrnd(zeta2_tmp, Omega2_tmp, q)
                if xRnd2_transBool: paramRnd2 = transRnd(paramRnd2, xRnd2_trans)
                paramRnd2PerRow = np.repeat(paramRnd2, sim_rowsPerInd, axis = 0)
                vRnd2 = np.sum(sim_xRnd2 * paramRnd2PerRow, axis = 1)
            v = vFix + vRnd + vRnd2
            pPred_take += pPredMxl(v, sim_map_avail_to_obs, nSim, chosenIdx, nonChosenIdx)
            
        pPred[i,:] = pPred_take / nTakes
        
        if ((i + 1) % mcmc_disp) == 0:
            print('Chain ' + str(chainID + 1) + '; iteration: ' + str(i + 1) + ' (predictive simulation)')
            sys.stdout.flush()
    return pPred
    
def ppdB(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_disp, nTakes, nSim,
        seed,
        modelName, deleteDraws,
        indID, obsID, altID, chosen,
        xFix, xFix_trans, 
        xRnd, xRnd_trans,
        xRnd2, xRnd2_trans):
    ###
    #Prepare data
    ###
    
    nFix = xFix.shape[1]
    nRnd = xRnd.shape[1]
    nRnd2 = xRnd2.shape[1]
    
    xFix_transBool = np.sum(xFix_trans) > 0
    xRnd_transBool = np.sum(xRnd_trans) > 0
    xRnd2_transBool = np.sum(xRnd_trans) > 0
    
    xList = [xFix, xRnd, xRnd2]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     _, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xFix, xRnd, xRnd2 = xList[0], xList[1], xList[2]
    
    sim_xRnd = np.tile(xRnd, (nSim, 1)); 
    sim_xRnd2 = np.tile(xRnd2, (nSim, 1));
    sim_rowsPerInd = np.tile(rowsPerInd, (nSim,))
    sim_map_avail_to_obs = scipy.sparse.kron(scipy.sparse.eye(nSim), map_avail_to_obs)
    
    ### 
    #Predictive simulation
    ###
    
    mcmc_iterSampleThin = floor(mcmc_iterSample / mcmc_thin)
    
    pPred = np.zeros((mcmc_nChain, mcmc_iterSampleThin, nObs + nRow))
    for c in np.arange(mcmc_nChain):
        pPred[c,:,:] = mcmcChainPredB(
                c, seed,
                mcmc_iterSampleThin, mcmc_disp, nTakes, nSim,
                modelName,
                xFix, xFix_transBool, xFix_trans, nFix, 
                sim_xRnd, xRnd_transBool, xRnd_trans, nRnd,
                sim_xRnd2, xRnd2_transBool, xRnd2_trans, nRnd2,
                nInd, nObs, nRow,
                sim_rowsPerInd, sim_map_avail_to_obs, chosenIdx, nonChosenIdx)
        
    pPred_chosen = pPred[:,:,chosenIdx]
    
    ###
    #Delete draws
    ###
    
    if deleteDraws:
        for c in range(mcmc_nChain):
            os.remove(modelName + '_draws_chain' + str(c + 1) + '.hdf5') 

    return pPred, pPred_chosen

###
#Prediction: within
###
    
def mcmcChainPredW(
        chainID, seed,
        mcmc_iterSampleThin, mcmc_disp,
        modelName,
        xFix, xFix_transBool, xFix_trans, nFix, 
        xRnd, xRnd_transBool, xRnd_trans, nRnd, 
        xRnd2, xRnd2_transBool, xRnd2_trans, nRnd2, 
        nInd, nObs, nRow,
        rowsPerInd, map_avail_to_obs, chosenIdx, nonChosenIdx):   
    
    np.random.seed(seed + chainID)
    
    ###
    #Retrieve draws
    ###
    
    fileName = modelName + '_draws_chain' + str(chainID + 1) + '.hdf5'
    file = h5py.File(fileName, "r")
    
    paramFix_store = None
    paramRnd_store = None
    paramRnd2_store = None
    if nFix: paramFix_store = np.array(file['paramFix_store'])
    if nRnd: paramRnd_store = np.array(file['paramRnd_store'][:,:nInd,:])
    if nRnd2: paramRnd2_store = np.array(file['paramRnd2_store'][:,:nInd,:])
    
    ###
    #Simulate
    ###

    pPred = np.zeros((mcmc_iterSampleThin, nRow + nObs))
    vFix = 0; vRnd = 0; vRnd2 = 0;
    
    for i in np.arange(mcmc_iterSampleThin):
        if nFix: 
            paramFix = np.array(paramFix_store[i,:])
            if xFix_transBool: paramFix = transFix(paramFix, xFix_trans)
            vFix = xFix @ paramFix
        if nRnd:
            paramRnd = paramRnd_store[i,:,:]
            if xRnd_transBool: paramRnd = transRnd(paramRnd, xRnd_trans)
            paramRndPerRow = np.repeat(paramRnd, rowsPerInd, axis = 0)
            vRnd = np.sum(xRnd * paramRndPerRow, axis = 1)
        if nRnd2:
            paramRnd2 = paramRnd2_store[i,:,:]
            if xRnd2_transBool: paramRnd2 = transRnd(paramRnd2, xRnd2_trans)
            paramRnd2PerRow = np.repeat(paramRnd2, rowsPerInd, axis = 0)
            vRnd2 = np.sum(xRnd2 * paramRnd2PerRow, axis = 1)
        v = vFix + vRnd + vRnd2
        pPred[i,:] = pPredMxl(v, map_avail_to_obs, 1, chosenIdx, nonChosenIdx)
        
        if ((i + 1) % mcmc_disp) == 0:
            print('Chain ' + str(chainID + 1) + '; iteration: ' + str(i + 1) + ' (predictive simulation)')
            sys.stdout.flush()
    return pPred
    
def ppdW(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_disp,
        seed,
        modelName, deleteDraws,
        indID, obsID, altID, chosen,
        xFix, xFix_trans, 
        xRnd, xRnd_trans,
        xRnd2, xRnd2_trans):
    ###
    #Prepare data
    ###
    
    nFix = xFix.shape[1]
    nRnd = xRnd.shape[1]
    nRnd2 = xRnd2.shape[1]
    
    xFix_transBool = np.sum(xFix_trans) > 0
    xRnd_transBool = np.sum(xRnd_trans) > 0
    xRnd2_transBool = np.sum(xRnd_trans) > 0
    
    xList = [xFix, xRnd, xRnd2]
    (xList,
     nInd, nObs, nRow,
     chosenIdx, nonChosenIdx,
     rowsPerInd, rowsPerObs,
     _, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xFix, xRnd, xRnd2 = xList[0], xList[1], xList[2]
    
    ### 
    #Predictive simulation
    ###
    
    mcmc_iterSampleThin = floor(mcmc_iterSample / mcmc_thin)
    
    pPred = np.zeros((mcmc_nChain, mcmc_iterSampleThin, nObs + nRow))
    for c in np.arange(mcmc_nChain):
        pPred[c,:,:] = mcmcChainPredW(
                c, seed,
                mcmc_iterSampleThin, mcmc_disp,
                modelName,
                xFix, xFix_transBool, xFix_trans, nFix, 
                xRnd, xRnd_transBool, xRnd_trans, nRnd, 
                xRnd2, xRnd2_transBool, xRnd2_trans, nRnd2, 
                nInd, nObs, nRow,
                rowsPerInd, map_avail_to_obs, chosenIdx, nonChosenIdx)
    
    pPred_chosen = pPred[:,:,chosenIdx]
    
    ###
    #Delete draws
    ###
    
    if deleteDraws:
        for c in range(mcmc_nChain):
            os.remove(modelName + '_draws_chain' + str(c + 1) + '.hdf5') 

    return pPred, pPred_chosen