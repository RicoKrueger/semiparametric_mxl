import os
import numpy as np
import itertools
import scipy.sparse
import pickle

from mxl import prepareData, pPredMxl
from skewnormal import skewnormallogistic_rnd

###
#Set seed
###

np.random.seed(4711)

###
#True parameter values
###

nRnd = 2

#For scenario 2
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

###
#Generate semi-synthetic data
###

R = 20
S_list = [1, 2]
N_list = [1000]
T_list = [8, 16]
A = 5

N_val = 25
D = 1000; nTakes = 10; 

for S, N, T, r in itertools.product(S_list, N_list, T_list, np.arange(R)):
    print(" ")
    print("S = " + str(S) + "; " + 
          "N = " + str(N) + "; " + 
          "T = " + str(T) + "; " + 
          "r = " + str(r) + ";")
    
    T_val = 1
    NT_val = N_val * T_val
    NTA_val = NT_val * A
    
    N_tot = N + N_val
    
    obsPerInd_tot = np.concatenate((np.repeat(T, N),
                                    np.repeat(1, N_val)))
    obsPerInd_tot[:N_val] += 1
    rowsPerInd_tot = obsPerInd_tot * A
    rowsPerObs_tot = np.repeat(A, np.sum(obsPerInd_tot))
    
    NT_tot = np.sum(obsPerInd_tot)
    NTA_tot = np.sum(rowsPerObs_tot)
    
    indObsID_tot = np.zeros((NTA_tot,))
    u = 0
    for n in np.arange(N_tot):
        indObsID_ind = np.kron(np.arange(1, obsPerInd_tot[n] + 1).reshape((-1,1)), 
                               np.ones((A, 1))).reshape((-1,))
        l = u; u += rowsPerInd_tot[n];
        indObsID_tot[l:u] = indObsID_ind      
    
    #Generate attributes    
    xRnd_tot = -5 + 10 * np.random.rand(NTA_tot, nRnd)
    
    #Generate parameters   
    if S == 1:
        betaRndInd = skewnormallogistic_rnd(0, 1, 50, nRnd * N_tot).reshape((N_tot, nRnd))
    elif S == 2:
        betaRndInd = np.zeros((N_tot, nRnd))
        q = np.random.choice(np.arange(nComp), size = N_tot, replace = True, p = pComp)   
        for n, k in itertools.product(np.arange(N_tot), np.arange(nRnd)):
            betaRndInd[n,k] = skewnormallogistic_rnd(loc[q[n]][k], scale[q[n]][k], alpha[q[n]][k])  
    
    betaRndInd_perRow = np.repeat(betaRndInd, rowsPerInd_tot, axis = 0)    
        
    #Simulate choices
    chosen_tot = np.zeros((NTA_tot,), dtype = 'int64')
    
    eps = -np.log(-np.log(np.random.rand(NTA_tot,)))
    vDet = np.sum(xRnd_tot * betaRndInd_perRow, axis = 1)
    v = vDet + eps
    
    errorChoice = np.empty((NT_tot,), dtype = 'bool')
    
    for i in np.arange(NT_tot):
        sl = slice(i * A, (i + 1) * A)
        choiceDet = np.where(vDet[sl] == vDet[sl].max())
        choiceRnd = np.where(v[sl] == v[sl].max())
        errorChoice[i] = choiceRnd[0] not in choiceDet[0]
        altMax = np.random.choice(choiceRnd[0], size = 1)
        chosen_tot[i * A + altMax] = 1
        
    altID_tot = np.tile(np.arange(1, A + 1), (NT_tot,)) 
        
    error = np.sum(errorChoice) / NT_tot
    print("Error rate: " + str(error))
    
    chosenAlt_tot = chosen_tot * altID_tot
    chosenAlt_tot = np.array(chosenAlt_tot[chosen_tot == 1])
    _, ms = np.unique(chosenAlt_tot, return_counts = True)
    ms = ms / NT_tot
    print("Market shares :")
    print(ms)
    
    #Extract relevant data    
    indID_tot = np.repeat(np.arange(N_tot), rowsPerInd_tot)
    obsID_tot = np.repeat(np.arange(NT_tot), rowsPerObs_tot)

    idx = np.logical_and(indObsID_tot <= T, indID_tot < N) #Training data
    indID = np.array(indID_tot[idx])
    obsID = np.array(obsID_tot[idx])
    altID = np.array(altID_tot[idx])
    chosen = np.array(chosen_tot[idx])
    xRnd = np.array(xRnd_tot[idx,:])
    paramRnd_true = np.array(betaRndInd[:N,:])
    
    idx = indID_tot >= N #Between validation
    indID_valB = np.array(indID_tot[idx])
    obsID_valB = np.array(obsID_tot[idx])
    altID_valB = np.array(altID_tot[idx])
    chosen_valB = np.array(chosen_tot[idx])
    xRnd_valB = np.array(xRnd_tot[idx,:])
    
    idx = indObsID_tot > T #Within validation
    indID_valW = np.array(indID_tot[idx])
    obsID_valW = np.array(obsID_tot[idx])
    altID_valW = np.array(altID_tot[idx])
    chosen_valW = np.array(chosen_tot[idx])
    xRnd_valW = np.array(xRnd_tot[idx,:]) 
    
    #Calculate choice probabilities for the training sample 
    xList = [xRnd]
        
    (xList,
     _, _, _,
     chosenIdx, nonChosenIdx,
     rowsPerInd, _,
     _, map_avail_to_obs) = prepareData(xList, indID, obsID, chosen)
    xRnd_valDiff = xList[0]
    
    vFix = 0    
    betaRndInd_val = np.array(betaRndInd[:N,:])
    betaRndInd_perRow_val = np.repeat(betaRndInd_val, rowsPerInd, axis = 0)
    vRnd = np.sum(xRnd_valDiff * betaRndInd_perRow_val, axis = 1)
    pPredT_true = pPredMxl(vFix, vRnd, map_avail_to_obs, 1, chosenIdx, nonChosenIdx)  
   
    #Simulate predictive choice distributions for the between validation sample 
    xList = [xRnd_valB]
        
    (xList,
     _, _, _,
     chosenIdx, nonChosenIdx,
     rowsPerInd, _,
     _, map_avail_to_obs) = prepareData(xList, indID_valB, obsID_valB, chosen_valB)
    
    xRnd_valDiff = xList[0]
    xRnd_valDiff = np.tile(xRnd_valDiff, (D, 1))
    sim_map_avail_to_obs = scipy.sparse.kron(scipy.sparse.eye(D), map_avail_to_obs)
    sim_rowsPerInd = np.tile(rowsPerInd, D)
    rowsPerObs = np.squeeze(np.asarray(np.sum(map_avail_to_obs, axis = 0)))
    sim_rowsPerObs = np.tile(rowsPerObs, D)
    
    pPred_true = np.zeros((NTA_val,))
    
    vFix = 0
    
    for t in np.arange(nTakes): 
        if S == 1:
            betaRndInd_val = skewnormallogistic_rnd(0, 1, 50, nRnd * N_val * D).reshape((N_val * D, nRnd))
        elif S == 2:
            betaRndInd_val = np.zeros((N_val * D, nRnd))
            q = np.random.choice(np.arange(nComp), size = N_val * D, replace = True, p = pComp)   
            for n, k in itertools.product(np.arange(N_val * D), np.arange(nRnd)):
                betaRndInd_val[n,k] = skewnormallogistic_rnd(loc[q[n]][k], scale[q[n]][k], alpha[q[n]][k])       
  
        betaRndInd_perRow_val = np.repeat(betaRndInd_val, sim_rowsPerInd, axis = 0)
        
        vRnd = np.sum(xRnd_valDiff * betaRndInd_perRow_val, axis = 1)
        pPred_true_take = pPredMxl(vFix, vRnd, sim_map_avail_to_obs, D, chosenIdx, nonChosenIdx)
        pPred_true += pPred_true_take 
    pPred_true /= nTakes
    pPredB_true = np.array(pPred_true)

    #Calculate predictive choice distributions for the within validation sample 
    xList = [xRnd_valW]
        
    (xList,
     _, _, _,
     chosenIdx, nonChosenIdx,
     rowsPerInd, _,
     _, map_avail_to_obs) = prepareData(xList, indID_valW, obsID_valW, chosen_valW)
    xRnd_valDiff = xList[0]
    
    vFix = 0    
    betaRndInd_val = np.array(betaRndInd[:N_val,:])
    betaRndInd_perRow_val = np.repeat(betaRndInd_val, rowsPerInd, axis = 0)
    vRnd = np.sum(xRnd_valDiff * betaRndInd_perRow_val, axis = 1)
    pPredW_true = pPredMxl(vFix, vRnd, map_avail_to_obs, 1, chosenIdx, nonChosenIdx)  
    
    #Save data    
    sim = {'indID': indID, 'obsID': obsID, 
           'altID': altID, 'chosen': chosen,
           'xRnd': xRnd, 'nRnd': nRnd,
           'paramRnd_true': paramRnd_true,
           'indID_valB': indID_valB, 'obsID_valB': obsID_valB, 
           'altID_valB': altID_valB, 'chosen_valB': chosen_valB,
           'xRnd_valB': xRnd_valB,
           'indID_valW': indID_valW, 'obsID_valW': obsID_valW, 
           'altID_valW': altID_valW, 'chosen_valW': chosen_valW,
           'xRnd_valW': xRnd_valW,
           'pPredT_true': pPredT_true, 'pPredB_true': pPredB_true, 'pPredW_true': pPredW_true}
    
    filename = 'sim' + '_S' + str(S) + '_N' + str(N) + '_T' + str(T) + '_r' + str(r)
    if os.path.exists(filename): os.remove(filename) 
    outfile = open(filename, 'wb')
    pickle.dump(sim, outfile)
    outfile.close()