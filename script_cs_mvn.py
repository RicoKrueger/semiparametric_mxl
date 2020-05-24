import os
import numpy as np
import pandas as pd
import pickle

from cs_mvn_mxlMonMcmc import estimate, ppdB, ppdW
from metrics import waic

task = int(os.getenv('TASK'))

######
#Training
######

###
#Load data
###

data = pd.read_csv('data_train_s' + str(task) + '.csv')

###
#Prepare data
###

indID = np.array(data['person_id'].values, dtype = 'int64')
obsID = np.array(data['choice_situ_id'].values, dtype = 'int64')
altID = None

chosen = np.array(data['chosen_best'].values, dtype = 'int64')

ovtt = np.array(data['ovtt'].values) / 4
ivtt = np.array(data['ivtt'].values) / 4
cost = -np.array(data['totalcost'].values)
electric = np.array(data['electric'].values)
automation = np.array(data['automation'].values)
ASC_uber = np.array(data['ASC_uber'].values)
ASC_uberpool = np.array(data['ASC_uberpool'].values)

ivtt_automation = ivtt * automation
ivtt_pool = ivtt * ASC_uberpool
ASC_rs = ASC_uber + ASC_uberpool

###
#Estimate MXL via MCMC
###

xFix = np.zeros((0,0))
xFix_trans = np.array([0, 0, 0])

xRnd = np.stack((ASC_rs, cost, ovtt, ivtt, ivtt_pool, electric, automation), axis = 1)
xRnd_trans = np.array([0, 1, 0, 0, 0, 0, 0])

xRnd2 = np.zeros((0,0))
xRnd2_trans = np.array([0, 0, 0, 0, 0])

paramFix_inits = np.zeros((xFix.shape[1],))
zeta_inits = np.zeros((xRnd.shape[1],))
Omega_inits = 0.1 * np.eye(xRnd.shape[1])

mu0 = np.zeros((xRnd.shape[1],))
Si0 = np.eye(xRnd.shape[1])
A = 1.04
nu = 2
diagCov = True

method = 'f'
alpha = 5
K = 2
g0_mu0 = np.zeros((xRnd2.shape[1],))
g0_Si0 = np.eye(xRnd2.shape[1])
g0_nu = 2
g0_A = 1.04
g0_s = (2, 2)
diagCov2 = False 

mcmc_nChain = 2
mcmc_iterBurn = 50000
mcmc_iterSample = 50000
mcmc_thin = 10
mcmc_iterMem = 50000
mcmc_disp = 1000
seed = 4711
simDraws = 10000    

rho = 0.1
rhoF = 0.01

modelName = 'mvn' + '_cs' + '_s' + str(task)
deleteDraws = False

results = estimate(
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
        xFix_trans, xRnd_trans, xRnd2_trans)

###
#Prediction
###

mcmc_disp = 1000
deleteDraws = False

_, pPredT_chosen = ppdW(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_disp,
        seed,
        modelName, deleteDraws,
        indID, obsID, altID, chosen,
        xFix, xFix_trans,
        xRnd, xRnd_trans,
        xRnd2, xRnd2_trans)
waicT, lppdT = waic(pPredT_chosen)

######
#Prediction: between
######

###
#Load data
###

data = pd.read_csv('data_valB_s' + str(task) + '.csv')

###
#Prepare data
###

indID = np.array(data['person_id'].values, dtype = 'int64')
obsID = np.array(data['choice_situ_id'].values, dtype = 'int64')
altID = None

chosen = np.array(data['chosen_best'].values, dtype = 'int64')

ovtt = np.array(data['ovtt'].values) / 4
ivtt = np.array(data['ivtt'].values) / 4
cost = -np.array(data['totalcost'].values)
electric = np.array(data['electric'].values)
automation = np.array(data['automation'].values)
ASC_uber = np.array(data['ASC_uber'].values)
ASC_uberpool = np.array(data['ASC_uberpool'].values)

ivtt_automation = ivtt * automation
ivtt_pool = ivtt * ASC_uberpool
ASC_rs = ASC_uber + ASC_uberpool

###
#Prediction
###

xFix = np.zeros((0,0))
xRnd = np.stack((ASC_rs, cost, ovtt, ivtt, ivtt_pool, electric, automation), axis = 1)
xRnd2 = np.zeros((0,0))

nTakes = 20
nSim = 100

mcmc_disp = 100
deleteDraws = task > 0

_, pPredB_chosen = ppdB(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_disp, nTakes, nSim,
        seed,
        modelName, deleteDraws,
        indID, obsID, altID, chosen,
        xFix, xFix_trans,
        xRnd, xRnd_trans,
        xRnd2, xRnd2_trans)
waicB, lppdB = waic(pPredB_chosen)

######
#Save results
######

results['estimation_time_eff'] = results['estimation_time'] / mcmc_nChain
results['waicT'] = waicT; results['lppdT'] = lppdT; results['pPredT_chosen'] = pPredT_chosen.mean();
results['waicB'] = waicB; results['lppdB'] = lppdB; results['pPredB_chosen'] = pPredB_chosen.mean();
#results['waicW'] = waicW; results['lppdW'] = lppdW; results['pPredW_chosen'] = pPredW_chosen;

filename = 'results/' + modelName
if os.path.exists(filename): os.remove(filename) 
outfile = open(filename, 'wb')
pickle.dump(results, outfile)
outfile.close()