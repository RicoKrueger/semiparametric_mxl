import os
import numpy as np
import pickle

from mxlMonMcmc import estimate, ppdB, ppdW
from metrics import waic, tvd

###
#Obtain task
###

task = int(os.getenv('TASK'))

filename = "taskplan"
infile = open(filename, 'rb')
taskplan = pickle.load(infile)
infile.close()

S = taskplan[task, 0]
N = taskplan[task, 1]
T = taskplan[task, 2]
r = taskplan[task, 3]

###
#Load data
###

filename = 'sim' + '_S' + str(S) + '_N' + str(N) + '_T' + str(T) + '_r' + str(r)
infile = open(filename, 'rb')
sim = pickle.load(infile)
infile.close()

locals().update(sim)

###
#Estimate MXL via MCMC
###

xFix = np.zeros((0,0))
xFix_valB = np.zeros((0,0))
xFix_valW = np.zeros((0,0))

#Fixed parameter distributions
#0: normal
#1: log-normal (to assure that fixed parameter is striclty negative or positive)
xFix_trans = np.array([0, 0, 0])

#Random parameter distributions
#0: normal
#1: log-normal
#2: S_B
xRnd_trans = np.array([0, 0, 0, 0])

xRnd2 = np.array(xRnd)
xRnd2_valB = np.array(xRnd_valB)
xRnd2_valW = np.array(xRnd_valW)
xRnd2_trans = np.array([0, 0, 0, 0])

xRnd = np.zeros((0,0))
xRnd_valB = np.zeros((0,0))
xRnd_valW = np.zeros((0,0))

paramFix_inits = np.zeros((xFix.shape[1],))
zeta_inits = np.zeros((xRnd.shape[1],))
Omega_inits = 0.1 * np.eye(xRnd.shape[1])

mu0 = np.zeros((xRnd.shape[1],))
Si0 = np.eye(xRnd.shape[1])
A = 1.04
nu = 2
diagCov = False

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

modelName = 'f' + '_sim' + '_S' + str(S) + '_N' + str(N) + '_T' + str(T) + '_r' + str(r)
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
#Prediction: training
###

mcmc_disp = 1000
deleteDraws = False

pPredT, pPredT_chosen = ppdW(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_disp,
        seed,
        modelName, deleteDraws,
        indID, obsID, altID, chosen,
        xFix, xFix_trans,
        xRnd, xRnd_trans,
        xRnd2, xRnd2_trans)
waicT, lppdT = waic(pPredT_chosen)
tvdT = tvd(pPredT, pPredT_true)

###
#Prediction: between
###

nTakes = 2
nSim = 1000

mcmc_disp = 100
deleteDraws = False

pPredB, pPredB_chosen = ppdB(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_disp, nTakes, nSim,
        seed,
        modelName, deleteDraws,
        indID_valB, obsID_valB, altID_valB, chosen_valB,
        xFix_valB, xFix_trans,
        xRnd_valB, xRnd_trans,
        xRnd2_valB, xRnd2_trans)
waicB, lppdB = waic(pPredB_chosen)
tvdB = tvd(pPredB, pPredB_true)

###
#Prediction: within
###

mcmc_disp = 1000
deleteDraws = not((T == 8) and (r == 0))

pPredW, pPredW_chosen = ppdW(
        mcmc_nChain, mcmc_iterSample, mcmc_thin, mcmc_disp,
        seed,
        modelName, deleteDraws,
        indID_valW, obsID_valW, altID_valW, chosen_valW,
        xFix_valW, xFix_trans,
        xRnd_valW, xRnd_trans,
        xRnd2_valW, xRnd2_trans)
waicW, lppdW = waic(pPredW_chosen)
tvdW = tvd(pPredW, pPredW_true)

###
#Save results
###

res = np.zeros((10,))

res[0] = results['estimation_time'] / mcmc_nChain

res[1] = lppdT
res[2] = waicT
res[3] = tvdT

res[4] = lppdB
res[5] = waicB
res[6] = tvdB

res[7] = lppdW
res[8] = waicW
res[9] = tvdW

resList = [res, results]

filename = 'results/' + modelName
if os.path.exists(filename): os.remove(filename) 
outfile = open(filename, 'wb')
pickle.dump(resList, outfile)
outfile.close()