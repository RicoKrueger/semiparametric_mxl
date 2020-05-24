import numpy as np

###
#TVD
###

def tvd(pred, true):
    return np.mean(0.5 * np.absolute(true - pred.mean(axis = (0,1))))

###
#Model comparison
###

def waic(p):   
    lppd = np.sum(np.log(np.mean(p, axis = (0, 1))))
    v = np.sum(np.var(np.log(p), axis = (0, 1)))
    waic = lppd - v
    return waic, lppd