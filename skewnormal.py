import numpy as np
#import time
from scipy.stats import norm
import matplotlib.pyplot as plt

def adaptive_rejection(G, loc, scale):
    M = 2 / scale
    while True:
        y = loc + scale * np.random.randn()  
        u = np.random.rand()           
        if u * M < G(y): break
    return y

def skewnormal_rnd(G, loc, scale, alpha, N):    
    r = np.zeros((N,))
    for n in np.arange(N):
        r[n] = adaptive_rejection(G, loc, scale)
    return r

def skewnormalnormal_rnd(loc, scale, alpha, N = 1): 
    G = lambda x: norm.cdf(alpha * (x - loc) / scale)
    r = skewnormal_rnd(G, loc, scale, alpha, N)
    return r

def skewnormallogistic_rnd(loc, scale, alpha, N = 1):    
    G = lambda x: 1 / (1 + np.exp(-(alpha * (x - loc))))
    r = skewnormal_rnd(G, loc, scale, alpha, N)
    return r

###
#If main: test
###
    
if __name__ == "__main__":   

    alpha = 4
    loc = -2
    scale = 2
    N = 1000
    
    #tic = time.time()
    x = skewnormallogistic_rnd(loc, scale, alpha, N)
    #toc = time.time() - tic
    #print(toc)
    
    fig = plt.figure()
    b = 50
    ax = plt.subplot(1, 1, 1)
    
    #True
    data = x
    lower = np.min(data)
    upper = np.max(data)
    rng = upper - lower
    lower -= 0.2 * rng
    upper += 0.2 * rng
    
    n, bins, patches = ax.hist(data, b, (lower, upper), density = 1)
    #n, bins, patches = ax.hist(skewnorm.rvs(alpha, loc, scale, N), b, (lower, upper), density = 1)
    
    ax.grid()
    plt.xlim((lower, upper))
    plt.show()
    #fig.savefig("figure.pdf", dpi = 300)
    