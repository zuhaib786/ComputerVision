import numpy as np
# def Kernel(x, mu,sigma):
#     assert(x.shape == mu.shape and x.shape == sigma.shape)
#     const = 1/np.sqrt(2* np.pi)
#     p = const * (1/sigma) * np.exp(-0.5 *((x - mu)**2)/(sigma**2))
#     return p[:,:, 0]*p[:, :, 1]*p[:, :, 2]  

def Kernel(x, mu,sigma):
    assert(x.shape == mu.shape and x.shape == sigma.shape)
    const = 1/np.sqrt(2* np.pi)
    p = const * (1/sigma) * np.exp(-0.5 *((x - mu)**2)/(sigma**2))
    return p[:, :, 0]


def ExpAvgProbability(x, prevframes, sigma, alpha):
    
    const = 0 
    l = len(prevframes)
    prob = np.zeros((sigma.shape[0], sigma.shape[1]))
    for idx , frame in enumerate(prevframes):
        mul = pow(alpha, len(prevframes) - 1 - idx)
        prob += mul * Kernel(x, frame, sigma)
        const += mul
    assert(const > 0)
    return 1/const * prob
# def ExpAvgProbability(x, prevframes, sigma, alpha):
#     prob = np.zeros((sigma.shape[0], sigma.shape[1]))
#     for frame in prevframes:
#         prob  = prob + alpha * (Kernel(x, frame, sigma) - prob)
#     return  prob

def EqWtProbability(x, pixel_intensities, sigma):
    N = len(pixel_intensities)
    prob = 0
    for intensity in pixel_intensities:
        prob += Kernel(x, intensity, sigma)
    prob /= N
    return prob

def ExpAvgProbabilityConstSigma(x, prevAvg, intensity, sigma, alpha):
    return (1 - alpha) * Kernel(x, intensity, sigma) + alpha * prevAvg



def EqWtProbabilityConstSigma(x, prevAvg, intensity, N,sigma):
    return prevAvg + 1/(N + 1) *(Kernel(x, intensity, sigma) - prevAvg)   
