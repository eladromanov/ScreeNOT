"""
Implementation of the ScreeNOT procedure.
The function adaptiveHardThresholding performs adaptive singular value thresholding
on its input; from a user's perspecive, it is the main (only) API for this
code package.

Project homepage: https://github.com/eladromanov/ScreeNOT

Copyright (c) 2020 Elad Romanov
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from numpy.linalg import svd

"""
Performs optimal adaptive hard thresholding on the matrix Y
Parameters:
    Y: the measured matrix
    k: an upper bound on the signal rank. 
    strategy: method to reconstruct the noise bulk. 
        one of '0' (tranpsort to zero), 'w' (winsorization), 'i' (imputation)
        default='i'
        
Returns:
    Xest: an estimate of the low rank signal
    Topt: the hard threshold used
    r: the number of relevant components, r = rank(Xest)
"""
def adaptiveHardThresholding(Y, k, strategy='i'):
    
    U, fY, Vt = svd(Y, full_matrices=False)
    
    gamma = np.min( [ Y.shape[0]/Y.shape[1], Y.shape[1]/Y.shape[0] ] )
    
    fZ = createPseudoNoise(fY, k, strategy=strategy)
    Topt = computeOptThreshold(fZ, gamma)
    
    fY_new = fY*(fY>Topt)
    Xest = U @ np.diag(fY_new) @ Vt
    r = np.sum(fY_new>0)
    
    return Xest, Topt, r
    
    


"""
Creates a "pseudo-noise" singular from the singular values of the observed matrix 
Y, which is given in the array fY. 
Parameters:
    fY: a numpy array containing the observed singular values, one which we operate
    k: an upper bound on the signal rank. The leading k values in fY are discarded
    strategy: one of '0' (tranpsort to zero), 'w' (winsorization), 'i' (imputation)
        default='i'
"""
def createPseudoNoise(fY, k, strategy='i'):
    
    # sort fZ into increasing order
    fZ = np.sort(fY)
    
    p = fZ.size
    if k >= p:
        raise ValueError('k too large. procedure requires k < min(n,p)')
    # assert( k+1 <= p )
    
    if k > 0:
        if strategy == '0': # transport to zero
            fZ[-k:] = 0
        elif strategy == 'w': # winsorization
            fZ[-k:] = fZ[-k-1]  
        elif strategy == 'i':   # imputation
            # assert(2*k+1 < p)
            if 2*k+1 >= p:
                raise ValueError('k too large. imputation requires 2*k+1 < min(n,p)')
            diff = fZ[-k-1] - fZ[-2*k-1]
            for l in range(1,k+1):
                a = (1 - ((l-1)/k)**(2/3)) / (2**(2/3)-1) 
                fZ[-l] = fZ[-k-1] + a*diff
        else:
            raise ValueError('unknown strategy, should be one of \'0\',\'w\',\'i\'. given: ' + str(strategy))
    
    return fZ
        


"""
Computes the optimal hard thershold for a given (empirical) noise distribution fZ
and shape parameter gamma. The optimal threshold t* is the unique number satisfying
F_gamma(t*;fZ)=-4 .
Parameters:
    fZ: numpy array, whose entries define the counting measure to use
    gamma: shape parameter, assumed 0 < gamma <= 1.
"""
def computeOptThreshold(fZ, gamma):
    
    low = np.max(fZ)
    high = low + 2.0
    
    while F(high, fZ, gamma) < -4:
        low = high
        high = 2*high
    
    # F is increasing, do binary search:
    eps = 10e-6
    while high-low > eps:
        mid = (high+low)/2
        if F(mid, fZ, gamma) < -4:
            low = mid
        else:
            high = mid
    return mid

"""
Compute the functional Phi(y;fZ), evaluated at y and counting (discrete) measure
defined by the entries of fZ.
Parameters:
    y: values to evaluate at
    fZ: numpy array, whose entries define the counting measure to use
"""
def Phi(y, fZ):
    phi = y/(y**2 - fZ**2)
    return np.mean(phi)

"""
Compute the functional Phi'(y;fZ) (derivative of Phi w.r.t y), evaluated at y and counting (discrete) measure
defined by the entries of fZ.
Parameters:
    y: values to evaluate at
    fZ: numpy array, whose entries define the counting measure to use
"""
def Phid(y, fZ):
    phid = -(y**2+fZ**2)/(y**2-fZ**2)**2
    return np.mean(phid)
    

"""
Compute the functional D_gamma(y;fZ), evaluated at y and counting (discrete) measure
defined by the entries of fZ, with shape parameter gamma.
Parameters:
    y: values to evaluate at
    fZ: numpy array, whose entries define the counting measure to use
    gamma: shape parameter, assumed 0 < gamma <= 1.
"""
def D(y, fZ, gamma):
    phi = Phi(y, fZ)
    return phi * (gamma*phi + (1-gamma)/y)
    

"""
Compute the functional D_gamma'(y;fZ) (derivative of D_gamma w.r.t y), evaluated at y and counting (discrete) measure
defined by the entries of fZ, with shape parameter gamma.
Parameters:
    y: values to evaluate at
    fZ: numpy array, whose entries define the counting measure to use
    gamma: shape parameter, assumed 0 < gamma <= 1.
"""
def Dd(y, fZ, gamma):
    phi = Phi(y, fZ)
    phid = Phid(y, fZ)
    return phid * (gamma*phi + (1-gamma)/y) + phi * (gamma*phid - (1-gamma)/y**2)

"""
Compute the functional Psi_gamma(y;fZ), evaluated at y and counting (discrete) measure
defined by the entries of fZ, with shape parameter gamma.
Parameters:
    y: values to evaluate at
    fZ: numpy array, whose entries define the counting measure to use
    gamma: shape parameter, assumed 0 < gamma <= 1.
"""
def F(y, fZ, gamma):
    d = D(y, fZ, gamma)
    dd = Dd(y, fZ, gamma)
    return y * dd / d