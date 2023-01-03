# Implementation of the ScreeNOT procedure.
# The function adaptiveHardThresholding performs adaptive singular value thresholding
# on its input; from a user's perspecive, it is the main (only) API for this
# code package.
# 
# ``ScreeNOT : Exact MSE-Optimal Singular Value Thresholding in Correlated Noise ''
# by David L. Donoho, Matan Gavish, Elad Romanov. arxiv:2009.12297
# 
# Cite this as: 
# Donoho, David L., Gavish, Matan and Romanov, Elad. (2020). Code Supplement for "ScreeNOT: Exact MSE-Optimal Singular Value Thresholding in Correlated Noise". Stanford Digital Repository. https://arxiv.org/abs/2009.12297
# 
# 
# MIT License
# 
# Copyright (c) 2020 David L. Donoho, Matan Gavish, Elad Romanov
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#'Performs optimal adaptive hard thresholding on the matrix Y
#'@param Y: the measured matrix
#'@param k: an upper bound on the signal rank.
#'@param strategy: method to reconstruct the noise bulk. 
#'           one of '0' (tranpsort to zero), 'w' (winsorization), 'i' (imputation)
#'           default='i'
#' @return The returned value is a list
#' with the following components:
#'    Xest: an estimate of the low rank signal
#'    Topt: the hard threshold used
#'    r: the number of relevant components, r = rank(Xest)

ScreeNOT = function(Y, k, strategy='i') {
  Y_svd = svd(Y)
  fY=Y_svd$d
  Y_dims = dim(Y)
  gamma = min(Y_dims[[1]]/Y_dims[[2]], Y_dims[[2]]/Y_dims[[1]])
  
  fZ = createPseudoNoise(fY, k, strategy=strategy)
  Topt = computeOptThreshold(fZ, gamma)

  fY_new = fY*(fY>Topt)
  r = sum( fY>Topt )
  
  U = Y_svd$u
  Vt = t( Y_svd$v )
  Xest = U %*% diag(fY_new) %*% Vt
  
  return( list( Xest=Xest, Topt=Topt, r=r ) )
}

#Creates a "pseudo-noise" singular from the singular values of the observed matrix 
#'Y, which is given in the array fY.
#'@param fY: a numpy array containing the observed singular values, one which we operate
#'@param k: k: an upper bound on the signal rank. The leading k values in fY are discarded
#'@param strategy: strategy: one of '0' (tranpsort to zero), 'w' (winsorization), 'i' (imputation) 
#'default='i'    r: the number of relevant components, r = rank(Xest)
#'
createPseudoNoise = function(fY, k, strategy='i') {
  # sort fZ into increasing order
  fZ = sort(fY)
  
  p = length(fZ)
  if (k >= p)
  {
    stop('k too large. procedure requires k < min(n,p)')
  }
  
  if (k > 0)
  {
    
    if (strategy == '0') # transport to zero
    {
      fZ[(p-k+1):p] = 0
    }
    else if (strategy == 'w') #winsorization
    {
      fZ[(p-k+1):p] = fZ[p-k]  
    }
    else if (strategy == 'i')   # imputation
    {
      if (2*k+1 >= p)
      {
        stop('k too large. imputation requires 2*k+1 < min(n,p)')
      }        
      diff = fZ[p-k] - fZ[p-2*k]
      for (l in 1:k)
      {
        a = (1 - ((l-1)/k)**(2/3)) / (2**(2/3)-1) 
        fZ[p-l+1] = fZ[p-k] + a*diff
      }
    }
    else
    {
      err_str = cat('unknown strategy, should be one of \'0\',\'w\',\'i\'. given: ', format(strategy))
      stop( err_str )
    }
  }
  return (fZ)
}

#'Computes the optimal hard thershold for a given (empirical) noise distribution fZ
#'and shape parameter gamma. The optimal threshold t* is the unique number satisfying
#'F_gamma(t*;fZ)=-4 .
#'@param fZ: array, whose entries define the counting measure to use
#'@param gamma: dim parameter, assumed 0 < gamma <= 1.

computeOptThreshold = function(fZ, gamma) {
  low = max(fZ)
  high = low + 2.0
  
  while (F(high, fZ, gamma) < -4) {
    low = high
    high = 2*high
  }
  
  # F is increasing, do binary search:
  eps = 10e-6
  while (high-low > eps) {
    mid = (high+low)/2
    if (F(mid, fZ, gamma) < -4) {
      low = mid
    }
    else {
      high = mid
    }
  }
  return (mid)
}

#' Compute the functional Phi(y;fZ), evaluated at y and counting (discrete) measure
#' defined by the entries of fZ.
#' @param  y: values to evaluate at
#' @param fZ: array, whose entries define the counting measure to use

Phi = function(y, fZ) {
  phi = mean(y/(y**2 - fZ**2))
  return (phi)
}

#' Compute the functional Phi'(y;fZ) (derivative of Phi w.r.t y), evaluated at 
#' y and counting (discrete) measure defined by the entries of fZ.
#' @param  y: values to evaluate at
#' @param fZ: array, whose entries define the counting measure to use

Phid = function(y, fZ) {
  fz2 <- fZ**2
  phid = (-(y**2+fz2)/(y**2-fz2)**2)
  return( mean(phid) )
}

#'Compute the functional D_gamma(y;fZ), evaluated at y and counting (discrete) 
#'measure defined by the entries of fZ, with shape parameter gamma.
#'@param y: values to evaluate at
#'@param fZ: numpy array, whose entries define the counting measure to use
#'@param gamma: shape parameter, assumed 0 < gamma <= 1.

D = function(y, fZ, gamma){
  phi = Phi(y, fZ)
  return (phi * (gamma*phi + (1-gamma)/y))
}

#'Compute the functional D_gamma'(y;fZ) (derivative of D_gamma w.r.t y), 
#'evaluated at y and counting (discrete) measure defined by the entries of fZ, 
#'with shape parameter gamma.
#'@param y: values to evaluate at
#'@param fZ: numpy array, whose entries define the counting measure to use
#'@param gamma: shape parameter, assumed 0 < gamma <= 1.

Dd = function(y, fZ, gamma){
  phi = Phi(y, fZ)
  phid = Phid(y, fZ)
  return (phid * (gamma*phi + (1-gamma)/y) + phi * (gamma*phid - (1-gamma)/y**2))
}

#'Compute the functional Psi_gamma(y;fZ), evaluated at y and counting (discrete) 
#'measure defined by the entries of fZ, with shape parameter gamma.
#'@param y: values to evaluate at
#'@param fZ: numpy array, whose entries define the counting measure to use
#'@param strategy: gamma: shape parameter, assumed 0 < gamma <= 1.

F = function(y, fZ, gamma)
{
  d = D(y, fZ, gamma)
  dd = Dd(y, fZ, gamma)
  return (y * dd / d)
}
