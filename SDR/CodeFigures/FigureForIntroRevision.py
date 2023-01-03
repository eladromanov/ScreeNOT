# -*- coding: utf-8 -*-
"""
Code for generating the "stylized example" in our ScreeNOT paper intro.

Note: The parameters of the simulation have changed quite a bit after Revision1
of the paper.

"""

import OptimalAdaptiveSVThreshold as Opt
import Noise
import Experiments
import numpy as np
from numpy.linalg import svd, norm
from numpy import sqrt
import matplotlib.pyplot as plt


def AR1_Noise(n, p, rho):
    Z = np.zeros((n,p))
    for i in range(0,n):
        Z[i,0] = np.random.normal(0,1)
        for j in range(1,p):
            Z[i,j] = rho*Z[i,j-1] + np.sqrt((1-rho**2))*np.random.normal(0,1)
    return Z/sqrt(n)

np.random.seed(1234)


# x  = np.array([1.0,2.0,3.0,4.0])
# r = 4
# k = 12
r = 10
k = 15
jump=0.15
x = np.arange(1.0, 1.0 + jump*r, jump)

n = 1000
p = 1000
gamma = p/n
rho = 0.4
Z = AR1_Noise(n,p,rho)

G = np.random.normal(0,1,(n,p))
A, _, Bt = svd(G)
X = A[:,0:r] @ np.diag(x) @ Bt[0:r,:]
Y = X + Z

_, fZ, _ = svd(Z)
bulkEdge = fZ[0]
print(bulkEdge)
print('Z bulkEdge = ' + str(fZ[0]))

vals_count = 30 

U_Y, fY, Vt_Y = svd(Y)

optLoss, optHowMany = Experiments.SEnOpt(X, r, Y)
MethodEst, Topt, _ = Opt.adaptiveHardThresholding(Y, k)
_, ToptOracle, _ = Opt.adaptiveHardThresholding(Z, 0)
print('ToptOracle=' + str(ToptOracle))
print('Topt=' + str(Topt))


"""
IntroFig1
"""
plt.clf()
# plt.rc('text', usetex=True)
plt.plot(range(1,optHowMany+1), fY[0:optHowMany], linestyle='None', marker='o', color='red')
plt.plot(range(optHowMany+1,vals_count+1), fY[optHowMany:vals_count], linestyle='None', marker='o', color='blue')
plt.axhline(y=Topt, color='green', linestyle='--', label='ScreeNOT')
plt.axhline(y=bulkEdge, color='grey', linestyle='--', label='The ``elbow\'\' in the scree plot (?)')
plt.xlabel(r'Component number')
plt.ylabel(r'Singular value')
plt.title(r'Scree plot: AR(1) noise')
plt.legend()
plt.savefig('IntroFig1.pdf', bbox_inches='tight')


"""
IntroFig2
"""
plt.clf()
# plt.rc('text', usetex=True)
pseudoNoise = Opt.createPseudoNoise(fY, k)
edge = np.max(pseudoNoise)
thetas = np.arange(edge+0.05, Topt+0.3, 0.01)
Psis = np.zeros(thetas.shape)
for i in range(0,thetas.size):
    Psis[i] = Opt.F(thetas[i],pseudoNoise,gamma)
plt.plot(thetas,Psis)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\Psi(\theta)$')
plt.axhline(y=-4, color='blue', linestyle='--')
plt.axvline(x=Topt, color='green', linestyle='--')
plt.savefig('IntroFig2.pdf', bbox_inches='tight')


"""
IntroFig3
"""
plt.clf()
# plt.rc('text', usetex=True)
thetas = np.arange(fY[vals_count], fY[0]+0.5, 0.01)
SEs = Experiments.SEnt(X, Y, thetas)
plt.plot(thetas, SEs, label=r'$SE[\theta|X]$')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$SE[\theta|X]$')
plt.axvline(x=Topt, color='green', linestyle='--', label=r'$\hat{\theta}$')
plt.axhline(y=optLoss, color='red', linestyle='--', label=r'$\min_{\theta} \,\,SE[\theta|X]$')
plt.legend()
plt.savefig('IntroFig3.pdf', bbox_inches='tight')


# Some calculations, valid FOR THIS PARTICULAR SEED, 
# which are to be reported in the paper.
scree_plot_r = 6;
fY_ScreePlot = fY
fY_ScreePlot[scree_plot_r:] = 0
scree_plot_Est = U_Y @ np.diag(fY_ScreePlot) @ Vt_Y
ScreePlot_SE = np.linalg.norm(X-scree_plot_Est)**2
Method_SE = np.linalg.norm(X-MethodEst)**2

print('SE for scree plot: ' + str(ScreePlot_SE))
print('SE for ScreeNot: ' + str(Method_SE))
print('======================')

"""
IntroFig4

I want a signal with larger rank!

"""
r=40
k=60
vals_count = 150
jump=0.04
x = 1.0 + jump*np.arange(0,r)
X = A[:,0:r] @ np.diag(x) @ Bt[0:r,:]    
Z = AR1_Noise(n,p,rho)
Y = X + Z
U_Y, fY, Vt_Y = svd(Y)
optLoss, optHowMany = Experiments.SEnOpt(X, r, Y)
MethodEst, Topt, _ = Opt.adaptiveHardThresholding(Y, k)
plt.clf()
# plt.rc('text', usetex=True)
plt.plot(range(1,optHowMany+1), fY[0:optHowMany], linestyle='None', marker='o', color='red')
plt.plot(range(optHowMany+1,vals_count+1), fY[optHowMany:vals_count], linestyle='None', marker='o', color='blue')
plt.axhline(y=Topt, color='green', linestyle='--', label='ScreeNOT')
plt.axhline(y=bulkEdge, color='grey', linestyle='--', label='The ``elbow\'\' in the scree plot (?)')
plt.xlabel(r'Component number')
plt.ylabel(r'Singular value')
plt.title(r'Scree plot: AR(1) noise')
plt.legend()
plt.savefig('IntroFig4.pdf', bbox_inches='tight')

print("OptLoss = " + str(optLoss))
print('ScreeNOT loss =' + str(np.linalg.norm(X-MethodEst)**2) )
fY_Dummy=fY
fY_Dummy[fY<= bulkEdge]=0.0
elbowEst = U_Y @ np.diag(fY_Dummy) @ Vt_Y
print('Elbow loss = ' + str(np.linalg.norm(X-elbowEst)**2) )


np.random.seed()
