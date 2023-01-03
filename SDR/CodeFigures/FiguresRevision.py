# -*- coding: utf-8 -*-


import OptimalAdaptiveSVThreshold as Opt
import Noise
import Experiments as Exp
import numpy as np
from numpy.linalg import svd, norm
from numpy import sqrt
import matplotlib.pyplot as plt
import os


def directory_name(noiseFunc, gamma):
    return noiseFunc.name + ', gamma=' + str(gamma)

"""
Returns:
    bulkEdge, x_BBP, T_opt, x_opt
"""
def noiseProperties(noiseFunc, gamma):
    p_oracle = 3000 
    _, fZ, _ = svd( noiseFunc(p_oracle, gamma) )
    
    bulkEdge = fZ[0]
    T_opt = Opt.computeOptThreshold(fZ, gamma)
    x_opt = Noise.invertSpike(T_opt, fZ, gamma)
    x_BBP=Noise.invertSpike(bulkEdge+0.01,fZ,gamma)
    
    return round(bulkEdge,2), round(x_BBP,2), round(T_opt,2), round(x_opt,2)

    
def figureHist(noiseFunc, gamma):
    
    p_oracle = 3000
    _, fZ, _ = svd( noiseFunc(p_oracle, gamma) )
    
    title = 'Noise = ' + noiseFunc.name + ', $\gamma=' + str(gamma) + '$'
    plt.clf()
    plt.title(title)
    plt.hist(fZ, bins=int(p_oracle/50), density=True)
    direc = directory_name(noiseFunc,gamma)
    if not os.path.exists(direc):
        os.makedirs(direc)
    plt.savefig(direc + '/hist.pdf')

def figureR0vsR1(noiseFunc, gamma):
    
    title = 'Noise = ' + noiseFunc.name + ', $\gamma=' + str(gamma) + '$'
    
    # Estimate some oracle quantities:
    
    p_oracle = 3000
    _, fZ, _ = svd(noiseFunc(p_oracle,gamma))
    
    bulkEdge=fZ[0]
    xBBP=Noise.invertSpike(bulkEdge+0.01,fZ,gamma)
    Topt = Opt.computeOptThreshold(fZ, gamma)
    Topt_x = Noise.invertSpike(Topt,fZ,gamma)
    xUpper = Topt_x + 1.0
    xsOracle = np.arange(xBBP, xUpper, 0.01)
    R1_oracle = np.zeros(xsOracle.size)
    for i in range(0,xsOracle.size):
        y = Noise.spikeForward(xsOracle[i],fZ,gamma)
        R1_oracle[i] = Exp.R1(y,fZ,gamma)
        
    
    # Experiment on generated data:    
        
    p = 500
    n = np.int( np.ceil(p/gamma) )
    a = np.random.normal(0,1,n)
    a = a/norm(a)
    b = np.random.normal(0,1,p)
    b = b/norm(b)
    
    xs = np.arange(0.05, xUpper, 0.05)
    
    Z = noiseFunc(p,gamma)
    
    R0_emp = xs**2
    R1_emp = np.zeros(xs.size)
    for i in range(0,xs.size):
        X = xs[i]*np.outer(a,b)
        Y = X + Z
        U, y, Vt = svd(Y)
        Xest = y[0]*np.outer( U[:,0], Vt[0,:].T )
        R1_emp[i] = norm(X-Xest)**2
    
    # Plot stuff:
    
    plt.clf()
    plt.rc('text', usetex=True)

    plt.plot(xs, R0_emp, color='blue', label='$R_0(x)=x^2$', linewidth=1)
    plt.plot(xs, R1_emp, color='orange',
             label='$\|x\mathbf{a}\mathbf{b}^{T} - y_{1}\mathbf{u}_1\mathbf{v}_1^{T}\|_F^2$', linewidth=1)
    plt.plot(xsOracle, R1_oracle, color='red', linestyle='-.', label='$R_1(x)$', linewidth=1)
    
    plt.axvline(x=Topt_x, linestyle='--', color='black', linewidth=1)
    plt.axvline(x=xBBP, linestyle='--', color='orange', linewidth=1)
    plt.text(Topt_x, 0, '$\mathcal{Y}^{-1}(T_{\gamma}(F^Z))$', color='black')
    plt.text(xBBP, 0, '$\hat{x}_{+}$', color='orange')

    
    plt.legend(fontsize='large', loc='upper left')
    plt.xlabel('$x$')
    plt.title(title)
    
    direc = directory_name(noiseFunc,gamma)
    if not os.path.exists(direc):
        os.makedirs(direc)
    plt.savefig(direc + '/R0vsR1.pdf')
    
def figureSEvsASE(noiseFunc, gamma):
    
    title = 'Noise = ' + noiseFunc.name + ', $\gamma=' + str(gamma) + '$'
    
    
    # Create sample    
        
    p = 500
    n = np.int( np.ceil(p/gamma) )
    x = np.array( [0.5, 1.0, 1.3, 2.5, 5.2] )
    r = 5
    k = 4*r
    
    G = np.random.normal(0,1,(n,n))
    A, _, _ = svd(G)
    A = A[:,0:r]
    G = np.random.normal(0,1,(p,p))
    B, _, _ = svd(G)
    B = B[:,0:r]
    X = A @ np.diag(x) @ B.T
    
    Z = noiseFunc(p,gamma)
    Y = X + Z
    U, fY, Vt = svd(Y)
    
    
    _, T_0, _ = Opt.adaptiveHardThresholding(Y, k, strategy='0')
    _, T_w, _ = Opt.adaptiveHardThresholding(Y, k, strategy='w')
    _, T_i, _ = Opt.adaptiveHardThresholding(Y, k, strategy='i')

    
    # Estimate oracle quantities:
    
    p_oracle = 3000
    _, fZ, _ = svd( noiseFunc(p_oracle, gamma) )    # need this to plot ASE
    bulkEdge = fZ[0]
    T_opt = Opt.computeOptThreshold(fZ, gamma)    
    
    tsOracle = np.arange(bulkEdge+0.001, fY[0]+0.5, 0.005)
    ASE = Exp.ASEt(x,fZ,gamma,tsOracle)
    
    
    # Prepare to plot SEn[x|t]:
        
    # how many additional empirical svs should we plot? 
    # as a rule of thump, stop once SE[x|t] > 1.5 max_{t=y[0],..y[r]} SE[x|t]
    
    SE_r = np.zeros(r+1)
    for k in range(0,r+1):
        SE_r[k] = norm(X - U[:,0:k] @ np.diag(fY[0:k]) @ Vt[0:k,:])**2
    SE_r_max = np.max(SE_r)
    
    pos = r+1 # want to plot up to fY[r+1], exclusive
    SE_pos = SE_r[k] = norm(X - U[:,0:pos] @ np.diag(fY[0:pos]) @ Vt[0:pos,:])**2
    while SE_pos <= 1.5 * SE_r_max and pos < p:
        pos = pos + 1
        SE_pos = norm(X - U[:,0:pos] @ np.diag(fY[0:pos]) @ Vt[0:pos,:])**2
        
    tsEmp = np.arange(fY[pos]+0.005, fY[0]+0.5, 0.005)
    SE = Exp.SEnt(X, Y, tsEmp)
    
    # Plot stuff:
    
    plt.clf()
    plt.rc('text', usetex=True)

    plt.plot(tsEmp, SE, color='blue', label=r'$SE_n[\mathbf{x}|\theta]$', linewidth=1)
    plt.plot(tsOracle, ASE, color='orange',
             label=r'$ASE[\mathbf{x}|\theta]$', linestyle='-.', linewidth=1)
    
    plt.axvline(x=T_opt, linestyle='--', color='black', label='$T_{\gamma}(F_Z)$', linewidth=1)
    plt.axvline(x=T_0, linestyle='--', color='green', label='$T_{p/n}(F_{n,k}^{0})$', linewidth=1)
    plt.axvline(x=T_w, linestyle='--', color='yellow', label='$T_{p/n}(F_{n,k}^{w})$', linewidth=1)
    plt.axvline(x=T_i, linestyle='--', color='red', label='$T_{p/n}(F_{n,k}^{i})$', linewidth=1)
    plt.axvline(x=bulkEdge, linestyle='--', color='gray', label='$\mathcal{Z}_{+}(F_Z)$', linewidth=1)
#    plt.text(Topt, 0, '$T_{\gamma}$', color='black')
#    plt.text(bulkEdge, 0, '$y_{+}$', color='gray')

    
    plt.legend(fontsize='medium')
    plt.xlabel('$t$')
    plt.title(title)
    
    direc = directory_name(noiseFunc,gamma)
    if not os.path.exists(direc):
        os.makedirs(direc)
    plt.savefig(direc + '/SEvsASE.pdf')
        
def figureOracleAttainment(noiseFunc, gamma):
    
    # compute 'oracle' optimal treshold
    _, _, T_opt, _ = noiseProperties(noiseFunc, gamma)
    
    T = 50  # trials per value of p, want emp success probability over T trials
    
    x = np.array([0.5, 1.0, 1.3, 2.5, 5.2])
    r = 5
    k = 4*r

    ps = np.array([60, 120, 180, 240, 300, 360, 420, 480, 540, 600])
    # ps = np.array([100])
    success_oracle = np.zeros(ps.size)
    success_0 = np.zeros(ps.size)
    success_w = np.zeros(ps.size)
    success_i = np.zeros(ps.size)
 
    
    for l in range(0,ps.size):
        
        p = ps[l]
        
        for t in range(0,T):
            
            # generate problem instance
            n = np.int( np.ceil(p/gamma) )
            
            G = np.random.normal(0,1,(n,n))
            A, _, _ = svd(G)
            A = A[:,0:r]
            G = np.random.normal(0,1,(p,p))
            B, _, _ = svd(G)
            B = B[:,0:r]
            X = A @ np.diag(x) @ B.T
            
            Z = noiseFunc(p,gamma)
            Y = X + Z
            _, fY, _ = svd(Y)
            # print(fY[0:5])
            
            T_0 = Opt.computeOptThreshold( Opt.createPseudoNoise(fY, k, strategy='0'), gamma )
            T_w = Opt.computeOptThreshold( Opt.createPseudoNoise(fY, k, strategy='w'), gamma )
            T_i = Opt.computeOptThreshold( Opt.createPseudoNoise(fY, k, strategy='i'), gamma )
            # print(T_opt, T_0, T_w, T_i)

            _, pos = Exp.SEnOpt(X, r, Y)    # pos is how many leading principal component of Y are taken 
                                            # to achieve the optimum
            interv_up = np.Inf
            interv_down = 0      
            if pos == 0:
                interv_down = fY[0]
            elif pos == p:
                interv_up = fY[-1]
            else:
                interv_up = fY[pos-1]
                interv_down = fY[pos]
            # print(interv_down, interv_up)
            
            if T_opt >= interv_down and T_opt < interv_up:
                success_oracle[l] = success_oracle[l] + 1
            if T_0 >= interv_down and T_0 < interv_up:
                success_0[l] = success_0[l] + 1
            if T_w >= interv_down and T_w < interv_up:
                success_w[l] = success_w[l] + 1
            if T_i >= interv_down and T_i < interv_up:
                success_i[l] = success_i[l] + 1
    
    title = 'Noise = ' + noiseFunc.name + ', $\gamma=' + str(gamma) + '$'
    plt.clf()
    plt.rc('text', usetex=True)

    plt.plot(ps, success_oracle/T, label='$\hat{P}(T_{\gamma}(F_Z))$', marker='.')
    plt.plot(ps, success_0/T, label='$\hat{P}(T_{p/n}(F_{n,k}^{0}))$', marker='x')
    plt.plot(ps, success_w/T, label='$\hat{P}(T_{p/n}(F_{n,k}^{w}))$', marker='v')
    plt.plot(ps, success_i/T, label='$\hat{P}(T_{p/n}(F_{n,k}^{i}))$', marker='o')

    
    plt.legend(fontsize='large')
    plt.xlabel('$p$')
    plt.title(title)
    
    direc = directory_name(noiseFunc,gamma)
    if not os.path.exists(direc):
        os.makedirs(direc)
    plt.savefig(direc + '/oracleAttainment.pdf')
    
def figureRegret(noiseFunc, gamma):

    # compute 'oracle' optimal treshold
    _, x_BBP, T_opt, x_opt = noiseProperties(noiseFunc, gamma)
    
    x_low = np.max([x_opt-0.6, 0.05])
    x_high = x_opt + 0.6
    xs = np.arange(x_low, x_high, 0.05)
    p = 500
    n = np.int( np.ceil(p/gamma) )
    k= 10
    
    T = 20  # trials
    regret_opt_tot = np.zeros(xs.size)
    regret_0_tot = np.zeros(xs.size)
    regret_w_tot = np.zeros(xs.size)
    regret_i_tot = np.zeros(xs.size)
    
    # generate noise and spikes
    
    for t in range(0,T):
        
        a = np.random.normal(0,1,n)
        a = a/norm(a)
        b = np.random.normal(0,1,p)
        b = b/norm(b)
        Xouter = np.outer(a,b)
        
        
        Z = noiseFunc(p, gamma)
        
        regret_opt = []
        regret_0 = []
        regret_w = []
        regret_i = []
        
        for x in xs:
            
            X = x*Xouter
            Y = X + Z
            U, y, Vt = svd(Y)
            SE_oracle, _ = Exp.SEnOpt(X,1,Y)
            
            # adaptively calculate thresholds
            T_0 = Opt.computeOptThreshold( Opt.createPseudoNoise(y,k, strategy='0'),gamma )
            T_w = Opt.computeOptThreshold( Opt.createPseudoNoise(y,k, strategy='w'),gamma )
            T_i = Opt.computeOptThreshold( Opt.createPseudoNoise(y,k, strategy='i'),gamma )
            
            reff = np.sum(y>T_opt)
            Xhat_opt = U[:,0:reff] @ np.diag(y[0:reff]) @ Vt[0:reff,:]
            SE_opt = norm(X-Xhat_opt)**2
            regret_opt.append( SE_opt/SE_oracle )
            
            reff = np.sum(y>T_0)
            Xhat_0 = U[:,0:reff] @ np.diag(y[0:reff]) @ Vt[0:reff,:]
            SE_0 = norm(X-Xhat_0)**2
            regret_0.append( SE_0/SE_oracle )
            
            reff = np.sum(y>T_w)
            Xhat_w = U[:,0:reff] @ np.diag(y[0:reff]) @ Vt[0:reff,:]
            SE_w = norm(X-Xhat_w)**2
            regret_w.append( SE_w/SE_oracle )
            
            reff = np.sum(y>T_i)
            Xhat_i= U[:,0:reff] @ np.diag(y[0:reff]) @ Vt[0:reff,:]
            SE_i = norm(X-Xhat_i)**2
            regret_i.append( SE_i/SE_oracle )
            
        regret_opt_tot = regret_opt_tot + np.array(regret_opt)/T
        regret_0_tot = regret_0_tot + np.array(regret_0)/T
        regret_w_tot = regret_w_tot + np.array(regret_w)/T
        regret_i_tot = regret_i_tot + np.array(regret_i)/T
        
    title = 'Noise = ' + noiseFunc.name + ', $\gamma=' + str(gamma) + '$'
    plt.clf()
    plt.rc('text', usetex=True)

    plt.plot(xs, regret_opt_tot, label='$T_{\gamma}(F_Z)$', marker='s', linewidth=0.5)
    plt.plot(xs, regret_0_tot, label='$T_{p/n}(F_{n,k}^{0})$', marker='x', linewidth=0.5)
    plt.plot(xs, regret_w_tot, label='$T_{p/n}(F_{n,k}^{w})$', marker='v', linewidth=0.5)
    plt.plot(xs, regret_i_tot, label='$T_{p/n}(F_{n,k}^{i})$', marker='o', linewidth=0.5)
    
    plt.axvline(x=x_opt, linewidth=0.5, linestyle='--', color='black')
    # plt.text(x_opt, 0.9, '$x^*$', color='black')

    plt.legend(fontsize='medium')
    plt.xlabel('$x$')
    plt.ylabel('$SE_n[\mathbf{x}|\hat{\theta}]/SE^*_n[\mathbf{x}]$')
    plt.title(title)
    
    direc = directory_name(noiseFunc,gamma)
    if not os.path.exists(direc):
        os.makedirs(direc)
    plt.savefig(direc + '/regret.pdf')

def figureConvergenceRate(noiseFunc, gamma):
    
    ps = np.arange(60, 781, 30)
    
    x = np.arange(1,11,1)
    r = 10
    k = 20
    
    T = 50
    
    _, _, T_opt, _ = noiseProperties(noiseFunc, gamma)
    
    err_0_arr = []
    err_w_arr = []
    err_i_arr = []
    
    for p in ps:
        
        err_0 = 0
        err_w = 0
        err_i = 0
        
        for t in range(0,T):
            
            # generate problem instance
            n = np.int( np.ceil(p/gamma) )
            
            G = np.random.normal(0,1,(n,n))
            A, _, _ = svd(G)
            A = A[:,0:r]
            G = np.random.normal(0,1,(p,p))
            B, _, _ = svd(G)
            B = B[:,0:r]
            X = A @ np.diag(x) @ B.T
            
            Z = noiseFunc(p,gamma)
            Y = X + Z
            _, fY, _ = svd(Y)
            # print(fY[0:5])
            
            T_0 = Opt.computeOptThreshold( Opt.createPseudoNoise(fY, k, strategy='0'), gamma )
            T_w = Opt.computeOptThreshold( Opt.createPseudoNoise(fY, k, strategy='w'), gamma )
            T_i = Opt.computeOptThreshold( Opt.createPseudoNoise(fY, k, strategy='i'), gamma )
            # print(T_opt, T_0, T_w, T_i)
            
            err_0 = err_0 + np.abs(T_0-T_opt)/T_opt
            err_w = err_w + np.abs(T_w-T_opt)/T_opt
            err_i = err_i + np.abs(T_i-T_opt)/T_opt
         
        err_0_arr.append(err_0/T)
        err_i_arr.append(err_w/T)
        err_w_arr.append(err_i/T)
    
    title = 'Noise = ' + noiseFunc.name + ', $\gamma=' + str(gamma) + '$'
    plt.clf()
    plt.rc('text', usetex=True)

    plt.yscale('log')
    plt.xscale('log')

    plt.plot(ps, err_0_arr, label='$T_{p/n}(F_{n,k}^{0})$', marker='s', linewidth=0.5, color='orange')
    plt.plot(ps, err_w_arr, label='$T_{p/n}(F_{n,k}^{w})$', marker='x', linewidth=0.5, color='blue')
    plt.plot(ps, err_i_arr, label='$T_{p/n}(F_{n,k}^{i})$', marker='v', linewidth=0.5, color='red')

    plt.legend(fontsize='medium')
    plt.xlabel('$p$')
    plt.ylabel('rel. abs. error' )
    plt.title(title)
    
    direc = directory_name(noiseFunc,gamma)
    if not os.path.exists(direc):
        os.makedirs(direc)
    plt.savefig(direc + '/convergenceRate.pdf')

# generate latex for plots    
def generateLatex(noiseFunc, gamma):
    plots_dir = 'SI_figures'
    dist_dir = directory_name(noiseFunc, gamma)
    tex = r"""
\subsection{Distribution: """ + noiseFunc.name + r""", $\gamma = """ + str(gamma) + r"""$}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{{""" + plots_dir + r"""/""" + dist_dir + r"""/hist}}
    \caption{Experiment: {\bf Hist}}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{{""" + plots_dir + r"""/""" + dist_dir + r"""/R0vsR1}}
    \caption{Experiment: {\bf R0-vs-R1}}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{{""" + plots_dir + r"""/""" + dist_dir + r"""/SEvsASE}}
    \caption{Experiment: {\bf SE-vs-ASE}}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{{""" + plots_dir + r"""/""" + dist_dir + r"""/oracleAttainment}}
    \caption{Experiment: {\bf OracleAttainment}}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{{""" + plots_dir + r"""/""" + dist_dir + r"""/regret}}
    \caption{Experiment: {\bf Regret}}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{{""" + plots_dir + r"""/""" + dist_dir + r"""/convergenceRate}}
    \caption{Experiment: {\bf ConvergenceRate}}
\end{figure}


"""
    return tex
    
NoiseFuncs = [Noise.noiseMarcenkoPastur]
# NoiseFuncs = [Noise.noiseMarcenkoPastur,Noise.noiseChi10,
#               Noise.noiseFisher3n,Noise.noiseMix2,
#               Noise.noiseUnif_1_to_10,Noise.noisePaddedIdentity]
# gammas = [0.5, 1.0]
# gammas = [1.0]
gammas = [0.5]
for noiseFunc in NoiseFuncs:
    for gamma in gammas:
        # figureHist(noiseFunc,gamma)
        # figureR0vsR1(noiseFunc,gamma)
        figureSEvsASE(noiseFunc,gamma)
        # figureOracleAttainment(noiseFunc, gamma)
        # figureRegret(noiseFunc,gamma)
        # figureConvergenceRate(noiseFunc, gamma)
        # with open('tex_dump.tex', 'a') as file:
        #     file.write( generateLatex(noiseFunc, gamma) )
        # print(' & '.join( [ str(x) for x in (noiseFunc.name, gamma) + noiseProperties(noiseFunc, gamma) ] ) )
    
