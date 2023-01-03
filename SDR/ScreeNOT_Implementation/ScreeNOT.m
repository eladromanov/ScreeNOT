function [Xest, Topt, r] = ScreeNOT(Y, k, strategy)
% ScreeNOT - Performs optimal adaptive hard thresholding 
% on the matrix Y
% 
% Syntax [Xest, Topt,r] = ScreeNOT(Y, k, strategy)
%
% Inputs:
%     Y - the measured matrix
%     k - an upper bound on the signal rank. 
%     strategy: method to reconstruct the noise bulk. 
%         one of '0' (tranpsort to zero), 'w' (winsorization), 'i' (imputation)
%         default='i'
%         
% Outputs:
%     Xest - an estimate of the low rank signal
%     Topt - the hard threshold used
%     r - the number of relevant components, r = rank(Xest)

if nargin <= 2
   strategy = 'i';
end
[U, fY_diag, V] = svd(Y,'econ');
fY = diag( fY_diag );

sY = size(Y);
gamma = min( [sY(1)/sY(2), sY(2)/sY(1)] );

fZ = createPseudoNoise( fY, k, strategy);

Topt = computeOptThreshold(fZ, gamma);

fY_new = fY .* (fY > Topt);
r = sum( fY > Topt );

Xest = U * diag(fY_new) * V.';
end

function fZ = createPseudoNoise(fY, k, strategy)
% CREATEPSEUDONOISE - Creates a "pseudo-noise" singular from the singular 
% values of the observed matrix Y, which is given in the array fY. 
%
% Syntax: fZ = createPseudoNoise(fY, k, strategy)
%
% Inputs:
%     fY - a vector containing the observed singular values, one which we operate
%     k - an upper bound on the signal rank. The leading k values in fY are discarded
%     strategy - one of '0' (tranpsort to zero), 'w' (winsorization), 'i' (imputation)
%         default='i'
% Ouputs:
%     fZ - estimated noise bulk
if nargin < 2
   strategy = 'i';
end
fZ = sort(fY);
p = length(fZ);
if k >= p
    error('k too large. procedure requires k < min(n,p)');
end
if k > 0
    if strategy == '0' % transport to zero
        fZ( (p-k+1):p ) = 0;
    elseif strategy == 'w' %winsorization
        fZ( (p-k+1):p ) = fZ(p-k);
    elseif strategy == 'i'   %imputation
        if 2*k+1 >= p
            error('k too large. imputation requires 2*k+1 < min(n,p)');
        end
        diff = fZ(p-k) - fZ(p-2*k);
        for l = 1:k
            a = (1 - ((l-1)/k)^(2/3)) / (2.^(2/3)-1);
            fZ(p-l+1) = fZ(p-k) + a*diff;
        end
    else
        error_str = strcat('unknown strategy, should be one of 0, w, i. given:  ', string(strategy));
        error( error_str )
    end
end
end

function mid = computeOptThreshold(fZ, gamma)
% COMPUTEOPTTHRESHOLD -Computes the optimal hard thershold for a given 
% (empirical) noise distribution fZ and shape parameter gamma. The optimal 
% threshold t* is the unique number satisfying F_gamma(t*;fZ)=-4 .
% 
% Syntax mid = computeOptThreshold(fZ, gamma)
%
% Input:
%     fZ - vector, whose entries define the counting measure to use
%     gamma - shape parameter, assumed 0 < gamma <= 1.
% Output:   
%     mid - threshold for noise distribuition
low = max(fZ);
high = low + 2.0;
while F(high, fZ, gamma) < -4
    low = high;
    high = 2*high;
end
%F is increasing, do binary search:
eps = 10e-6;
while high-low > eps
    mid = (high+low)/2;
    if F(mid, fZ, gamma) < -4
        low = mid;
    else
        high = mid;
    end
end
end 

function phi = Phi(y, fz)
% PHI Compute the functional Phi(y;fZ), evaluated at y and counting 
%(discrete) measure defined by the entries of fZ.
%
%Syntax phi = Phi(y, fZ)
%
% Input:
%     y - value to evaluate at
%     fZ - vector, whose entries define the counting measure to use
% Output:
%     phi - function value of phi
phi = mean(y ./ (y^2-fz.^2));
end

function phid = Phid(y, fZ)
% PHID Compute the functional Phi'(y;fZ) (derivative of Phi w.r.t y), evaluated 
% at y and counting (discrete) measure defined by the entries of fZ.
%
% Syntax phi = Phi(y, fZ)
%
% Input:
%     y - value to evaluate at
%     fZ - vector, whose entries define the counting measure to use
% Output:
%     phid - function value of phid
phid = mean(-((y^2+fZ.^2)./((y^2-fZ.^2).^2)));
end


function d = D(y, fZ, gamma)
% Dd Compute the functional D_gamma(y;fZ), evaluated at y and counting 
% (discrete) measure defined by the entries of fZ, with shape parameter gamma.
%
%Syntax d = D(y, fZ)
%
% Input:
%     y - value to evaluate at
%     fZ - vector, whose entries define the counting measure to use
%     gamma - shape parameter, assumed 0 < gamma <= 1.
% Output:
%     d - function value of D
phi = Phi(y,fZ);
d = phi * (gamma * phi + (1 - gamma)/y);
end


function dd = Dd(y, fZ, gamma)
% DD Compute the functional D_gamma'(y;fZ) (derivative of D_gamma w.r.t y),
% evaluated at y and counting (discrete) measure
% defined by the entries of fZ, with shape parameter gamma.
%
%Syntax dd = Dd(y, fZ)
%
% Input:
%     y - value to evaluate at
%     fZ - vector, whose entries define the counting measure to use
%     gamma - shape parameter, assumed 0 < gamma <= 1.
% Output:
%     d - function value of Dd
phi = Phi(y,fZ);
phid = Phid(y, fZ);
dd = phid * (gamma*phi + (1-gamma)/y) + phi * (gamma*phid - (1-gamma)/y^2);
end

function f = F(y, fZ, gamma)
% F Compute the functional Psi_gamma(y;fZ), evaluated at y and counting 
% (discrete) measure defined by the entries of fZ, with shape parameter gamma.
%
% Syntax f = F(y, fZ, gamma)
%
% Input:
%     y - value to evaluate at
%     fZ - vector, whose entries define the counting measure to use
%     gamma - shape parameter, assumed 0 < gamma <= 1.
% Output:
%     d - function value of Dd
d = D(y, fZ, gamma);
dd = Dd(y, fZ, gamma);
f = y * dd / d;
end
