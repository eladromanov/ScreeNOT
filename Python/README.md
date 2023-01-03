# *ScreeNOT*: Optimal Singular Value Thresholding in Correlated Noise

ScreeNOT is a code library for hard thresholding of singular values. The procedure adaptively estimates the best singular value threshold under unknown noise characteristics. The threshold chosen by ScreeNOT is optimal (asymptotically, in the sense of minimum Frobenius error) under the the so-called "Spiked model" of a low-rank matrix observed in additive noise. In contrast to previous works, the noise is *not* assumed to be i.i.d. or white; it can have an essentially and **unknown** correlation structure, across either rows, columns or both. 
ScreeNOT is proposed to practitioners as a mathematically solid alternative to Cattell's ever-popular but vague Scree Plot heuristic from 1966.


Please refer to the project's homepage: https://github.com/eladromanov/ScreeNOT


## Usage

The main API for this package is the function

`adaptiveHardThresholding(Y, k, strategy='i')`

It receives as input a matrix `Y` (in Python: a `numpy` array); a non-negative integer `k`, an upper bound (potentially loose) on the low-rank signal to-be-recovered; (optional) a string `strategy`, which defines an estimation strategy for the noise bulk - valid options are `'i'` (imputation; this is the default, and should be best for most practical uses), `'0'` (transport to zero) or `'w'` (winsorization) - see paper for details.

The function returns three values:

`Xest`: a matrix with same dimensions as `Y`; an estimator for the low-rank signal (the result of performing thresholding on the singular values of `Y`).
`Topt`: the hard threshold which was used; computed adaptively from the singular values of `Y`.
`r`: the number of "relevant"/"strong" components of the signal: r=rank(Xest).

