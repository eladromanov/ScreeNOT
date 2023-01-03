# *ScreeNOT*: Optimal Singular Value Thresholding in Correlated Noise

ScreeNOT is a code library for hard thresholding of singular values. The procedure adaptively estimates the best singular value threshold under unknown noise characteristics. The threshold chosen by ScreeNOT is optimal (asymptotically, in the sense of minimum Frobenius error) under the the so-called "Spiked model" of a low-rank matrix observed in additive noise. In contrast to previous works, the noise is *not* assumed to be i.i.d. or white; it can have an essentially arbitrary and **unknown** correlation structure, across either rows, columns or both. 
ScreeNOT is proposed to practitioners as a mathematically solid alternative to Cattell's ever-popular but vague Scree Plot heuristic from 1966.

If you use this package, please cite our paper:
* David L. Donoho, Matan Gavish, and Elad Romanov. 
"ScreeNot: Exact MSE-optimal singular value thresholding in correlated noise." 
arXiv preprint arXiv:2009.12297 (2020).

**TODO: Paper accepted for publication in the Annals of Statistics. This reference will be updated once it appears.**


## Installing this package
ScreeNOT is available for Python (source code or package via PyPI), R (source code or package via CRAN) and Matlab (source code only).
Source code can be downloaded directly from this repository. It is also permanently deposited at the Stanford Digital Repository: https://purl.stanford.edu/py196rk3919

### Python:

* PyPI: `pip install screenot`   (package: https://pypi.org/project/screenot/)
* Source code:  ./Python/src/screenot/ScreeNOT.py

### R:

* CRAN: *To be updated soon!*
* Source code: ./R/ScreeNOT.R

### Matlab:

* Source code: ./Matlab/ScreeNOT.m

## Usage

The main API for this package is the function

`adaptiveHardThresholding(Y, k, strategy='i')`

It receives as input a matrix `Y` (in Python: a `numpy` array); a non-negative integer `k`, an upper bound (potentially loose) on the low-rank signal to-be-recovered; (optional) a string `strategy`, which defines an estimation strategy for the noise bulk - valid options are `'i'` (imputation; this is the default, and should be best for most practical uses), `'0'` (transport to zero) or `'w'` (winsorization) - see paper for details.

The function returns three values:

`Xest`: a matrix with same dimensions as `Y`; an estimator for the low-rank signal (the result of performing thresholding on the singular values of `Y`).
`Topt`: the hard threshold which was used; computed adaptively from the singular values of `Y`.
`r`: the number of "relevant"/"strong" components of the signal: `r=rank(Xest)`.

## Example

For usage example in `Python`, please consult the jupyter notebook "ScreeNOTExample.ipynb" in this repo.
