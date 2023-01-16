\name{adaptiveHardThresholding}
\alias{adaptiveHardThresholding}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
adaptiveHardThresholding
}
\description{
Performs optimal adaptive hard thresholding on the input matrix Y.
}
\usage{
adaptiveHardThresholding(Y, k, strategy='i')
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Y}{
  A data matrix, on whose singular values thresholding should be performed.
}
  \item{k}{
  An upper bound (potentially loose) on the latent signal rank. That is, the procedure assumes that
  there are AT MOST k informative principal components of Y.
  }
  \item{strategy}{
  Method for reconstructing the noise bulk (optional). Can be one of the following:
  '0' (tranpsort to zero),
  'w' (winsorization),
  'i' (imputation).
  Default value = 'i'.
  }
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
  \item{Xest}{An estimate of the low-rank signal. That is: the matrix obtained
  by thresholding the singular values of Y.}
  \item{Topt}{The hard threshold computed by the procedure. To wit, the procedure
  retains the i-th PC of Y if and only if the corresponding singular value, y_i,
  satisifies
  y_i > Topt.}
  \item{r}{The number of "relevant" components: r = rank(Xest).}
}
\references{
David L. Donoho, Matan Gavish and Elad Romanov.
"ScreeNot: Exact MSE-optimal singular value thresholding in correlated noise."
Annals of Statistics (2023).

\url{https://github.com/eladromanov/ScreeNOT}
}
\author{
Elad Romanov
}
