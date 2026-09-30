#!/usr/bin/env Rscript

# Run through pixi: `pixi run install-ebm`
# The pixi environment provides R >= 4.4, cmake, expm, nloptr and numDeriv.
# FKF is not on conda-forge, so it comes from CRAN; EBM is Donald Cummins'
# package, pinned to v1.1.0.

repos <- "https://cloud.r-project.org"
install.packages("FKF", repos = repos)
install.packages(
  "https://github.com/donaldcummins/EBM/archive/refs/tags/v1.1.0.tar.gz",
  repos = NULL,
  type = "source"
)
library(EBM)
