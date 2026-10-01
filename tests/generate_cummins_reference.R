#!/usr/bin/env Rscript
# Regenerate tests/data/cummins_reference.json from Donald Cummins' EBM package.
# Run from the repository root:  pixi run Rscript tests/generate_cummins_reference.R
#
# The synthetic series in tests/data/cummins_synthetic.csv was simulated from
# the three-layer model with the parameters below (numpy seed 42).

library(EBM)
library(jsonlite)

d <- read.csv("tests/data/cummins_synthetic.csv")
ds <- rbind(d$T, d$N)

truth <- list(
  gamma = 1.8, C = c(7.5, 25, 110), kappa = c(1.1, 2.3, 0.9),
  epsilon = 1.2, sigma_eta = 0.45, sigma_xi = 0.55, F_4xCO2 = 7.6
)
alpha <- 1e-5
par <- EBM:::Transform(truth)
p <- EBM:::BackTransform(par)
m <- with(p, EBM:::BuildMatrices(gamma, C, kappa, epsilon, sigma_eta, sigma_xi))
kf <- with(c(p, m), EBM:::KalmanFilter(Ad, Bd, Qd, Gamma0, Cd, F_4xCO2, ds))

inits3 <- list(
  gamma = 2, C = c(4, 15, 80), kappa = c(1, 2, 1), epsilon = 1.1,
  sigma_eta = 0.5, sigma_xi = 0.5, F_4xCO2 = 8
)
fit <- FitKalman(inits3, T1 = d$T, N = d$N, alpha = alpha, maxeval = 20000)

# High-gamma case: the R fit of FGOALS-f3-L r3i1p1f1 (gamma = 30). scipy's
# matrix exponential gives an indefinite Qd here via the Van Loan route, which is
# why fair_calibrate.cummins_ebm uses the Lyapunov equation instead. R's own Qd
# is only accurate to about 1e-5 at this gamma, so tests use loose tolerances.
stress <- list(
  gamma = 30.40906, C = c(1.020977, 10.19185, 67.75533),
  kappa = c(1.46019, 8.69747, 0.6793861), epsilon = 1.438151,
  sigma_eta = 2.479109, sigma_xi = 0.5120712, F_4xCO2 = 9.326111
)
spar <- EBM:::Transform(stress)
sp <- EBM:::BackTransform(spar)
sm <- with(sp, EBM:::BuildMatrices(gamma, C, kappa, epsilon, sigma_eta, sigma_xi))
skf <- with(c(sp, sm), EBM:::KalmanFilter(Ad, Bd, Qd, Gamma0, Cd, F_4xCO2, ds))

# Two-layer model (EBM supports k = 2 natively): fixed-parameter matrices and
# likelihood, and a fit from the starting values calibrate_cummins_2layer.r used,
# all on the same synthetic series.
two <- list(
  gamma = 1.8, C = c(7.5, 60), kappa = c(1.1, 0.8),
  epsilon = 1.2, sigma_eta = 0.45, sigma_xi = 0.55, F_4xCO2 = 7.6
)
tpar <- EBM:::Transform(two)
tp <- EBM:::BackTransform(tpar)
tm <- with(tp, EBM:::BuildMatrices(gamma, C, kappa, epsilon, sigma_eta, sigma_xi))
tkf <- with(c(tp, tm), EBM:::KalmanFilter(Ad, Bd, Qd, Gamma0, Cd, F_4xCO2, ds))
inits2 <- list(
  gamma = 2, C = c(7.5, 75), kappa = c(1, 0.8), epsilon = 1.2,
  sigma_eta = 0.5, sigma_xi = 0.5, F_4xCO2 = 8
)
fit2 <- FitKalman(inits2, T1 = d$T, N = d$N, alpha = alpha, maxeval = 20000)

out <- list(
  two_layer = list(
    truth = two,
    par = tpar,
    matrices = tm[c("A", "Ad", "Bd", "Qd", "Gamma0", "Cd")],
    loglik = tkf$logLik,
    neg_log_lik = EBM:::KalmanNegLogLik(tpar, ds, alpha),
    inits = inits2,
    fit = list(p = fit2$p, neg_log_lik = fit2$AIC / 2 - length(fit2$mle))
  ),
  stress = list(
    truth = stress,
    par = spar,
    matrices = sm[c("Ad", "Qd", "Gamma0", "Cd")],
    loglik = skf$logLik
  ),
  alpha = alpha,
  truth = truth,
  par = par,
  matrices = m[c("A", "B", "Q", "Ad", "Bd", "Qd", "Gamma0", "Cd")],
  loglik = kf$logLik,
  neg_log_lik = EBM:::KalmanNegLogLik(par, ds, alpha),
  fit = list(
    p = fit$p,
    neg_log_lik = fit$AIC / 2 - length(fit$mle)
  )
)
write(toJSON(out, digits = NA, auto_unbox = TRUE, matrix = "rowmajor"),
      "tests/data/cummins_reference.json")
cat("wrote tests/data/cummins_reference.json\n")
