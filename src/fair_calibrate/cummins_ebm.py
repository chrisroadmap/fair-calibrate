"""Maximum-likelihood fit of the Cummins two- and three-layer energy balance models.

Python port of ``EBM::FitKalman`` (donaldcummins/EBM v1.1.0,
https://doi.org/10.5281/zenodo.5217975) for two and three layers, following
Cummins, Stephenson & Stott (2020), Journal of Climate 33(18), 7909-7926,
https://doi.org/10.1175/JCLI-D-19-0589.1.

The model is fitted to a global-mean surface temperature (T) and top-of-
atmosphere imbalance (N) response to an abrupt 4xCO2 forcing. Parameters are
optimised in log space with BOBYQA (NLopt, as in R's ``nloptr::bobyqa``) on the
negative Kalman-filter log-likelihood plus a quadratic penalty on the deepest
heat capacity.
"""

import nlopt
import numpy as np
import scipy.linalg

SUPPORTED_LAYERS = (2, 3)
# Parameters: gamma, C[n], kappa[n], epsilon, sigma_eta, sigma_xi, F_4xCO2
N_SCALARS = 5  # all but C and kappa

# Measurement noise variance used by EBM for both T and N.
MEASUREMENT_VARIANCE = 1e-12


class ConvergenceError(RuntimeError):
    """The optimiser failed or ran out of iterations."""


def transform(p):
    """Flatten a parameter dict to the log-space optimiser vector."""
    return np.log(
        np.concatenate(
            [
                [p["gamma"]],
                p["C"],
                p["kappa"],
                [p["epsilon"], p["sigma_eta"], p["sigma_xi"], p["F_4xCO2"]],
            ]
        ).astype(float)
    )


def back_transform(par):
    """Inverse of :func:`transform`."""
    par = np.exp(np.asarray(par, dtype=float))
    n = (par.size - N_SCALARS) // 2
    if par.size != 2 * n + N_SCALARS or n not in SUPPORTED_LAYERS:
        raise ValueError(
            f"expected 9 (two layers) or 11 (three layers) parameters, got {par.size}"
        )
    return {
        "gamma": par[0],
        "C": par[1 : 1 + n],
        "kappa": par[1 + n : 1 + 2 * n],
        "epsilon": par[1 + 2 * n],
        "sigma_eta": par[2 + 2 * n],
        "sigma_xi": par[3 + 2 * n],
        "F_4xCO2": par[4 + 2 * n],
    }


def build_matrices(gamma, C, kappa, epsilon, sigma_eta, sigma_xi):
    """Continuous and discretised state-space matrices (EBM ``BuildMatrices``).

    The first state is the AR(1) forcing/noise state with autocorrelation
    ``gamma``; the others are the layer temperatures (two or three layers,
    from the length of ``C``). ``kappa[0]`` is the climate feedback and
    ``kappa[i]`` the heat exchange between layers ``i - 1`` and ``i``. The
    deep-ocean efficacy ``epsilon`` multiplies the exchange with the deepest
    layer, as seen from the layer above it.
    """
    C = np.asarray(C, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    n = C.size
    if n not in SUPPORTED_LAYERS or kappa.size != n:
        raise ValueError("C and kappa must both have 2 or 3 elements")
    k = n + 1

    # efficacy of the transfer from layer i to layer i + 1, seen from layer i
    efficacy = np.ones(n - 1)
    efficacy[-1] = epsilon

    A = np.zeros((k, k))
    A[0, 0] = -gamma
    A[1, 0] = 1 / C[0]  # the forcing state drives the surface layer
    for i in range(n):  # layer i is state i + 1
        below = efficacy[i] * kappa[i + 1] if i < n - 1 else 0.0
        A[i + 1, i + 1] = -(kappa[i] + below) / C[i]
        if i > 0:
            A[i + 1, i] = kappa[i] / C[i]
        if i < n - 1:
            A[i + 1, i + 2] = below / C[i]
    B = np.zeros(k)
    B[0] = gamma
    Q = np.zeros((k, k))
    Q[0, 0] = sigma_eta**2
    Q[1, 1] = (sigma_xi / C[0]) ** 2

    Ad = scipy.linalg.expm(A)
    Bd = np.linalg.solve(A, (Ad - np.identity(k)) @ B)

    # Stationary state covariance from the continuous Lyapunov equation
    # A P + P A' + Q = 0, and the discretised process noise covariance
    # Qd = P - Ad P Ad'. EBM gets Qd from the Van Loan (1978) matrix exponential
    # and Gamma0 from Qd, which is the same quantity in exact arithmetic. But
    # scipy's expm loses all precision on the Van Loan matrix once gamma is
    # large (exp(gamma) multiplies rounding error in entries that should be
    # ~exp(-gamma)): Qd comes out indefinite and the filter fails. R's expm
    # copes; the Lyapunov route is stable in both.
    gamma0 = scipy.linalg.solve_continuous_lyapunov(A, -Q)
    Qd = gamma0 - Ad @ gamma0 @ Ad.T

    # Measurement matrix: rows are T1 and N
    Cd = np.zeros((2, k))
    Cd[0, 1] = 1
    Cd[1, 0] = 1
    Cd[1, 1] = -kappa[0]
    Cd[1, n - 1] += (1 - epsilon) * kappa[n - 1]
    Cd[1, n] -= (1 - epsilon) * kappa[n - 1]
    return {
        "A": A,
        "B": B,
        "Q": Q,
        "Ad": Ad,
        "Bd": Bd,
        "Qd": Qd,
        "Gamma0": gamma0,
        "Cd": Cd,
    }


def kalman_loglik(Ad, Bd, Qd, Gamma0, Cd, F_4xCO2, dataset):
    """Kalman-filter log-likelihood (EBM ``KalmanFilter`` via ``FKF::fkf``).

    Parameters
    ----------
    dataset : `np.ndarray`
        Array of shape (2, n_years): temperature anomaly T, then TOA
        imbalance N.
    """
    k = Ad.shape[0]
    dataset = np.asarray(dataset, dtype=float)
    n_obs, n_time = dataset.shape
    ggt = MEASUREMENT_VARIANCE * np.identity(n_obs)

    x0 = np.zeros(k)
    x0[0] = F_4xCO2
    dt = Bd * F_4xCO2
    a = Ad @ x0 + dt  # predicted state for the first observation
    P = Gamma0

    loglik = -0.5 * n_obs * n_time * np.log(2 * np.pi)
    for t in range(n_time):
        v = dataset[:, t] - Cd @ a
        F = Cd @ P @ Cd.T + ggt
        sign, logdet = np.linalg.slogdet(F)
        if sign <= 0:
            return -np.inf
        PZt = P @ Cd.T
        F_inv_v = np.linalg.solve(F, v)
        loglik -= 0.5 * (logdet + v @ F_inv_v)
        a_filt = a + PZt @ F_inv_v
        P_filt = P - PZt @ np.linalg.solve(F, PZt.T)
        a = dt + Ad @ a_filt
        P = Ad @ P_filt @ Ad.T + Qd
    return loglik


def neg_log_lik(par, dataset, alpha):
    """Penalised negative log-likelihood (EBM ``KalmanNegLogLik``)."""
    p = back_transform(par)
    m = build_matrices(
        p["gamma"], p["C"], p["kappa"], p["epsilon"], p["sigma_eta"], p["sigma_xi"]
    )
    loglik = kalman_loglik(
        m["Ad"], m["Bd"], m["Qd"], m["Gamma0"], m["Cd"], p["F_4xCO2"], dataset
    )
    return -loglik + alpha * p["C"][-1] ** 2


def fit_kalman(
    inits,
    T1,
    N,
    alpha=0.0,
    maxeval=100000,
    xtol_rel=1e-6,
    gamma_max=None,
    deep_min_ratio=None,
):
    """Fit the two- or three-layer model (EBM ``FitKalman``, point estimate only).

    The number of layers is the length of ``inits["C"]``.

    Parameters
    ----------
    inits : dict
        Initial parameter values, keys as returned by :func:`back_transform`.
    T1, N : array-like
        Temperature anomaly and TOA imbalance time series.
    alpha : float
        Weight of the quadratic penalty on the deepest layer heat capacity.
    maxeval : int
        Maximum number of objective evaluations.
    xtol_rel : float
        Relative parameter tolerance (nloptr's default is 1e-6).
    gamma_max : float, optional
        Upper bound on gamma, the forcing autocorrelation rate (yr-1). For some
        runs the likelihood keeps improving as gamma grows without limit (the
        white-noise limit, which annual data cannot distinguish from a large
        finite gamma), so the fit only stops at ``maxeval`` or at an absurd gamma.
        EBM has no bound; it stops near gamma = 25-30 only because its matrix
        exponential loses accuracy there.
    deep_min_ratio : float, optional
        Lower bound on the heat capacity of the deepest layer divided by that
        of the layer above it, so ``deep_min_ratio=1`` enforces C3 >= C2 for
        three layers (C2 >= C1 for two). Without it, three-layer fits can
        collapse C3 to almost nothing while the deep-ocean efficacy epsilon runs
        off to 100 or more. Enforced exactly: the optimiser works on
        log(C_deep / C_above), which BOBYQA can bound directly.

    Returns
    -------
    dict
        The fitted parameters (same keys as ``inits``), plus ``nit`` (number
        of objective evaluations), ``neg_log_lik`` (the minimised value),
        ``gamma_at_bound`` (the fit ended on ``gamma_max``) and
        ``deep_at_bound`` (the fit ended on ``deep_min_ratio``).

    Raises
    ------
    ConvergenceError
        If iterations are exhausted or the optimiser reports failure.
    ValueError
        If ``gamma_max`` or ``deep_min_ratio`` is set and the initial values do
        not satisfy it strictly.
    """
    dataset = np.vstack([np.asarray(T1, dtype=float), np.asarray(N, dtype=float)])
    x0 = transform(inits)
    n = len(inits["C"])
    if gamma_max is not None and inits["gamma"] >= gamma_max:
        raise ValueError(
            f"initial gamma ({inits['gamma']}) must be below gamma_max ({gamma_max})"
        )
    c_ratio = inits["C"][-1] / inits["C"][-2]
    if deep_min_ratio is not None and c_ratio <= deep_min_ratio:
        raise ValueError(
            f"initial deep/above heat capacity ratio ({c_ratio:g}) must exceed "
            f"deep_min_ratio ({deep_min_ratio})"
        )

    # The optimiser's variables are the log parameters, except that log(C_deep)
    # is replaced by log(C_deep / C_above) when that ratio is bounded. In the
    # log-parameter vector C_i sits at index i, so C_deep is n, C_above n - 1.
    def to_optimiser(x):
        y = x.copy()
        if deep_min_ratio is not None:
            y[n] = x[n] - x[n - 1]
        return y

    def from_optimiser(y):
        x = np.array(y, dtype=float)
        if deep_min_ratio is not None:
            x[n] = y[n] + y[n - 1]
        return x

    def objective(y, grad):
        return float(neg_log_lik(from_optimiser(y), dataset, alpha))

    opt = nlopt.opt(nlopt.LN_BOBYQA, x0.size)
    opt.set_min_objective(objective)
    opt.set_xtol_rel(xtol_rel)
    opt.set_maxeval(maxeval)
    lower = np.full(x0.size, -np.inf)
    upper = np.full(x0.size, np.inf)
    if gamma_max is not None:
        upper[0] = np.log(gamma_max)  # gamma is the first parameter, in log space
    if deep_min_ratio is not None:
        lower[n] = np.log(deep_min_ratio)
    if gamma_max is not None or deep_min_ratio is not None:
        opt.set_lower_bounds(lower)
        opt.set_upper_bounds(upper)
    try:
        y = opt.optimize(to_optimiser(x0))
    except (RuntimeError, ValueError) as exc:  # nlopt failure codes raise
        raise ConvergenceError(f"Convergence failure: {exc}") from exc

    nit = opt.get_numevals()
    if nit >= maxeval:
        raise ConvergenceError("Iterations exhausted.")
    result = back_transform(from_optimiser(y))
    result["nit"] = nit
    result["neg_log_lik"] = opt.last_optimum_value()
    result["gamma_at_bound"] = bool(
        gamma_max is not None and result["gamma"] >= gamma_max * (1 - 1e-6)
    )
    result["deep_at_bound"] = bool(
        deep_min_ratio is not None
        and result["C"][-1] / result["C"][-2] <= deep_min_ratio * (1 + 1e-6)
    )
    return result


EPSILON_RANGE = (0.5, 2.5)
START_TOL = 0.1  # log-likelihood units


def suspect_reasons(result, start_gap=None, epsilon_range=EPSILON_RANGE):
    """List why a fit should be treated with suspicion (empty if it looks fine).

    Parameters
    ----------
    result : dict
        A :func:`fit_kalman` result.
    start_gap : float, optional
        Absolute difference in objective between fits from two starting points
        (NaN if only one converged; ``None`` skips the check).
    epsilon_range : tuple of float
        Plausible range of the deep-ocean efficacy epsilon. Values outside it
        are usually a trade-off with the layer heat capacities, not a property
        of the climate model.
    """
    reasons = []
    low, high = epsilon_range
    if result["epsilon"] < low:
        reasons.append(f"epsilon below {low:g}")
    if result["epsilon"] > high:
        reasons.append(f"epsilon above {high:g}")
    if result.get("deep_at_bound"):
        reasons.append("deepest/above heat capacity ratio on its bound")
    if start_gap is not None:
        if np.isnan(start_gap):
            reasons.append("only one start converged")
        elif start_gap >= START_TOL:
            reasons.append(f"starts differ by {start_gap:.2g}")
    return reasons
