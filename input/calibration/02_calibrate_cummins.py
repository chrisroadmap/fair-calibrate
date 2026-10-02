#!/usr/bin/env python
# coding: utf-8

"""Calibrate the Cummins energy balance model to CMIP6 4xCO2 runs."""

# Goes through each of the models in turn and tunes the parameters of the
# Cummins two or three layer model (N_LAYERS in parameters.py). Python port of
# the R script 02_calibrate_cummins_3layer.r, which used Donald Cummins' R package
# (EBM); the R version is kept in alternatives/cummins/.
#
# It will produce an output csv table of Cummins parameters
# (4xCO2_cummins_ebm<N_LAYERS>_cmip6.csv), which we will then put in
# impulse-response form for FaIR.
#
# References:
# Cummins, D. P., Stephenson, D. B., & Stott, P. A. (2020). Optimal
# Estimation of Stochastic Energy Balance Model Parameters, Journal of Climate,
# 33(18), 7909-7926, https://doi.org/10.1175/JCLI-D-19-0589.1
#
# donaldcummins. (2021). donaldcummins/EBM: Optional quadratic penalty
# (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.5217975

import os
from fair_calibrate.paths import ROOT

# One fit is a long loop over small matrices, which multithreaded BLAS cannot speed
# up. With several worker processes the BLAS thread pools oversubscribe the CPUs
# and the fits crawl, so pin each worker to one thread. Must precede numpy import.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from fair_calibrate.cummins_ebm import (  # noqa: E402
    START_TOL,
    ConvergenceError,
    fit_kalman,
    suspect_reasons,
)
from fair_calibrate.layers import get_n_layers  # noqa: E402

load_dotenv()

workers = int(os.getenv("WORKERS", "1"))

# Number of ocean layers in the energy balance model: N_LAYERS in parameters.py.
n_layers = get_n_layers()

# Upper bound on gamma, the stochastic forcing autocorrelation rate (yr-1). For
# about a fifth of the CMIP6 runs the exact likelihood keeps improving as gamma
# grows without limit (the white-noise limit, which annual data cannot tell from
# a large finite gamma), so an unbounded fit never converges. R's EBM only
# stopped at gamma = 24-30 because its matrix exponential loses accuracy there.
# 10 reproduces R on the other runs; 30 let BOBYQA stop early on several of
# them. Set GAMMA_MAX in .env to change it, or to "none" for no bound.
GAMMA_MAX_DEFAULT = "10"
_gamma_max = os.getenv("GAMMA_MAX", GAMMA_MAX_DEFAULT).strip().lower()
gamma_max = None if _gamma_max in ("", "none") else float(_gamma_max)

# Lower bound on the heat capacity of the deepest layer divided by that of the
# layer above it (C3/C2 for three layers, C2/C1 for two): the deep ocean must be at
# least as large as the layer above. Without it, 5 of 66 three-layer fits
# collapsed C3 to between 0.001 and 1.3 while the deep-ocean efficacy epsilon ran
# to 100-28,000. Set DEEP_MIN_RATIO in .env to change it, or to "none" for no
# constraint. Fits with epsilon outside EPSILON_RANGE (0.5-2.5) are marked
# suspect in the output rather than bounded.
DEEP_MIN_RATIO_DEFAULT = "1"
_deep_min_ratio = os.getenv("DEEP_MIN_RATIO", DEEP_MIN_RATIO_DEFAULT).strip().lower()
deep_min_ratio = None if _deep_min_ratio in ("", "none") else float(_deep_min_ratio)

N_YEARS = 150
MAX_ATTEMPTS = 5
MAXEVAL = 20000

# Initial guess for parameter values. The two-layer values are those the R script
# calibrate_cummins_2layer.r used.
INITS = {
    3: dict(
        gamma=2,
        C=np.array([4, 15, 80]),
        kappa=np.array([1, 2, 1]),
        epsilon=1.1,
        sigma_eta=0.5,
        sigma_xi=0.5,
        F_4xCO2=8,
    ),
    2: dict(
        gamma=2,
        C=np.array([7.5, 75]),
        kappa=np.array([1, 0.8]),
        epsilon=1.2,
        sigma_eta=0.5,
        sigma_xi=0.5,
        F_4xCO2=8,
    ),
}

# A second, different starting point. BOBYQA from one start can stop early at a
# poor point without complaint (several fits ended 20-120 log-likelihood units
# below R's), so every run is fitted from both starts. The first start's result
# is kept unless the second is better by more than START_TOL; the gap is saved.
INITS_ALT = {
    3: dict(
        gamma=5,
        C=np.array([8, 25, 100]),
        kappa=np.array([1.2, 1.5, 0.8]),
        epsilon=1.0,
        sigma_eta=1.0,
        sigma_xi=0.3,
        F_4xCO2=7,
    ),
    2: dict(
        gamma=5,
        C=np.array([10, 100]),
        kappa=np.array([1.3, 0.6]),
        epsilon=1.0,
        sigma_eta=1.0,
        sigma_xi=0.3,
        F_4xCO2=7,
    ),
}

COLUMNS = (
    ["model", "run", "conv", "nit", "gamma"]
    + [f"C{i}" for i in range(1, n_layers + 1)]
    + [f"kappa{i}" for i in range(1, n_layers + 1)]
    + [
        "epsilon",
        "sigma_eta",
        "sigma_xi",
        "F_4xCO2",
        "gamma_at_bound",
        "start_gap",
        "deep_at_bound",
        "suspect",
        "suspect_reasons",
    ]
)


def fit_from_both_starts(model, run, tas, rndt, alpha):
    """Fit from both starting points; return (chosen result, gap between starts).

    The gap is the absolute difference in the penalised objective between the
    two converged fits, or NaN when only one start converged. Raises
    ConvergenceError when neither does.
    """
    results = []
    for name, inits in (
        ("start 1", INITS[n_layers]),
        ("start 2", INITS_ALT[n_layers]),
    ):
        try:
            results.append(
                fit_kalman(
                    inits,
                    tas,
                    rndt,
                    alpha=alpha,
                    maxeval=MAXEVAL,
                    gamma_max=gamma_max,
                    deep_min_ratio=deep_min_ratio,
                )
            )
        except (ConvergenceError, np.linalg.LinAlgError) as exc:
            print(f"{model} {run}: {name} did not converge ({exc})", flush=True)
            results.append(None)
    first, second = results
    if first is None and second is None:
        raise ConvergenceError("neither start converged")
    if first is None or second is None:
        return (first or second), np.nan
    gap = abs(first["neg_log_lik"] - second["neg_log_lik"])
    if second["neg_log_lik"] < first["neg_log_lik"] - START_TOL:
        print(
            f"{model} {run}: start 2 is better by {gap:.2f}, using it",
            flush=True,
        )
        return second, gap
    return first, gap


def calibrate_run(job):
    """Fit one model/run, retrying with a more liberal penalty on failure.

    Returns a row of the output table, or None if it never converged.
    """
    model, run, tas, rndt = job
    for attempt in range(MAX_ATTEMPTS):
        alpha = 1e-05 * 10**attempt
        try:
            print(
                f"{model} {run}: attempt {attempt + 1}/{MAX_ATTEMPTS}, alpha={alpha:g}",
                flush=True,
            )
            result, gap = fit_from_both_starts(model, run, tas, rndt, alpha)
        except ConvergenceError as exc:
            print(
                f"{model} {run}: did not converge or ran out of iterations ({exc})",
                flush=True,
            )
            continue
        print(f"{model} {run}: converged in {result['nit']} iterations", flush=True)
        reasons = suspect_reasons(result, start_gap=gap)
        if reasons:
            print(f"{model} {run}: SUSPECT ({'; '.join(reasons)})", flush=True)
        return [
            model,
            run,
            True,
            result["nit"],
            result["gamma"],
            *result["C"],
            *result["kappa"],
            result["epsilon"],
            result["sigma_eta"],
            result["sigma_xi"],
            result["F_4xCO2"],
            result["gamma_at_bound"],
            gap,
            result["deep_at_bound"],
            bool(reasons),
            "; ".join(reasons),
        ]
    print(f"I am excluding {model} {run} from my table of results.")
    return None


if __name__ == "__main__":
    print(f"Running Python script for {n_layers} layer model calibrations...")

    # Get the precalculated 4xCO2 N and T data
    input_data = pd.read_csv(f"{ROOT}/output/calibrations/4xCO2_cmip6.csv")
    year_columns = input_data.columns[9 : 9 + N_YEARS]

    # iterate through runs of the same model for now, though we will probably
    # want to combine/downweight this in the final edition
    jobs = []
    for model in input_data["climate_model"].unique():
        in_model = input_data["climate_model"] == model
        tas_rows = input_data[in_model & (input_data["variable"] == "tas")]
        for run in tas_rows["member_id"]:
            in_run = in_model & (input_data["member_id"] == run)
            tas = input_data.loc[
                in_run & (input_data["variable"] == "tas"), year_columns
            ]
            rndt = input_data.loc[
                in_run & (input_data["variable"] == "rndt"), year_columns
            ]
            if len(tas) != 1 or len(rndt) != 1:
                print(f"{model} {run}: needs one tas and one rndt row, skipping")
                continue
            jobs.append((model, run, tas.values.squeeze(), rndt.values.squeeze()))

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(calibrate_run, jobs))
    else:
        rows = [calibrate_run(job) for job in jobs]

    output = pd.DataFrame([row for row in rows if row is not None], columns=COLUMNS)

    os.makedirs(f"{ROOT}/output/calibrations/", exist_ok=True)
    output.to_csv(
        f"{ROOT}/output/calibrations/4xCO2_cummins_ebm{n_layers}_cmip6.csv",
        index=False,
    )
