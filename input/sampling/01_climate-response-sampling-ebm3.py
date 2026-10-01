#!/usr/bin/env python
# coding: utf-8

"""Climate response calibrations"""
# The purpose here is to provide correlated calibrations to the climate response in
# CMIP6 models.
#
# We will apply a very naive model weighting to the 4xCO2 results. We won't downweight
# for similar models*, but we will only select one ensemble member from models that
# provide multiple runs (the ensemble member that I deem the most reliable).
#
# *maybe the same model at different resolution should be downweighted.

import os
import warnings

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats
from dotenv import load_dotenv
from fair.energy_balance_model import EnergyBalanceModel
from tqdm import tqdm

from fair_calibrate.layers import (
    calibration_file,
    climate_response_columns,
    climate_response_file,
    get_n_layers,
)
from fair_calibrate.parameters import PRIOR_SAMPLES
from fair_calibrate.paths import ROOT

warnings.simplefilter("error", RuntimeWarning)

load_dotenv()
n_layers = get_n_layers()  # N_LAYERS in parameters.py: 2 or 3

print(f"Making {n_layers}-layer climate response calibrations...")

samples = PRIOR_SAMPLES
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
pl.style.use(f"{ROOT}/defaults.mplstyle")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")

df = pd.read_csv(f"{ROOT}/output/calibrations/" + calibration_file(n_layers))

# 02_calibrate_cummins.py marks fits to be wary of (epsilon outside 0.5-2.5, a
# constraint on its bound, the two starts disagreeing or only one converging).
# By default they stay in the sample, with a warning; EXCLUDE_SUSPECT=True drops them.
if "suspect" in df.columns and df["suspect"].any():
    suspect = df.loc[df["suspect"], ["model", "run", "suspect_reasons"]]
    if os.getenv("EXCLUDE_SUSPECT", "False").lower() in ("true", "1", "t"):
        print(f"EXCLUDE_SUSPECT: dropping {len(suspect)} suspect fits:")
        print(suspect.to_string(index=False))
        df = df.loc[~df["suspect"]]
    else:
        print(f"WARNING: {len(suspect)} suspect fits are in the sample "
              "(EXCLUDE_SUSPECT=True drops them):")
        print(suspect.to_string(index=False))
models = df["model"].unique()

# NorESM2-LM is currently INCLUDED. Comment below left in for train-of-thought.
# Executive decision: remove NorESM2-LM. It is always incredibly difficult to calibrate.
# Sometimes it fails completely, sometimes it gives nonsense values that wreck the rest
# of the distribution.
#
# if "NorESM2-LM" in models:
#     noresm2lm_index = np.argwhere(models=="NorESM2-LM")
#     np.delete(models, noresm2lm_index)
#     print("NorESM2-LM is being removed.")

for model in models:
    print(model, df.loc[df["model"] == model, "run"].values)

n_models = len(models)

multi_runs = {
    "GISS-E2-1-G": "r1i1p1f1",
    "GISS-E2-1-H": "r1i1p3f1",
    "MRI-ESM2-0": "r1i1p1f1",
    "EC-Earth3": "r3i1p1f1",
    "FIO-ESM-2-0": "r1i1p1f1",
    "CanESM5": "r1i1p2f1",
    "FGOALS-f3-L": "r1i1p1f1",
    "CNRM-ESM2-1": "r1i1p1f2",
}

params = {}

params[r"$\gamma$"] = np.ones(n_models) * np.nan
for i in range(1, n_layers + 1):
    params[f"$c_{i}$"] = np.ones(n_models) * np.nan
for i in range(1, n_layers + 1):
    params[rf"$\kappa_{i}$"] = np.ones(n_models) * np.nan
params[r"$\epsilon$"] = np.ones(n_models) * np.nan
params[r"$\sigma_{\eta}$"] = np.ones(n_models) * np.nan
params[r"$\sigma_{\xi}$"] = np.ones(n_models) * np.nan
params[r"$F_{4\times}$"] = np.ones(n_models) * np.nan

for im, model in enumerate(models):
    condition = df["model"] == model
    if model in multi_runs:
        preferred = condition & (df["run"] == multi_runs[model])
        if preferred.any():  # the preferred run may have been excluded as suspect
            condition = preferred
    params[r"$\gamma$"][im] = df.loc[condition, "gamma"].values[0]
    for i in range(1, n_layers + 1):
        params[f"$c_{i}$"][im] = df.loc[condition, f"C{i}"].values[0]
        params[rf"$\kappa_{i}$"][im] = df.loc[condition, f"kappa{i}"].values[0]
    params[r"$\epsilon$"][im] = df.loc[condition, "epsilon"].values[0]
    params[r"$\sigma_{\eta}$"][im] = df.loc[condition, "sigma_eta"].values[0]
    params[r"$\sigma_{\xi}$"][im] = df.loc[condition, "sigma_xi"].values[0]
    params[r"$F_{4\times}$"][im] = df.loc[condition, "F_4xCO2"].values[0]

params = pd.DataFrame(params)
print(params.corr())

if plots:
    fig = pl.figure(figsize=(18 / 2.54, 13 / 2.54))
    pd.plotting.scatter_matrix(params)
    pl.suptitle("Distributions and correlations of CMIP6 calibrations")
    pl.tight_layout()
    pl.subplots_adjust(wspace=0, hspace=0)
    os.makedirs(
        f"{ROOT}/plots/", exist_ok=True
    )
    pl.savefig(
        f"{ROOT}/plots/"
        f"ebm{n_layers}_distributions.png"
    )
    pl.savefig(
        f"{ROOT}/plots/"
        f"ebm{n_layers}_distributions.pdf"
    )
    pl.close()

NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)

kde = scipy.stats.gaussian_kde(params.T)
ebm_sample = kde.resample(size=int(samples * 4), seed=2181882)

# Row layout of the sample: gamma, C1..Cn, kappa1..kappan, epsilon, sigma_eta,
# sigma_xi, F_4xCO2.
i_kappa1 = n_layers + 1
i_epsilon = 2 * n_layers + 1
i_sigma_eta = 2 * n_layers + 2
i_sigma_xi = 2 * n_layers + 3
i_f4xco2 = 2 * n_layers + 4

# remove unphysical combinations
for col in range(i_f4xco2):
    ebm_sample[:, ebm_sample[col, :] <= 0] = np.nan
ebm_sample[:, ebm_sample[0, :] <= 0.5] = np.nan  # gamma
ebm_sample[:, ebm_sample[1, :] <= 1.8] = np.nan  # C1
for layer in range(2, n_layers + 1):  # each layer larger than the one above
    ebm_sample[:, ebm_sample[layer, :] <= ebm_sample[layer - 1, :]] = np.nan
ebm_sample[:, ebm_sample[i_kappa1, :] <= 0.3] = np.nan  # kappa1 = lambda

mask = np.all(np.isnan(ebm_sample), axis=0)
ebm_sample = ebm_sample[:, ~mask]

# check that covariance matrix is positive semidefinite and if not, remove param combo.
# to do: change away from sparse, once we move away from R
for isample in tqdm(range(len(ebm_sample.T)), disable=1 - progress):
    ebm = EnergyBalanceModel(
        ocean_heat_capacity=ebm_sample[1 : n_layers + 1, isample],
        ocean_heat_transfer=ebm_sample[i_kappa1:i_epsilon, isample],
        deep_ocean_efficacy=ebm_sample[i_epsilon, isample],
        gamma_autocorrelation=ebm_sample[0, isample],
        sigma_xi=ebm_sample[i_sigma_xi, isample],
        sigma_eta=ebm_sample[i_sigma_eta, isample],
        forcing_4co2=ebm_sample[i_f4xco2, isample],
        stochastic_run=True,
    )
    eb_matrix = ebm._eb_matrix()
    n_state = n_layers + 1
    q_mat = np.zeros((n_state, n_state))
    q_mat[0, 0] = ebm.sigma_eta**2
    q_mat[1, 1] = (ebm.sigma_xi / ebm.ocean_heat_capacity[0]) ** 2
    h_mat = np.zeros((2 * n_state, 2 * n_state))
    h_mat[:n_state, :n_state] = -eb_matrix
    h_mat[:n_state, n_state:] = q_mat
    h_mat[n_state:, n_state:] = eb_matrix.T
    g_mat = scipy.sparse.linalg.expm(h_mat)
    q_mat_d = g_mat[n_state:, n_state:].T @ g_mat[:n_state, n_state:]
    q_mat_d = q_mat_d.astype(np.float64)

    # I can't work out exactly what checks scipy is doing to decide the param
    # set is a fail. Best to just let it tell me if it likes it or not.
    try:
        scipy.stats.multivariate_normal.rvs(
            size=1, mean=np.zeros(n_state), cov=q_mat_d
        )
    except:  # noqa: E722
        ebm_sample[:, isample] = np.nan

mask = np.all(np.isnan(ebm_sample), axis=0)
ebm_sample = ebm_sample[:, ~mask]

print("Total number of retained samples:", len(ebm_sample.T))

ebm_sample_df = pd.DataFrame(
    data=ebm_sample[:, :samples].T,
    columns=climate_response_columns(n_layers),
)

assert len(ebm_sample_df) >= samples

os.makedirs(
    f"{ROOT}/output/priors/",
    exist_ok=True,
)

ebm_sample_df.to_csv(
    f"{ROOT}/output/priors/" + climate_response_file(n_layers),
    index=False,
)

# what we do want to do is to scale the variability in 4xCO2 (correlated with the other
# EBM parameters)
# to feed into the effective radiative forcing scaling factor.
