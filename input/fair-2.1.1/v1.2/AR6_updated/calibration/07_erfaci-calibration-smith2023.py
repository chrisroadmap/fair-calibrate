#!/usr/bin/env python
# coding: utf-8

"""Sample aerosol indirect."""

# # Using the fair-2.1 pure log formula
#
# **Note**
# Estimating aerosol cloud interactions from 11 CMIP6 models was performed in Smith
#  et al. 2021: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JD033622.
#
# The underlying APRP code was slightly wrong, and has been updated thanks to Mark
# Zelinka (released as climateforcing v0.3.0). Two more models are now available.
# Actually three are, but EC-Earth3 is unusable due to unphysical values of rsuscs and
# rsdscs leading to biased ERFaci estimates.
#
# \begin{equation}
# F = \beta \log \left( 1 + \sum_{i} n_i A_i \right)
# \end{equation}
#
# where
# - $A_i$ is the atmospheric input (concentrations or emissions of a specie),
# - $\beta_i$ is a scale factor,
# - $n_i$ dictates how much emissions of a specie contributes to CDNC.
#
# **Note also** the uniform prior from -2 to 0. A lot of the sublteties here might also
# want to go into the paper.


import glob
import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import pooch
from dotenv import load_dotenv
from scipy.optimize import curve_fit

load_dotenv()

pl.style.use("../../../../../defaults.mplstyle")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")

print("Calibrating aerosol cloud interactions...")


files = glob.glob("../../../../../data/smith2023aerosol/*.csv")

ari = {}
aci = {}
models = []
models_runs = {}
years = {}
for file in files:
    model = os.path.split(file)[1].split("_")[0]
    run = os.path.split(file)[1].split("_")[1]
    models.append(model)
    if run not in models_runs:
        models_runs[model] = []
    models_runs[model].append(run)

models = list(models_runs.keys())

for model in models:
    nruns = 0
    for run in models_runs[model]:
        file = f"../../../../../data/smith2023aerosol/{model}_{run}_aerosol_forcing.csv"
        df = pd.read_csv(file, index_col=0)
        if nruns == 0:
            ari_temp = df["ERFari"].values.squeeze()
            aci_temp = df["ERFaci"].values.squeeze()
        else:
            ari_temp = ari_temp + df["ERFari"].values.squeeze()
            aci_temp = aci_temp + df["ERFaci"].values.squeeze()
        years[model] = df.index + 0.5
        nruns = nruns + 1
    ari[model] = ari_temp / nruns
    aci[model] = aci_temp / nruns


rcmip_emissions_file = pooch.retrieve(
    url="doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
)

emis_df = pd.read_csv(rcmip_emissions_file)

bc = (
    emis_df.loc[
        (emis_df["Scenario"] == "ssp245")
        & (emis_df["Region"] == "World")
        & (emis_df["Variable"] == "Emissions|BC"),
        "1750":"2100",
    ]
    .interpolate(axis=1)
    .squeeze()
    .values
)

oc = (
    emis_df.loc[
        (emis_df["Scenario"] == "ssp245")
        & (emis_df["Region"] == "World")
        & (emis_df["Variable"] == "Emissions|OC"),
        "1750":"2100",
    ]
    .interpolate(axis=1)
    .squeeze()
    .values
)

so2 = (
    emis_df.loc[
        (emis_df["Scenario"] == "ssp245")
        & (emis_df["Region"] == "World")
        & (emis_df["Variable"] == "Emissions|Sulfur"),
        "1750":"2100",
    ]
    .interpolate(axis=1)
    .squeeze()
    .values
)


def aci_log(x, beta, n0, n1, n2):
    aci = beta * np.log(1 + x[0] * n0 + x[1] * n1 + x[2] * n2)
    aci_1850 = beta * np.log(1 + so2[100] * n0 + bc[100] * n1 + oc[100] * n2)
    return aci - aci_1850


param_fits = {}

for model in models:
    ist = int(np.floor(years[model][0] - 1750))
    ien = int(np.ceil(years[model][-1] - 1750))
    param_fits[model], cov = curve_fit(
        aci_log,
        [so2[ist:ien], bc[ist:ien], oc[ist:ien]],
        aci[model],
        bounds=((-np.inf, 0, 0, 0), (0, np.inf, np.inf, np.inf)),
        max_nfev=10000,
    )


def aci_log1750(x, beta, n0, n1, n2):
    aci = beta * np.log(1 + x[0] * n0 + x[1] * n1 + x[2] * n2)
    aci_1750 = beta * np.log(1 + so2[0] * n0 + bc[0] * n1 + oc[0] * n2)
    return aci - aci_1750


if plots:
    colors = {
        "CanESM5": "red",
        "CNRM-CM6-1": "orangered",
        "E3SM-2-0": "darkorange",
        "GFDL-ESM4": "yellowgreen",
        "GFDL-CM4": "yellow",
        "GISS-E2-1-G": "green",
        "HadGEM3-GC31-LL": "turquoise",
        "IPSL-CM6A-LR": "teal",
        "MIROC6": "blue",
        "MPI-ESM-1-2-HAM": "darkslateblue",
        "MRI-ESM2-0": "blueviolet",
        "NorESM2-LM": "purple",
        "UKESM1-0-LL": "crimson",
        "mean": "black",
    }

    fig, ax = pl.subplots(4, 4, figsize=(10, 6.5), squeeze=False)
    for imodel, model in enumerate(sorted(models, key=str.lower)):
        i = imodel // 4
        j = imodel % 4
        ax[i, j].plot(years[model], aci[model], color="k", ls="-", alpha=0.5)
        ax[i, j].plot(
            np.arange(1750.5, 2101),
            aci_log([so2, bc, oc], *param_fits[model]),
            color=colors[model],
            zorder=7,
        )

        ax[i, j].set_xlim(1750, 2100)
        ax[i, j].set_ylim(-1.7, 0.5)
        ax[i, j].axhline(0, lw=0.5, ls=":", color="k")
        ax[i, j].fill_between(
            np.arange(1850, 2015), -10, 10, color="#e0e0e0", zorder=-20
        )
        ax[i, j].get_xticklabels()[-1].set_ha("right")
        if model == "HadGEM3-GC31-LL":
            modlab = "HadGEM3"
        elif model == "MPI-ESM-1-2-HAM":
            modlab = "MPI-ESM1-2"
        else:
            modlab = model
        ax[i, j].text(
            0.03, 0.05, modlab, transform=ax[i, j].transAxes, fontweight="bold"
        )

    ax[0, 0].set_ylabel("W m$^{-2}$")
    ax[1, 0].set_ylabel("W m$^{-2}$")
    ax[2, 0].set_ylabel("W m$^{-2}$")
    ax[3, 0].set_ylabel("W m$^{-2}$")
    ax[3, 1].axis("off")
    ax[3, 2].axis("off")
    ax[3, 3].axis("off")

    pl.suptitle("Aerosol-cloud interactions parameterisations")

    fig.tight_layout()
    os.makedirs(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}", exist_ok=True
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "aci_calibration.png"
    )
    pl.close()

df_params = pd.DataFrame(param_fits, index=["aci_scale", "Sulfur", "BC", "OC"]).T

df_params.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
    "aerosol_cloud.csv"
)

print(df_params)
