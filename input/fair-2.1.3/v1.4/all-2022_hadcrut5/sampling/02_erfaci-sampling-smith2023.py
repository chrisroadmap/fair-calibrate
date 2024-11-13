#!/usr/bin/env python
# coding: utf-8

"""Sample aerosol indirect."""

# **Note also** the uniform prior from -2 to 0. A lot of the sublteties here might also
# want to go into the paper.


import glob
import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import pooch
import scipy.stats
from dotenv import load_dotenv
from scipy.optimize import curve_fit
from tqdm import tqdm

load_dotenv()

pl.style.use("../../../../../defaults.mplstyle")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

print("Sampling aerosol cloud interactions...")


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


# Calibrate on RCMIP
rcmip_emissions_file = pooch.retrieve(
    url="https://zenodo.org/records/4589756/files/"
    "rcmip-emissions-annual-means-v5-1-0.csv",
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
    progressbar=progress,
    path=datadir,
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


def aci_log_nocorrect(x, beta, n0, n1, n2):
    aci = beta * np.log(1 + x[0] * n0 + x[1] * n1 + x[2] * n2)
    return aci


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

    fig, ax = pl.subplots(4, 4, figsize=(18 / 2.54, 12 / 2.54), squeeze=False)
    for imodel, model in enumerate(sorted(models, key=str.lower)):
        i = imodel // 4
        j = imodel % 4
        ax[i, j].plot(years[model], aci[model], color="k", ls="-", alpha=0.5, lw=1)
        ax[i, j].plot(
            np.arange(1750.5, 2101),
            aci_log([so2, bc, oc], *param_fits[model]),
            color=colors[model],
            zorder=7,
            lw=1,
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
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "aci_calibration.pdf"
    )
    pl.close()

df_params = pd.DataFrame(param_fits, index=["aci_scale", "Sulfur", "BC", "OC"]).T

df_params.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
    "aerosol_cloud.csv"
)

print("Correlation coefficients between aci parameters")
print(df_params.corr())

beta_samp = df_params["aci_scale"]
n0_samp = df_params["Sulfur"]
n1_samp = df_params["BC"]
n2_samp = df_params["OC"]

kde = scipy.stats.gaussian_kde(
    [np.log(n0_samp), np.log(n1_samp), np.log(n2_samp)], bw_method=0.1
)
aci_sample = kde.resample(size=samples * 1, seed=63648708)

# aci_sample[0, aci_sample[0, :] < 0] = np.nan
# aci_sample[1, aci_sample[1, :] < 0] = np.nan
# aci_sample[2, aci_sample[2, :] < 0] = np.nan
# mask = np.any(np.isnan(aci_sample), axis=0)
# aci_sample = aci_sample[:, ~mask]

# trapezoid distribution [-2.2, -1.7, -1.0, -0.3, +0.2]
erfaci_sample = scipy.stats.trapezoid.rvs(
    0.25, 0.75, size=samples, loc=-2.2, scale=2.4, random_state=71271
)

# Sampling with updated emissions.
df_emis_obs = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "slcf_emissions_1750-2022.csv",
    index_col=0,
)

# overwrite RCMIP
so2 = df_emis_obs["SO2"].values
bc = df_emis_obs["BC"].values
oc = df_emis_obs["OC"].values

beta = np.zeros(samples)
# erfaci = np.zeros((273, samples))
for i in tqdm(range(samples), desc="aci samples", disable=1 - progress):
    ts2010 = np.mean(
        aci_log_nocorrect(
            [so2[255:265], bc[255:265], oc[255:265]],
            1,
            np.exp(aci_sample[0, i]),
            np.exp(aci_sample[1, i]),
            np.exp(aci_sample[2, i]),
        )
    )
    ts1750 = aci_log_nocorrect(
        [so2[0], bc[0], oc[0]],
        1,
        np.exp(aci_sample[0, i]),
        np.exp(aci_sample[1, i]),
        np.exp(aci_sample[2, i]),
    )
    beta[i] = erfaci_sample[i] / (ts2010 - ts1750)

df = pd.DataFrame(
    {
        "shape_so2": np.exp(aci_sample[0, :samples]),
        "shape_bc": np.exp(aci_sample[1, :samples]),
        "shape_oc": np.exp(aci_sample[2, :samples]),
        "beta": beta,
    }
)

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/",
    exist_ok=True,
)

df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "aerosol_cloud.csv",
    index=False,
)
