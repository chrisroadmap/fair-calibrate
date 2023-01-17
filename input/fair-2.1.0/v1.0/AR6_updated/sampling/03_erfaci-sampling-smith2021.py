#!/usr/bin/env python
# coding: utf-8

"""Sample aerosol indirect."""

# # Using the fair-2.1 pure log formula
#
# **TODO** separate into a calibration and a sampling part
#
# **Note**
# Estimating aerosol cloud interactions from 11 CMIP6 models was performed in Smith
#  et al. 2021: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JD033622
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

print("Sampling aerosol cloud interactions...")


files = glob.glob("../../../../../data/smith2021aerosol/*.csv")

ari = {}
aci = {}
models = []
years = {}
for file in files:
    model = os.path.split(file)[1][:-4]
    if model == "sumlog":
        continue
    models.append(model)
    df = pd.read_csv(file, index_col="year")
    ari[model] = (df["aprp_ERFariSW"] + df["aprp_ERFariLW"]).values.squeeze()
    aci[model] = (df["aprp_ERFaciSW"] + df["aprp_ERFaciLW"]).values.squeeze()
    years[model] = df.index


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
    print(model)
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


df_ar6 = pd.read_csv(
    "../../../../../data/forcing/table_A3.3_historical_ERF_1750-2019_best_estimate.csv"
)

params_ar6, cov = curve_fit(
    aci_log1750,
    [so2[:270], bc[:270], oc[:270]],
    df_ar6["aerosol-cloud_interactions"].values,
    bounds=((-np.inf, 0, 0, 0), (0, np.inf, np.inf, np.inf)),
    max_nfev=10000,
)

if plots:
    colors = {
        "CanESM5": "red",
        "E3SM": "darkorange",
        "GFDL-ESM4": "yellowgreen",
        "GFDL-CM4": "yellow",
        "GISS-E2-1-G": "green",
        "HadGEM3-GC31-LL": "turquoise",
        "IPSL-CM6A-LR": "teal",
        "MIROC6": "blue",
        "MRI-ESM2-0": "blueviolet",
        "NorESM2-LM": "purple",
        "UKESM1-0-LL": "crimson",
        "mean": "black",
        "AR5": "0.6",
        "AR6": "0.3",
        "Lund": "pink",
    }

    endyear = {
        "CanESM5": 2100,
        "MIROC6": 2100,
        "NorESM2-LM": 2100,
        "GISS-E2-1-G": 2100,
        "HadGEM3-GC31-LL": 2099,
        "GFDL-CM4": 2100,
        "E3SM": 2014,
        "UKESM1-0-LL": 2014,
        "GFDL-ESM4": 2014,
        "IPSL-CM6A-LR": 2100,
        "MRI-ESM2-0": 2014,
    }

    startyear = {
        "CanESM5": 1850,
        "MIROC6": 1850,
        "NorESM2-LM": 1850,
        "GISS-E2-1-G": 1850,
        "HadGEM3-GC31-LL": 1850,
        "GFDL-CM4": 1850,
        "E3SM": 1870,
        "GFDL-ESM4": 1850,
        "UKESM1-0-LL": 1850,
        "IPSL-CM6A-LR": 1850,
        "MRI-ESM2-0": 1850,
    }

    fig, ax = pl.subplots(3, 4, figsize=(10, 5), squeeze=False)
    for imodel, model in enumerate(sorted(models)):
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
        if model != "HadGEM3-GC31-LL":
            modlab = model
        else:
            modlab = "HadGEM3"
        ax[i, j].text(
            0.03, 0.05, modlab, transform=ax[i, j].transAxes, fontweight="bold"
        )

    ax[0, 0].set_ylabel("W m$^{-2}$")
    ax[1, 0].set_ylabel("W m$^{-2}$")
    ax[2, 0].set_ylabel("W m$^{-2}$")
    ax[2, 3].axis("off")

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

print("Correlation coefficients between aci parameters")
print(df_params.corr())

beta_samp = df_params["aci_scale"]
n0_samp = df_params["Sulfur"]
n1_samp = df_params["BC"]
n2_samp = df_params["OC"]

kde = scipy.stats.gaussian_kde([n0_samp, n1_samp, n2_samp])
aci_sample = kde.resample(size=samples * 4, seed=63648708)

aci_sample[1, :]

aci_sample[0, aci_sample[0, :] < 0] = np.nan
aci_sample[1, aci_sample[1, :] < 0] = np.nan
aci_sample[2, aci_sample[2, :] < 0] = np.nan

mask = np.any(np.isnan(aci_sample), axis=0)
aci_sample = aci_sample[:, ~mask]

NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)
erfaci_sample = scipy.stats.uniform.rvs(
    size=samples, loc=-2.0, scale=2.0, random_state=71271
)

beta = np.zeros(samples)
erfaci = np.zeros((351, samples))
for i in tqdm(range(samples), desc="aci samples", disable=1 - progress):
    ts2010 = np.mean(
        aci_log(
            [so2[255:265], bc[255:265], oc[255:265]],
            0.92,
            aci_sample[0, i],
            aci_sample[1, i],
            aci_sample[2, i],
        )
    )
    ts1850 = aci_log(
        [so2[100], bc[100], oc[100]],
        0.92,
        aci_sample[0, i],
        aci_sample[1, i],
        aci_sample[2, i],
    )
    ts1750 = aci_log(
        [so2[0], bc[0], oc[0]],
        0.92,
        aci_sample[0, i],
        aci_sample[1, i],
        aci_sample[2, i],
    )
    erfaci[:, i] = (
        (
            aci_log(
                [so2, bc, oc],
                0.92,
                aci_sample[0, i],
                aci_sample[1, i],
                aci_sample[2, i],
            )
            - ts1750
        )
        / (ts2010 - ts1850)
        * (erfaci_sample[i])
    )
    beta[i] = erfaci_sample[i] / (ts2010 - ts1750)


df = pd.DataFrame(
    {
        "shape_so2": aci_sample[0, :samples],
        "shape_bc": aci_sample[1, :samples],
        "shape_oc": aci_sample[2, :samples],
        "beta": beta,
    }
)

df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "aerosol_cloud.csv",
    index=False,
)
