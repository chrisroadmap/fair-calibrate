#!/usr/bin/env python
# coding: utf-8

"""Sample aerosol indirect."""


import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import pooch
import scipy.stats
from dotenv import load_dotenv
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


def aci_log(x, beta, n0, n1, n2):
    aci = beta * np.log(1 + x[0] * n0 + x[1] * n1 + x[2] * n2)
    aci_1850 = beta * np.log(1 + so2[100] * n0 + bc[100] * n1 + oc[100] * n2)
    return aci - aci_1850


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


df_params = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
    "aerosol_cloud.csv",
    index_col=0,
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

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/",
    exist_ok=True,
)

df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "aerosol_cloud.csv",
    index=False,
)
