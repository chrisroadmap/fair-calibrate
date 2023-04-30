#!/usr/bin/env python
# coding: utf-8

"""First constraint: RMSE < 0.16 K"""

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fair import __version__
from tqdm.auto import tqdm

load_dotenv()
pl.style.use("../../../../../defaults.mplstyle")

print("Doing RMSE constraint...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")

assert fair_v == __version__


temp_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
    "temperature_1850-2101.npy"
)

df_gmst = pd.read_csv("../../../../../data/forcing/AR6_GMST.csv")
gmst = df_gmst["gmst"].values


def rmse(obs, mod):
    return np.sqrt(np.sum((obs - mod) ** 2) / len(obs))


weights = np.ones(52)
weights[0] = 0.5
weights[-1] = 0.5

rmse_temp = np.zeros((samples))

if plots:
    fig, ax = pl.subplots(figsize=(5, 5))
    ax.fill_between(
        np.arange(1850, 2102),
        np.min(temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), axis=1),
        np.max(temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), axis=1),
        color="#000000",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850, 2102),
        np.percentile(
            temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), 5, axis=1
        ),
        np.percentile(
            temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), 95, axis=1
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850, 2102),
        np.percentile(
            temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), 16, axis=1
        ),
        np.percentile(
            temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), 84, axis=1
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.plot(
        np.arange(1850, 2102),
        np.median(
            temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), axis=1
        ),
        color="#000000",
    )
    ax.plot(np.arange(1850.5, 2023), gmst, color="b")

    ax.set_xlim(1850, 2100)
    ax.set_ylim(-1, 5)
    ax.set_ylabel("°C relative to 1850-1900")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    pl.title("Temperature anomaly: historical prior")
    pl.tight_layout()
    os.makedirs(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/", exist_ok=True
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_historical_gmst.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_historical_gmst.pdf"
    )
    pl.close()

for i in tqdm(range(samples), disable=1 - progress):
    rmse_temp[i] = rmse(
        gmst[:171],
        temp_in[:171, i] - np.average(temp_in[:52, i], weights=weights, axis=0),
    )

accept_temp = rmse_temp < 0.16
print("Passing RMSE constraint:", np.sum(accept_temp))
valid_temp = np.arange(samples, dtype=int)[accept_temp]

if plots:
    fig, ax = pl.subplots(figsize=(5, 5))
    ax.fill_between(
        np.arange(1850, 2102),
        np.min(
            temp_in[:, accept_temp]
            - np.average(temp_in[:52, accept_temp], weights=weights, axis=0),
            axis=1,
        ),
        np.max(
            temp_in[:, accept_temp]
            - np.average(temp_in[:52, accept_temp], weights=weights, axis=0),
            axis=1,
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850.5, 2102),
        np.percentile(
            temp_in[:, accept_temp]
            - np.average(temp_in[:52, accept_temp], weights=weights, axis=0),
            5,
            axis=1,
        ),
        np.percentile(
            temp_in[:, accept_temp]
            - np.average(temp_in[:52, accept_temp], weights=weights, axis=0),
            95,
            axis=1,
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850.5, 2102),
        np.percentile(
            temp_in[:, accept_temp]
            - np.average(temp_in[:52, accept_temp], weights=weights, axis=0),
            16,
            axis=1,
        ),
        np.percentile(
            temp_in[:, accept_temp]
            - np.average(temp_in[:52, accept_temp], weights=weights, axis=0),
            84,
            axis=1,
        ),
        color="#000000",
        alpha=0.2,
    )
    ax.plot(
        np.arange(1850.5, 2102),
        np.median(
            temp_in[:, accept_temp]
            - np.average(temp_in[:52, accept_temp], weights=weights, axis=0),
            axis=1,
        ),
        color="#000000",
    )

    ax.plot(np.arange(1850.5, 2023), gmst, color="b")

    ax.set_xlim(1850, 2100)
    ax.set_ylim(-1, 5)
    ax.set_ylabel("°C relative to 1850-1900")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    pl.title("Temperature anomaly: historical RMSE constraint")
    pl.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "post_rsme_historical.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "post_rsme_historical.pdf"
    )
    pl.close()

valid_temp = np.arange(samples, dtype=int)[accept_temp]
os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors",
    exist_ok=True,
)
np.savetxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_pass.csv",
    valid_temp.astype(int),
    fmt="%d",
)
