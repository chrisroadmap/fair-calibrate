#!/usr/bin/env python
# coding: utf-8

"""First constraint: RMSE < 0.17 K"""

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fair import __version__
from tqdm.auto import tqdm

pl.switch_backend("agg")

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
    "temperature_1850-2023.npy"
)

df_gmst = pd.read_csv("../../../../../data/forcing/IGCC_GMST_1850-2024.csv")
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
        np.arange(1850, 2025),
        np.min(temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), axis=1),
        np.max(temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), axis=1),
        color="#000000",
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850, 2025),
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
        np.arange(1850, 2025),
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
        np.arange(1850, 2025),
        np.median(
            temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), axis=1
        ),
        color="#000000",
    )
    ax.plot(np.arange(1850.5, 2025), gmst, color="b")

    ax.set_xlim(1850, 2025)
    ax.set_ylim(-1, 5)
    ax.set_ylabel("°C relative to 1850-1900")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    pl.title("Prior ensemble")
    pl.tight_layout()
    os.makedirs(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/", exist_ok=True
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_historical.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_historical.pdf"
    )
    pl.close()

# temperature is on timebounds, and observations are midyears
# but, this is OK, since we are subtracting a consistent baseline (1850-1900, weighting
# the bounding timebounds as 0.5)
# e.g. 1993.0 timebound has big pinatubo hit, timebound 143
# in obs this is 1992.5, timepoint 142
# compare the timebound after the obs, since the forcing has had chance to affect both
# the obs timepoint and the later timebound.
# the goal of RMSE is as much to match the shape of warming as the magnitude; we do not
# want to average out internal variability in the model or the obs.
for i in tqdm(range(samples), disable=1 - progress):
    rmse_temp[i] = rmse(
        gmst[:174],
        temp_in[1:175, i] - np.average(temp_in[:52, i], weights=weights, axis=0),
    )

accept_temp = rmse_temp < 0.19
print("Passing RMSE constraint:", np.sum(accept_temp))
valid_temp = np.arange(samples, dtype=int)[accept_temp]

# get 10 largest (but passing) and 10 smallest RMSEs
rmse_temp_accept = rmse_temp[accept_temp]
just_passing = np.argpartition(rmse_temp_accept, -10)[-10:]
smashing_it = np.argpartition(rmse_temp_accept, 10)[:10]
print(just_passing)
print(rmse_temp_accept[just_passing])
print(rmse_temp_accept[smashing_it])


if plots:
    # plot top 10 and "just squeaking in 10"
    fig, ax = pl.subplots(figsize=(5, 5))
    ax.plot(
        np.arange(1850.5, 2025),
        (
            temp_in[:, valid_temp[just_passing]]
            - np.average(
                temp_in[:52, valid_temp[just_passing]], weights=weights, axis=0
            )
        ),
        color="#ff0000",
        label=[r"RMSE $\approx$ 0.19°C"] + [""] * 9,
    )
    ax.plot(
        np.arange(1850.5, 2025),
        (
            temp_in[:, valid_temp[smashing_it]]
            - np.average(temp_in[:52, valid_temp[smashing_it]], weights=weights, axis=0)
        ),
        color="#0000ff",
        label=[r"RMSE $\approx$ 0.11°C"] + [""] * 9,
    )
    ax.axhspan(0.67, 0.99, color="k", alpha=0.15, lw=0)
    ax.axvspan(1995, 2015, color="k", alpha=0.15, lw=0)
    ax.plot(np.arange(1850.5, 2025), gmst, color="k", label="Best estimate historical")
    ax.set_xlim(1850, 2025)
    ax.set_ylim(-1, 4)
    ax.set_ylabel("°C relative to 1850-1900")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.text(1860, 0.83, "IPCC AR6 5--95% range", va="center")
    ax.legend(loc="upper left")
    pl.title("Historical GMST")
    pl.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "post_rsme_top10_bottom10_historical.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "post_rsme_top10_bottom10_historical.pdf"
    )
    pl.close()

    # ensemble wide
    fig, ax = pl.subplots(figsize=(5, 5))
    ax.fill_between(
        np.arange(1850, 2025),
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
        np.arange(1850.5, 2025),
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
        np.arange(1850.5, 2025),
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
        np.arange(1850.5, 2025),
        np.median(
            temp_in[:, accept_temp]
            - np.average(temp_in[:52, accept_temp], weights=weights, axis=0),
            axis=1,
        ),
        color="#000000",
    )

    ax.plot(np.arange(1850.5, 2025), gmst, color="b")

    ax.set_xlim(1850, 2025)
    ax.set_ylim(-1, 5)
    ax.set_ylabel("°C relative to 1850-1900")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    pl.title("After RMSE constraint")
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
