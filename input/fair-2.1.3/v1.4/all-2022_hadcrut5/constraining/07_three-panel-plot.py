#!/usr/bin/env python
# coding: utf-8

"""Plot three panel constraints, if plots switched on"""

import os
import sys

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fair import __version__
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# if we're not plotting, don't even start
load_dotenv()
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")

if not plots:
    sys.exit()

pl.switch_backend("agg")

pl.style.use("../../../../../defaults.mplstyle")

print("Making temperature plot...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")

assert fair_v == __version__


step1 = np.loadtxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_pass.csv",
    dtype="int",
)
accept_step1 = np.zeros(samples, dtype=bool)
accept_step1[step1] = True

step2 = np.loadtxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_reweighted_pass.csv",
    dtype="int",
)
accept_step2 = np.zeros(samples, dtype=bool)
accept_step2[step2] = True

temp_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
    "temperature_1850-2101.npy"
)

df_gmst = pd.read_csv("../../../../../data/forcing/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.rebased_1850-1900.csv")
gmst = df_gmst["gmst"].values

weights = np.ones(52)
weights[0] = 0.5
weights[-1] = 0.5

fig, ax = pl.subplots(1, 3, figsize=(18 / 2.54, 6 / 2.54))
ax[0].fill_between(
    np.arange(1850, 2102),
    np.min(temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), axis=1),
    np.max(temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), axis=1),
    color="#000000",
    alpha=0.2,
    lw=0,
)
ax[0].fill_between(
    np.arange(1850, 2102),
    np.percentile(
        temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), 5, axis=1
    ),
    np.percentile(
        temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), 95, axis=1
    ),
    color="#000000",
    alpha=0.2,
    lw=0,
)
ax[0].fill_between(
    np.arange(1850, 2102),
    np.percentile(
        temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), 16, axis=1
    ),
    np.percentile(
        temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), 84, axis=1
    ),
    color="#000000",
    alpha=0.2,
    lw=0,
)
ax[0].plot(
    np.arange(1850, 2102),
    np.median(temp_in - np.average(temp_in[:52, :], weights=weights, axis=0), axis=1),
    color="#000000",
    lw=1,
)
ax[0].plot(np.arange(1850.5, 2024), gmst, color="b", lw=1)

ax[0].set_xlim(1850, 2100)
ax[0].set_ylim(-1, 5)
ax[0].set_ylabel("°C relative to 1850-1900")
ax[0].axhline(0, color="k", ls=":", lw=0.5)
ax[0].set_title("(a) Prior")

ax[1].fill_between(
    np.arange(1850, 2102),
    np.min(
        temp_in[:, accept_step1]
        - np.average(temp_in[:52, accept_step1], weights=weights, axis=0),
        axis=1,
    ),
    np.max(
        temp_in[:, accept_step1]
        - np.average(temp_in[:52, accept_step1], weights=weights, axis=0),
        axis=1,
    ),
    color="#000000",
    alpha=0.2,
    lw=0,
)
ax[1].fill_between(
    np.arange(1850.5, 2102),
    np.percentile(
        temp_in[:, accept_step1]
        - np.average(temp_in[:52, accept_step1], weights=weights, axis=0),
        5,
        axis=1,
    ),
    np.percentile(
        temp_in[:, accept_step1]
        - np.average(temp_in[:52, accept_step1], weights=weights, axis=0),
        95,
        axis=1,
    ),
    color="#000000",
    alpha=0.2,
    lw=0,
)
ax[1].fill_between(
    np.arange(1850.5, 2102),
    np.percentile(
        temp_in[:, accept_step1]
        - np.average(temp_in[:52, accept_step1], weights=weights, axis=0),
        16,
        axis=1,
    ),
    np.percentile(
        temp_in[:, accept_step1]
        - np.average(temp_in[:52, accept_step1], weights=weights, axis=0),
        84,
        axis=1,
    ),
    color="#000000",
    alpha=0.2,
    lw=0,
)
ax[1].plot(
    np.arange(1850.5, 2102),
    np.median(
        temp_in[:, accept_step1]
        - np.average(temp_in[:52, accept_step1], weights=weights, axis=0),
        axis=1,
    ),
    color="#000000",
    lw=1,
)
ax[1].plot(np.arange(1850.5, 2024), gmst, color="b", lw=1)
ax[1].set_xlim(1850, 2100)
ax[1].set_ylim(-1, 5)
# ax[1].set_ylabel("°C relative to 1850-1900")
ax[1].axhline(0, color="k", ls=":", lw=0.5)
ax[1].set_title("(b) After RMSE constraint")

ax[2].fill_between(
    np.arange(1850, 2102),
    np.min(
        temp_in[:, accept_step2]
        - np.average(temp_in[:52, accept_step2], weights=weights, axis=0),
        axis=1,
    ),
    np.max(
        temp_in[:, accept_step2]
        - np.average(temp_in[:52, accept_step2], weights=weights, axis=0),
        axis=1,
    ),
    color="#000000",
    alpha=0.2,
    lw=0,
)
ax[2].fill_between(
    np.arange(1850, 2102),
    np.percentile(
        temp_in[:, accept_step2]
        - np.average(temp_in[:52, accept_step2], weights=weights, axis=0),
        5,
        axis=1,
    ),
    np.percentile(
        temp_in[:, accept_step2]
        - np.average(temp_in[:52, accept_step2], weights=weights, axis=0),
        95,
        axis=1,
    ),
    color="#000000",
    alpha=0.2,
    lw=0,
)
ax[2].fill_between(
    np.arange(1850, 2102),
    np.percentile(
        temp_in[:, accept_step2]
        - np.average(temp_in[:52, accept_step2], weights=weights, axis=0),
        16,
        axis=1,
    ),
    np.percentile(
        temp_in[:, accept_step2]
        - np.average(temp_in[:52, accept_step2], weights=weights, axis=0),
        84,
        axis=1,
    ),
    color="#000000",
    alpha=0.2,
    lw=0,
)
ax[2].plot(
    np.arange(1850, 2102),
    np.median(
        temp_in[:, accept_step2]
        - np.average(temp_in[:52, accept_step2], weights=weights, axis=0),
        axis=1,
    ),
    color="#000000",
    lw=1,
)

ax[2].plot(np.arange(1850.5, 2024), gmst, color="b", lw=1)
# ax.legend(frameon=False, loc="upper left")
ax[2].set_xlim(1850, 2100)
ax[2].set_ylim(-1, 5)
# ax[2].set_ylabel("°C relative to 1850-1900")
ax[2].axhline(0, color="k", ls=":", lw=0.5)
ax[2].set_title("(c) All constraints")

legend_elements = [
    Patch(facecolor="0.8", lw=0, label="min. to max."),
    Patch(facecolor="0.6", lw=0, label="5-95% range"),
    Patch(facecolor="0.4", lw=0, label="16-84% range"),
    Line2D([0], [0], lw=1, color="k", label="Median"),
    Line2D([0], [0], lw=1, color="b", label="Observations"),
]
ax[2].legend(handles=legend_elements, loc="upper left", frameon=False)


pl.tight_layout()
pl.savefig(
    f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
    "constraining_temperature.png"
)
pl.savefig(
    f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
    "constraining_temperature.pdf"
)
pl.close()
