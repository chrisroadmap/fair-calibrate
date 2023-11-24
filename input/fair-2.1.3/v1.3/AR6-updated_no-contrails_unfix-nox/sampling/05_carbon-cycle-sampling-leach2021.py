#!/usr/bin/env python
# coding: utf-8

"""Check and run carbon cycle calibrations."""
#
# Carbon cycle tunings for 11 C4MIP models are from FaIR 2.0 paper (Leach et al. 2021),
# calibrated on CMIP6 1pct runs. Let's see if they give reasonable concentrations in
# emissions-driven mode.

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import scipy.stats
from dotenv import load_dotenv
from fair import __version__
from fair.structure.units import compound_convert

load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
pl.style.use("../../../../../defaults.mplstyle")


print("Making carbon cycle calibrations...")

assert fair_v == __version__


# NB: rU and rA are in GtC units, we need to convert to GtCO2
data = np.array(
    [
        [
            36.73854601035055,
            25.589821019851797,
            40.704695982343765,
            38.09182601398885,
            35.70573492682388,
            34.26732262345922,
            32.223599635483424,
            33.39478016647172,
            33.33937342916488,
            40.735872526405046,
            37.91594456570033,
        ],
        [
            0.0349535801301073,
            0.00597614250950862,
            0.010664893971021883,
            0.0005810081769186404,
            -0.005958784801017192,
            0.021861410870304354,
            0.016608701817126814,
            0.013104461258272693,
            0.031043773610946346,
            0.009471296196005063,
            0.020138272127751655,
        ],
        [
            3.036651884848311,
            5.196160258410032,
            1.2786398011433562,
            2.472206604249436,
            -0.10385375927186047,
            4.855081881723322,
            1.0693983052255476,
            3.4644393974775767,
            1.499323874009292,
            1.5631932779473914,
            2.6714005898495543,
        ],
        [
            -0.0006603263192310749,
            0.004393751681079472,
            0.004211308668836011,
            0.009783189726962682,
            0.018116906645659014,
            -0.004242277713558451,
            0.012336113500092338,
            0.003993779169272571,
            -0.002570300844565665,
            0.004887468785878646,
            0.0018119017134572424,
        ],
    ]
)
data[1, :] = data[1, :] * compound_convert["CO2"]["C"]
data[3, :] = data[3, :] * compound_convert["CO2"]["C"]

models = [
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CESM2",
    "CNRM-ESM2-1",
    "CanESM5",
    "GFDL-ESM4",
    "IPSL-CM6A-LR",
    "MIROC-ES2L",
    "MPI-ESM1-2-LR",
    "NorESM2-LM",
    "UKESM1-0-LL",
]


params = pd.DataFrame(data.T, columns=["r0", "rU", "rT", "rA"], index=models)


kde = scipy.stats.gaussian_kde(params.T)
cc_sample = kde.resample(size=int(samples), seed=2421911)

mask = np.all(np.isnan(cc_sample), axis=0)
cc_sample = cc_sample[:, ~mask]
cc_sample_df = pd.DataFrame(
    data=cc_sample[:, :samples].T, columns=["r0", "rU", "rT", "rA"]
)

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/",
    exist_ok=True,
)

cc_sample_df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "carbon_cycle.csv",
    index=False,
)
