#!/usr/bin/env python
# coding: utf-8

"""Second constraint, more of a sense check: AF8xCO2 > AF4xCO2 > AF2xCO2"""

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fair import __version__
from tqdm.auto import tqdm

load_dotenv()
print("Doing AF constraint...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")

assert fair_v == __version__

af_in = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
    "airborne_fraction_1pctCO2_y70_y140_y210.npy"
)

valid_temp = np.loadtxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_pass.csv"
).astype(np.int64)


# I require:
# airborne fraction at 4xCO2 > airborne fraction at 2xCO2
# airborne fraction at 8xCO2 > airborne fraction at 4xCO2
# no other nans or weird stuff

af_in[:, af_in[1, :] < af_in[0, :]] = np.nan
af_in[:, af_in[2, :] < af_in[1, :]] = np.nan
mask = np.any(np.isnan(af_in), axis=0)
valid_temp_af = valid_temp[~mask]
af_out = af_in[:, ~mask]
print(np.min(af_out, axis=1))

print("Passing RMSE & AF constraint:", len(valid_temp_af))
os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors",
    exist_ok=True,
)
np.savetxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_af_pass.csv",
    valid_temp_af.astype(int),
    fmt="%d",
)
