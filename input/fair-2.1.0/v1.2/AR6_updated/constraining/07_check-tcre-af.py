#!/usr/bin/env python
# coding: utf-8

"""Spit out the TCRE and airborne fraction."""

# we don't constrain on these as they are model-based assessments, but we want to add
# to the table.

import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")

af = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/"
    "prior_runs/airborne_fraction_1pctCO2_y70_y140.npy"
)
temp = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/"
    "prior_runs/temperature_1pctCO2_y70_y140.npy"
)

pass1 = np.loadtxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/"
    "posteriors/runids_rmse_pass.csv",
    dtype=int,
)
pass2 = np.loadtxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/"
    "posteriors/runids_rmse_reweighted_pass.csv",
    dtype=int,
)

idx = np.in1d(pass1, pass2).nonzero()[0]
print("AF 2xCO2*:", np.percentile(af[0, idx], (16, 50, 84)))
print("AF 4xCO2*:", np.percentile(af[1, idx], (16, 50, 84)))
print("TCRE*:", np.percentile(af[0, idx] * temp[0, idx] / 0.593, (16, 50, 84)))
