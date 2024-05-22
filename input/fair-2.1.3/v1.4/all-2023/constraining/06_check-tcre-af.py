#!/usr/bin/env python
# coding: utf-8

"""Spit out the TCRE and airborne fraction."""

# we don't constrain on these as they are model-based assessments, but we want to add
# to the table.

import os

import numpy as np
from dotenv import load_dotenv
from fair.earth_params import mass_atmosphere, molecular_weight_air

load_dotenv(override=True)

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
temp1000 = np.load(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/"
    "prior_runs/temperature_1pctCO2_1000GtC.npy"
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

co2_1850 = 284.3169988
co2_1920 = co2_1850 * 1.01**70  # NOT 2x (69.66 yr), per definition of TCRE
mass_factor = 12.011 / molecular_weight_air * mass_atmosphere / 1e21
# mass_factor converts ppm CO2 to (1000 Gt C)

idx = np.in1d(pass1, pass2).nonzero()[0]
print("temperature 2xCO2:", np.percentile(temp[0, idx], (5, 50, 95)))
print("temperature 4xCO2:", np.percentile(temp[1, idx], (5, 50, 95)))
print("TCRE @1000GtC:", np.percentile(temp1000[idx], (5, 50, 95)))
print("AF 2xCO2*:", np.percentile(af[0, idx], (16, 50, 84)))
print("AF 4xCO2*:", np.percentile(af[1, idx], (16, 50, 84)))
print(
    "TCRE (IPCC method)*:",
    np.percentile(
        af[0, idx] * temp[0, idx] / ((co2_1920 - co2_1850) * mass_factor), (16, 50, 84)
    ),
)
print("*likely range")
