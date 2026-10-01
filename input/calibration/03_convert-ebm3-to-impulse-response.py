#!/usr/bin/env python
# coding: utf-8

"""Convert energy balance model to impulse response model."""

# This notebook takes the three-layer energy balance model tunings from Donald Cummins
# and converts them to a three-layer impulse response function.
#
# It will then save these into a CSV file.

import os

import numpy as np
import pandas as pd
from fair.energy_balance_model import EnergyBalanceModel
from fair.forcing.ghg import meinshausen2020

from fair_calibrate.layers import get_n_layers

# Number of ocean layers in the calibration to convert: N_LAYERS in parameters.py
n_layers = get_n_layers()

print(f"Converting {n_layers}-layer EBM parameters to IRM parameters...")

df = pd.read_csv(
    f"../../output/calibrations/4xCO2_cummins_ebm{n_layers}_cmip6.csv"
)

models = df["model"].unique()

params = {}
for model in models:
    params[model] = {}
    for run in df.loc[df["model"] == model, "run"]:
        condition = (df["model"] == model) & (df["run"] == run)
        params[model][run] = {}
        params[model][run]["gamma_autocorrelation"] = df.loc[condition, "gamma"].values[
            0
        ]
        params[model][run]["ocean_heat_capacity"] = df.loc[
            condition, "C1":f"C{n_layers}"
        ].values.squeeze()
        params[model][run]["ocean_heat_transfer"] = df.loc[
            condition, "kappa1":f"kappa{n_layers}"
        ].values.squeeze()
        params[model][run]["deep_ocean_efficacy"] = df.loc[condition, "epsilon"].values[
            0
        ]
        params[model][run]["sigma_eta"] = df.loc[condition, "sigma_eta"].values[0]
        params[model][run]["sigma_xi"] = df.loc[condition, "sigma_xi"].values[0]
        params[model][run]["forcing_4co2"] = df.loc[condition, "F_4xCO2"].values[0]


co2 = 284.3169988
ch4 = 808.2490285
n2o = 273.021047

double_co2 = co2 * 2
quadruple_co2 = co2 * 4

rf_4co2 = meinshausen2020(
    np.array([4 * co2, ch4, n2o]).reshape((1, 1, 1, 3)),
    np.array([co2, ch4, n2o]).reshape((1, 1, 1, 3)),
    np.array([1, 1, 1]).reshape((1, 1, 1, 3)),
    np.ones((1, 1, 1, 3)),
    0,
    1,
    2,
    [],
).squeeze()[0]

rf_2co2 = meinshausen2020(
    np.array([2 * co2, ch4, n2o]).reshape((1, 1, 1, 3)),
    np.array([co2, ch4, n2o]).reshape((1, 1, 1, 3)),
    np.array([1, 1, 1]).reshape((1, 1, 1, 3)),
    np.ones((1, 1, 1, 3)),
    0,
    1,
    2,
    [],
).squeeze()[0]

forcing_2co2_4co2_ratio = rf_2co2 / rf_4co2

for model in models:
    for run in df.loc[df["model"] == model, "run"]:
        condition = (df["model"] == model) & (df["run"] == run)
        ebm = EnergyBalanceModel(**params[model][run])
        ebm.emergent_parameters()
        params[model][run] = ebm.__dict__

rows_to_add = []
count = 0
for model in models:
    for run in df.loc[df["model"] == model, "run"]:
        values_to_add = {
            "model": model,
            "run": run,
            "ecs": params[model][run]["ecs"],
            "tcr": params[model][run]["tcr"],
        }
        for i in range(n_layers):
            values_to_add[f"tau{i + 1}"] = params[model][run]["timescales"][i]
        for i in range(n_layers):
            values_to_add[f"q{i + 1}"] = params[model][run]["response_coefficients"][i]
        row_to_add = pd.DataFrame(values_to_add, index=[count])
        rows_to_add.append(row_to_add)
        count = count + 1

# Make LaTeX format table
# don't make decision on run here for pipeline, but do for paper.

multi_runs = {
    "GISS-E2-1-G": "r1i1p1f1",
    "GISS-E2-1-H": "r1i1p3f1",
    "MRI-ESM2-0": "r1i1p1f1",
    "EC-Earth3": "r3i1p1f1",
    "FIO-ESM-2-0": "r1i1p1f1",
    "CanESM5": "r1i1p2f1",
    "FGOALS-f3-L": "r1i1p1f1",
    "CNRM-ESM2-1": "r1i1p1f2",
}

# the table layout is the three-layer one, so it is skipped for two layers
table_models = sorted(list(models)) if n_layers == 3 else []
for model in table_models:
    if model in multi_runs:
        run = multi_runs[model]
    else:
        run = df.loc[df["model"] == model, "run"].values[0]
    print(
        f"{model} & {params[model][run]['ocean_heat_transfer'][0]:.2f} & "
        f"{params[model][run]['ocean_heat_transfer'][1]:.2f} & "
        f"{params[model][run]['ocean_heat_transfer'][2]:.2f} & "
        f"{params[model][run]['ocean_heat_capacity'][0]:.2f} & "
        f"{params[model][run]['ocean_heat_capacity'][1]:.1f} & "
        f"{params[model][run]['ocean_heat_capacity'][2]:.0f} & "
        f"{params[model][run]['deep_ocean_efficacy']:.2f} & "
        f"{params[model][run]['gamma_autocorrelation']:.2f} & "
        f"{params[model][run]['sigma_xi']:.2f} & "
        f"{params[model][run]['sigma_eta']:.2f} & "
        f"{params[model][run]['forcing_4co2']:.2f} & {params[model][run]['ecs']:.2f} & "
        f"{params[model][run]['tcr']:.2f} \\\\"
    )


df_out = pd.concat(rows_to_add)
df_out.sort_values(["model", "run"], inplace=True)

os.makedirs(
    "../../output/calibrations/",
    exist_ok=True,
)

df_out.to_csv(
    "../../output/calibrations/"
    f"4xCO2_impulse_response_ebm{n_layers}_cmip6.csv",
    index=False,
)
