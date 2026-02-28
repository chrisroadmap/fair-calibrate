#!/usr/bin/env python
# coding: utf-8

"""Combine Hege-Beate's data into one file."""

import glob
import os
from pathlib import PurePath

import pandas as pd
from dotenv import load_dotenv
from fair import __version__

load_dotenv()

print("Making nice 4xCO2 data...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")

assert fair_v == __version__

available_files = glob.glob(
    "../../../../../data/cmip6-hbf/cmip_data/*/abrupt-4xCO2/"
    "*_abrupt-4xCO2_*_anomalies.txt"
)

maxlen = 0

models = []
runs = []
lines = []
for file in available_files:
    model = PurePath(file).parts[8]
    run = PurePath(file).parts[10].split("_")[2]
    models.append(model)
    runs.append(run)
    df = pd.read_csv(file, index_col=0)
    vars = {}
    for var in ["tas", "rlut", "rsut", "rsdt"]:
        vars[var] = df[var].values.squeeze()
        if len(vars[var]) > maxlen:
            maxlen = len(vars[var])
        line = [
            "CMIP",
            model,
            run,
            "CMIP6",
            "unspecified",
            "World",
            "abrupt-4xCO2",
            "W m^-2",
            var,
        ]
        line.extend(vars[var][:150])
        lines.append(line)
    vars["rndt"] = vars["rsdt"] - vars["rsut"] - vars["rlut"]
    line = [
        "CMIP",
        model,
        run,
        "CMIP6",
        "unspecified",
        "World",
        "abrupt-4xCO2",
        "W m^-2",
        "rndt",
    ]
    line.extend(vars["rndt"][:150])
    lines.append(line)

df = pd.DataFrame(
    lines,
    columns=(
        [
            "activity_id",
            "climate_model",
            "member_id",
            "mip_era",
            "model",
            "region",
            "scenario",
            "unit",
            "variable",
        ]
        + ["X%d" % year for year in range(1850, 2000)]
    ),
)

to_remove = [f"X{year}" for year in range(1850, 2000)]
df.dropna(subset=to_remove, inplace=True)

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/",
    exist_ok=True,
)

df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/"
    "calibrations/4xCO2_cmip6.csv",
    index=False,
)
