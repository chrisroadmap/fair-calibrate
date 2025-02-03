#!/usr/bin/env python
# coding: utf-8

"""Stitch historical and future emissions in a couple of pathways."""

# These harmonized files are for checking that the reconstructed historic to future
# emissions are roughly right when projected forward.

# 2021 is the last year where all emissions are available so we harmonize to this.
# from Zeb, we use 2021 from the scenario, NOT the historical.

import os

from fair import __version__
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

assert fair_v == __version__

print("Making emissions binary...")

models_scenarios = {
    "AIM 3.0" : "SSP3 - High Emissions",
    "IMAGE 3.4" : "SSP1 - Low Emissions",
    "IMAGE 3.4" : "SSP1 - Very Low Emissions",
    "MESSAGEix-GLOBIOM 2.1-M-R12" : "SSP2 - Medium Emissions",
    "WITCH 6.0" : "SSP5 - High Emissions",
}

df_future = pd.read_csv(
    "../../../../../data/emissions/scenario_subset_2021-2100.csv",
    index_col = [0, 1, 2, 3, 4]
)

df_future.interpolate(axis=1, inplace=True)


df_history = pd.read_csv(
    f"../../../../../data/emissions/historical_emissions_1750-2021.csv",
    index_col = [0, 1, 2, 3, 4]
)

# de-daft units and get 4310 harmonized
for specie in ['N2O', 'CO2|Energy and Industrial Processes', 'CO2|AFOLU']:
    df_future.iloc[df_future.index.get_level_values("variable") == f"Emissions|{specie}"] = (
        df_future.iloc[df_future.index.get_level_values("variable") == f"Emissions|{specie}"]
        * 0.001
    )

df_future = df_future.rename(index={'kt N2O/yr': 'Mt N2O/yr', 'Mt CO2/yr': 'Gt CO2/yr', 'kt HFC4310/yr': 'kt HFC4310mee/yr'})
df_history = df_history.rename(index={'kt HFC43-10/yr': 'kt HFC4310mee/yr'})

# want 2021 from scenario, so expunge from history (though, should correspond)
df_history.drop(columns=['2021'], inplace=True)

# also avoid naming clashes so drop superfluous info from history
df_history = df_history.droplevel(('model', 'scenario'), axis=0)

df_merged = df_history.join(df_future).reorder_levels(("model", "scenario", "region", "variable", "unit")).sort_values(["model", "variable"])

# drop "emissions" in variable name
variables_full = df_merged.index.unique(level='variable')
variable_mapping = {}
for variable in variables_full:
    variable_short = " ".join(variable.split("|")[1:])
    variable_mapping[variable] = variable_short
variable_mapping["Emissions|CO2|Energy and Industrial Processes"] = "CO2 FFI"
df_merged = df_merged.rename(index=variable_mapping)
print(df_merged)

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/",
    exist_ok=True,
)

df_merged.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "scenario_subset_1750-2100.csv",
)
