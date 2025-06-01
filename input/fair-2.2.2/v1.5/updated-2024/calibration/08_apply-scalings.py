#!/usr/bin/env python
# coding: utf-8

"""Apply scaling factors to N2O only."""

import os

import numpy as np
import pandas as pd
import pooch
import xarray as xr
from dotenv import load_dotenv
from fair import __version__
from fair.fair import DEFAULT_SPECIES_CONFIG_FILE
from fair.structure.units import desired_concentration_units

load_dotenv()

print("Apply scaling factors to scenarios...")

ar6_colors = {
    "ssp119": "#00a9cf",
    "ssp126": "#003466",
    "ssp245": "#f69320",
    "ssp370": "#df0000",
    "ssp434": "#2274ae",
    "ssp460": "#b0724e",
    "ssp534-over": "#92397a",
    "ssp585": "#980002",
}

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

assert fair_v == __version__

df_emis = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "ssps_harmonized_1750-2499.csv",
#    index_col=[0, 1, 2, 3, 4]
)

scalings_df = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
    "emissions_scalings.csv",
    index_col=0,
)

for specie in scalings_df.columns:
    sf = scalings_df.loc["historical_best", specie]

    df_emis.loc[(df_emis["variable"] == specie), "1750":] = df_emis.loc[(df_emis["variable"] == specie), "1750":] * sf

df_emis.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "ssps_harmonized_scaled_1750-2499.csv",
    index=False
)
