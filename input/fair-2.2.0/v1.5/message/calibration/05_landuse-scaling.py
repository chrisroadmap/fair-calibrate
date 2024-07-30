#!/usr/bin/env python
# coding: utf-8

"""Determining the land use scale factor to use."""

# Correlate cumulative CO2 AFOLU emissions to land use forcing in the present

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.interpolate import interp1d

load_dotenv()

print("Calculating land use scale factor...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
datadir = os.getenv("DATADIR")

target_forcing = -0.2
base_year = 1750.5
assessment_year = 2019.5

df_emis_message = pd.read_csv(
    "../../../../../data/emissions/message-baseline-2020.csv",
    index_col=0,
)

years_in = df_emis_message.loc[:,'1750.5':].columns.to_numpy().astype(float)

raw_data = (
    df_emis_message.loc[
        (df_emis_message["Scenario"] == "baseline")
        & (df_emis_message["Variable"] == "CO2 AFOLU")
        & (df_emis_message["Region"] == "World"),
        "1750.5":"2105.5",
    ]
    .values.squeeze()
)
interpolator = interp1d(years_in, raw_data)
co2_afolu = interpolator(np.arange(base_year, np.ceil(assessment_year)))

cumulative_co2_emissions = co2_afolu.sum()  # GtCO2

landuse_scale_factor = target_forcing / cumulative_co2_emissions

df = pd.DataFrame(
    landuse_scale_factor,
    columns=["CO2_AFOLU"],
    index=["historical_best"],
)
os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/",
    exist_ok=True,
)
df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
    "landuse_scale_factor.csv"
)
