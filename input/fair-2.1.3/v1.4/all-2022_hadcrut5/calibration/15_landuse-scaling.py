#!/usr/bin/env python
# coding: utf-8

"""Determining the land use scale factor to use."""

# Correlate cumulative CO2 AFOLU emissions to land use forcing in the present

import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

print("Calculating land use scale factor...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
datadir = os.getenv("DATADIR")

target_forcing = -0.2
base_year = 1750
assessment_year = 2019

df_emis = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "all_scaled_1750-2022.csv"
)
co2_afolu = df_emis.loc[
    df_emis["variable"] == "Emissions|CO2|AFOLU", str(base_year) : str(assessment_year)
]

cumulative_co2_emissions = co2_afolu.values.sum()  # GtCO2

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
