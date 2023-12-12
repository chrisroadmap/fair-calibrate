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

target_forcing = +0.08
base_year = 1750
assessment_year = 2019

df_emis = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "all_scaled_1750-2022.csv"
)
bc = (
    df_emis.loc[df_emis["variable"] == "Emissions|BC", str(assessment_year)]
    - df_emis.loc[df_emis["variable"] == "Emissions|BC", str(base_year)]
).values[0]

lapsi_scale_factor = target_forcing / bc

df = pd.DataFrame(
    lapsi_scale_factor,
    columns=["BC"],
    index=["historical_best"],
)
os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/",
    exist_ok=True,
)
df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
    "lapsi_scale_factor.csv"
)
