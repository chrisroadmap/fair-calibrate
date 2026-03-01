#!/usr/bin/env python
# coding: utf-8

"""Determining the LAPSI scale factor to use."""

# Correlate cumulative CO2 AFOLU emissions to land use forcing in the present

import os

import pandas as pd

print("Calculating LAPSI scale factor...")

target_forcing = +0.08
base_year = 1750
assessment_year = 2019

df_emis = pd.read_csv(
    "../../data/emissions/"
    "historical_emissions_1750-2023_cmip7.csv"
)
bc = (
    df_emis.loc[df_emis["variable"] == "BC", str(assessment_year)]
    - df_emis.loc[df_emis["variable"] == "BC", str(base_year)]
).values[0]

lapsi_scale_factor = target_forcing / bc

df = pd.DataFrame(
    lapsi_scale_factor,
    columns=["BC"],
    index=["historical_best"],
)
os.makedirs(
    "../../output/calibrations/",
    exist_ok=True,
)
df.to_csv(
    "../../output/calibrations/"
    "lapsi_scale_factor.csv"
)
