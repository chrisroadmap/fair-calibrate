#!/usr/bin/env python
# coding: utf-8

"""Making the volcanic forcing time series on fair timebounds"""

# Correlate cumulative CO2 AFOLU emissions to land use forcing in the present

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

print("Making volcanic forcing ERF time series...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
datadir = os.getenv("DATADIR")

df_saod = pd.read_csv(
    f"../../../../../data/forcing/aubry_cmip7_saod_draft.csv",
    index_col=0
)

df_saod['erf'] = -20 * (df_saod['Global mean SAOD 550nm'] - np.mean(df_saod.loc[1850:2021, 'Global mean SAOD 550nm']))

# rounding precision
df_saod.loc[np.abs(df_saod['erf']) < 1e-7, 'erf'] = 0

# move from midyear values to fair timebounds
df_saod.index = df_saod.index + 1

# fill in gaps
df_saod.loc[1750, 'erf'] = 0
for year in range(2102, 2502):
    df_saod.loc[year, 'erf'] = df_saod.loc[2101, 'erf']

# delete rows
df_saod.drop(columns = ["Global mean SAOD 550nm",  "Global mean TOA ERF (W/m2)"], inplace=True)

# get everything in right order
df_saod = df_saod.sort_index()

# save out
os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/forcing/",
    exist_ok=True,
)
df_saod.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/forcing/"
    "volcanic_forcing_timebounds.csv"
)
