#!/usr/bin/env python
# coding: utf-8

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from dotenv import load_dotenv

print("Making SLCF emissions...")

# Create a consolidated time series file from the CEDS emissions, then extend them to
# 2022 using the COVID-MIP assumptions (same process as Forster et al. 2023)

load_dotenv(override=True)

pl.style.use("../../../../../defaults.mplstyle")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

species = ["BC", "OC", "SO2", "NOx", "CO", "NMVOC", "NH3"]

rcmip_specie = {specie: specie for specie in species}
rcmip_specie["NMVOC"] = "VOC"

ceds_df = pd.DataFrame(columns=species, index=np.arange(1750, 2024, dtype=int))
covid_df = pd.read_csv(
    "../../../../../data/emissions/two_year_blip_emissions_ratios.csv", index_col=0
)

for specie in species:
    ceds_df.loc[:2022, specie] = (
        0.001
        * pd.read_csv(
            "../../../../../data/emissions/ceds/v20240401/"
            f"{specie}_CEDS_global_emissions_by_sector_v2024_04_01.csv"
        )
        .sum()["X1750":]
        .values
    )
    ceds_df.loc[2023, specie] = (
        ceds_df.loc[2019, specie] * covid_df.loc[2023, rcmip_specie[specie]]
    )

ceds_df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "ceds_emissions_extended_1750-2023.csv"
)
