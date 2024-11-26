#!/usr/bin/env python
# coding: utf-8

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from dotenv import load_dotenv

print("Making SLCF emissions...")

# Grab CEDS to 2022
# include methane which we will mash with PRIMAP later

load_dotenv()

pl.style.use("../../../../../defaults.mplstyle")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

species = ["CH4", "BC", "OC", "SO2", "NOx", "CO", "NMVOC", "NH3"]

rcmip_specie = {specie: specie for specie in species}
rcmip_specie["NMVOC"] = "VOC"

start_dates = {specie: 1750 for specie in species if specie != "CH4"}
start_dates["CH4"] = 1970

ceds_df = pd.DataFrame(columns=species, index=np.arange(1750, 2023, dtype=int))

for specie in species:
    ceds_df.loc[start_dates[specie]:, specie] = (
        0.001
        * pd.read_csv(
            "../../../../../data/emissions/ceds/v20240708/"
            f"{specie}_CEDS_global_emissions_by_fuel_v2024_07_08.csv"
        )
        .sum()[f"X{start_dates[specie]}":]
        .values
    )

ceds_df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "ceds_emissions_1750-2022.csv"
)
