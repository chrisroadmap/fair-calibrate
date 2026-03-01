#!/usr/bin/env python
# coding: utf-8

"""Make concentration binary file."""

# 1pctCO2 from RCMIP
# this is from CMIP6 and is also half a year out - does this matter?
# since CMIP7 RCMIP doesn't look like it has been updated, we'll ignore this.

import os

import pandas as pd
import pooch
from dotenv import load_dotenv
from fair import FAIR, __version__
from fair.interface import fill

from fair_calibrate.parameters import PRIOR_SAMPLES

load_dotenv()

samples = PRIOR_SAMPLES
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

print("Making 1pctCO2 concentration binary...")

scenarios = ["1pctCO2"]

species = ["CO2", "CH4", "N2O"]
properties = {
    "CO2": {
        "type": "co2",
        "input_mode": "concentration",
        "greenhouse_gas": True,
        "aerosol_chemistry_from_emissions": False,
        "aerosol_chemistry_from_concentration": False,
    },
    "CH4": {
        "type": "ch4",
        "input_mode": "concentration",
        "greenhouse_gas": True,
        "aerosol_chemistry_from_emissions": False,
        "aerosol_chemistry_from_concentration": False,
    },
    "N2O": {
        "type": "n2o",
        "input_mode": "concentration",
        "greenhouse_gas": True,
        "aerosol_chemistry_from_emissions": False,
        "aerosol_chemistry_from_concentration": False,
    },
}

f = FAIR()
f.define_time(1850, 2060, 1)
f.define_configs(["unspecified"])
f.define_scenarios(scenarios)
f.define_species(species, properties)

f.allocate()

rcmip_concentration_file = pooch.retrieve(
    url=(
        "https://zenodo.org/records/4589756/files/"
        "rcmip-concentrations-annual-means-v5-1-0.csv"
    ),
    known_hash="md5:0d82c3c3cdd4dd632b2bb9449a5c315f",
    progressbar=progress,
    path=datadir,
)

df_conc = pd.read_csv(rcmip_concentration_file)

input = {}
for specie in species:
    input[specie] = (
        df_conc.loc[
            (df_conc["Scenario"] == "1pctCO2")
            & (df_conc["Variable"].str.endswith(specie))
            & (df_conc["Region"] == "World"),
            "1850":"2060",
        ]
        .interpolate(axis=1)
        .values.squeeze()
    )
    fill(f.concentration, input[specie][:, None, None], specie=specie)

os.makedirs(
    "../../output/concentration/",
    exist_ok=True,
)

f.concentration.to_netcdf(
    "../../output/concentration/"
    "1pctCO2_concentration_1850-2060.nc"
)
