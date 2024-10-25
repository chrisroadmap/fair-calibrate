#!/usr/bin/env python
# coding: utf-8

"""Make emissions binary file."""
#
# SSPs from RCMIP

import os

import pandas as pd
import pooch
from dotenv import load_dotenv
from fair import FAIR, __version__
from fair.io import read_properties
from fair.interface import fill

load_dotenv()


cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

assert fair_v == __version__

print("Making emissions binary...")

scenarios = [
    "MESSAGE-GLOBIOM___SSP2-45",
    "MESSAGE-GLOBIOM___SSP2-45-MFR-CH4",
    "MESSAGE-GLOBIOM___SSP2-45-CLE-CH4",
]

species, properties = read_properties()
species.remove("NOx aviation")
species.remove("Contrails")

f = FAIR(ch4_method="thornhill2021")
f.define_time(1750, 2101, 1)
f.define_configs(["unspecified"])
f.define_scenarios(scenarios)
f.define_species(species, properties)

f.allocate()


df_in = pd.read_csv(
    "../../../../../data/emissions/CH4Pledge-rcmip-emissions-ssp245-CLE-MFR-filled.csv"
)
variables = list(df_in["variable"])
units = list(df_in["unit"])
var_units = {var: unit for var, unit in zip(variables, units)}

# fill emissions
for scenario in scenarios:
    for specie in variables:
        data = df_in.loc[
            (df_in["scenario"] == scenario)
            & (df_in["variable"] == specie),
            "1750.5":,
        ].values.squeeze()
        fill(
            f.emissions,
            data,
            config="unspecified",
            scenario=scenario,
            #        timepoints=np.arange(1750.5, 2021),
            specie=specie,
        )


os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/",
    exist_ok=True,
)

f.emissions.to_netcdf(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "emissions_1750-2100.nc"
)
