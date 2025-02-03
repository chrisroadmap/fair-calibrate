#!/usr/bin/env python
# coding: utf-8

"""Make emissions binary file."""

import datetime
import os
import warnings

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fair import FAIR, __version__
from fair.interface import fill
from fair.io import read_properties

load_dotenv()


cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

assert fair_v == __version__

print("Making SSP emissions binary...")

scenarios = [
    "SSP3 - High Emissions",
    "SSP1 - Low Emissions",
    "SSP1 - Very Low Emissions",
    "SSP2 - Medium Emissions",
    "SSP5 - High Emissions",
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
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "scenario_subset_1750-2100.csv",
#    index_col=[0, 1, 2, 3, 4]
)

# fill emissions
for scenario in scenarios:
    for specie in species:
        if properties[specie]['input_mode'] == 'emissions':
            fill(
                f.emissions,
                df_in.loc[(df_in["variable"] == specie) & (df_in["scenario"] == scenario), '1750':'2100'].values.squeeze(),
                config="unspecified",
                scenario=scenario,
                specie=specie,
            )

f.emissions.to_netcdf(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "scenario_subset_1750-2100.nc"
)
