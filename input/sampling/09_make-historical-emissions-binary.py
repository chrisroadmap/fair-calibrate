#!/usr/bin/env python
# coding: utf-8

"""Make emissions binary file and fair-readable CSV."""

import datetime
import os
import warnings

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fair import FAIR, __version__
from fair.interface import fill
from fair.io import read_properties

from fair_calibrate.parameters import PRIOR_SAMPLES

load_dotenv()


samples = PRIOR_SAMPLES
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

print("Making SSP emissions binary and CSV...")

scenarios = [
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp434",
    "ssp460",
    "ssp534-over",
    "ssp585",
]

species, properties = read_properties()
species.remove("NOx aviation")
species.remove("Contrails")
species.remove("Halon-1202")

species.append("Irrigation")
properties["Irrigation"] = {
    'type': 'unspecified',  # see issue #179 of FAIR
    'input_mode': 'forcing', 
    'greenhouse_gas': False, 
    'aerosol_chemistry_from_emissions': False, 
    'aerosol_chemistry_from_concentration': False
}

f = FAIR(ch4_method="thornhill2021")
f.define_time(1750, 2500, 1)
f.define_configs(["unspecified"])
f.define_scenarios(scenarios)
f.define_species(species, properties)

f.allocate()

df_in = pd.read_csv(
    "../../output/emissions/"
    "ssps_harmonized_1750-2499.csv",
)
# finally bash 1202
df_in = df_in.drop(df_in[df_in["variable"] == "Halon-1202"].index)

# fill emissions
for scenario in scenarios:
    for specie in species:
        if properties[specie]['input_mode'] == 'emissions':
            fill(
                f.emissions,
                df_in.loc[(df_in["variable"] == specie) & (df_in["scenario"] == scenario), '1750':'2499'].values.squeeze(),
                config="unspecified",
                scenario=scenario,
                specie=specie,
            )

# for CSV, drop "model" and put everything on half years
df_in = df_in.drop(columns=['model'])
half_year_mappings = {f'{year}': f'{year+0.5}' for year in np.arange(1750, 2501)}

df_in = df_in.rename(columns = half_year_mappings)
print(df_in)


f.emissions.to_netcdf(
    "../../output/emissions/"
    "ssps_harmonized_1750-2499.nc"
)

df_in.to_csv(
    "../../output/emissions/"
    "ssps_harmonized_scaled_fair_format_1750-2499.csv",
    index=False
)
