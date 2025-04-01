#!/usr/bin/env python
# coding: utf-8

"""Convert concentrations of N2O to equivalent emissions."""

# The goal is to try and make historical emissions match concentrations. Zeb's dataset
# starts in 1997, or 1970 if you take CEDS and BB4CMIP(6).

# Lifetime defaults are from AR6 and in fair already.

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fair import FAIR, __version__
from fair.interface import fill, initialise

load_dotenv()

print("Calculating historical equivalent emissions...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

assert fair_v == __version__
pl.style.use("../../../../../defaults.mplstyle")


f = FAIR(temperature_prescribed=True)
f.define_time(1750, 2024, 1)
f.define_scenarios(["historical"])
f.define_configs(["historical"])
species = [
    "N2O",
]

# since we only care about back-calculated emissions and not interactions or
# climate effects, treat everything as an F-gas which is inert
properties = {
    specie: {
        "type": "f-gas",
        "input_mode": "concentration",
        "greenhouse_gas": True,
        "aerosol_chemistry_from_emissions": False,
        "aerosol_chemistry_from_concentration": False,
    }
    for specie in species
}

f.define_species(species, properties)
f.allocate()

# Fill concentration time series with observed concentrations
# bear in mind AR6 is mid-year, we shift back six months

# TODO: update with the DOI from Climate Indicator Project
df_conc_obs = pd.read_csv(
    "../../../../../data/concentrations/ghg_concentrations_1750-2024.csv", index_col=0
)
for year in range(1751, 1850):
    df_conc_obs.loc[year, :] = np.nan
df_conc_obs.sort_index(inplace=True)
df_conc_obs.interpolate(inplace=True)

obs_species = {specie: specie for specie in species}

for specie in species:
    f.concentration.loc[
        dict(
            timebounds=slice(1751, 2024),
            specie=specie,
            scenario="historical",
            config="historical",
        )
    ] = 0.5 * (
        df_conc_obs.loc[1750:2023, obs_species[specie]].values
        + df_conc_obs.loc[1751:2024, obs_species[specie]].values
    )
    f.concentration.loc[
        dict(
            timebounds=1750,
            specie=specie,
            scenario="historical",
            config="historical",
        )
    ] = df_conc_obs.loc[1750, obs_species[specie]]


# default AR6 lifetime etc

# calculate steady state pre-industrial lifetime for each GHG
# we do this by setting the initial gas box as c0/m, where c0 is the
# 1750 concentration and m is the conversion from emissions to concentrations
# units.
# We do away with the correction for concentration and emissions.

# these constants are all defined in FaIR. Should probably import them
m = 1 / (5.1352 * f.species_configs["molecular_weight"] / 28.97)
c1 = f.concentration[0, ...]

f.fill_species_configs()
for specie in species:
    fill(f.species_configs["baseline_concentration"], df_conc_obs.loc[1750, obs_species[specie]], specie=specie)
    fill(f.species_configs["baseline_emissions"], 0.900393, specie=specie)  # TODO: don't hard code
    c1 = f.concentration.loc[
        dict(
            specie=specie,
            timebounds=1750,
            scenario="historical",
            config="historical",
        )
    ]
    m = 1 / (
        5.1352 * f.species_configs["molecular_weight"].loc[dict(specie=specie)] / 28.97
    )
    initialise(f.airborne_emissions, c1 / m, specie=specie)
    initialise(f.gas_partitions, np.array([c1 / m, 0, 0, 0]), specie=specie)

# don't calculate warming; we have to initialise it otherwise FaIR will complain about
# NaNs
fill(f.temperature, 0)

f.run(progress=progress)

output = f.emissions[:, 0, 0, :]

df_out = pd.DataFrame(output, index=np.arange(1750, 2024), columns=species)

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/",
    exist_ok=True,
)
df_out.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "n2o_inverse_1750-2023.csv"
)
