#!/usr/bin/env python
# coding: utf-8

"""Convert concentrations of minor GHGs to equivalent emissions."""

# This is needed for Montreal gases, and SO2F2.

# In this calibration, we'll just take the species that appear in RCMIP otherwise
# there's nothing to harmonize to.

# Lifetime defaults are from RCMIP and in FaIR already.

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
f.define_time(1750, 2022, 1)
f.define_scenarios(["historical"])
f.define_configs(["historical"])
species = [
    "CFC-11",
    "CFC-12",
    "CFC-113",
    "CFC-114",
    "CFC-115",
    "HCFC-22",
    "HCFC-141b",
    "HCFC-142b",
    "CCl4",
    "CHCl3",
    "CH2Cl2",
    "CH3Cl",
    "CH3CCl3",
    "CH3Br",
    "Halon-1211",
    "Halon-1301",
    "Halon-2402",
    "SO2F2",
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
    "../../../../../data/concentrations/ghg_concentrations_1750-2022.csv", index_col=0
)
for year in range(1751, 1850):
    df_conc_obs.loc[year, :] = np.nan
df_conc_obs.sort_index(inplace=True)
df_conc_obs.interpolate(inplace=True)

# it's unclear whether the isomer of C6F14 should be included
# Comparing to Meinshausen et al. 2017, I conclude it should be.
# treat as a special case below
obs_species = {specie: specie for specie in species}
obs_species["C4F10"] = "n-C4F10"
obs_species["C5F12"] = "n-C5F12"
obs_species["HFC-4310mee"] = "HFC-43-10mee"

for specie in species:
    if specie == "Halon-1202":
        f.concentration.loc[
            dict(
                timebounds=slice(1750, 2023),
                specie=specie,
                scenario="historical",
                config="historical",
            )
        ] = 0
        continue
    elif specie == "C6F14":
        f.concentration.loc[
            dict(
                timebounds=slice(1751, 2023),
                specie=specie,
                scenario="historical",
                config="historical",
            )
        ] = 0.5 * (
            df_conc_obs.loc[1750:2021, "n-C6F14"].values
            + df_conc_obs.loc[1751:2022, "n-C6F14"].values
            + df_conc_obs.loc[1750:2021, "i-C6F14"].values
            + df_conc_obs.loc[1751:2022, "i-C6F14"].values
        )
        f.concentration.loc[
            dict(
                timebounds=1750,
                specie=specie,
                scenario="historical",
                config="historical",
            )
        ] = (
            df_conc_obs.loc[1750, "n-C6F14"] + df_conc_obs.loc[1750, "i-C6F14"]
        )
        continue

    f.concentration.loc[
        dict(
            timebounds=slice(1751, 2023),
            specie=specie,
            scenario="historical",
            config="historical",
        )
    ] = 0.5 * (
        df_conc_obs.loc[1750:2021, obs_species[specie]].values
        + df_conc_obs.loc[1751:2022, obs_species[specie]].values
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
    fill(f.species_configs["baseline_concentration"], 0, specie=specie)
    fill(f.species_configs["baseline_emissions"], 0, specie=specie)
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

# on the basis of no better information, set 2022 equal to 2021
output = np.ones((273, len(species))) * np.nan
output[:272, :] = f.emissions[:, 0, 0, :]
output[272] = f.emissions[-1, 0, 0, :]

df_out = pd.DataFrame(output, index=np.arange(1750, 2023), columns=species)

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/",
    exist_ok=True,
)
df_out.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "minor_ghg_inverse_1750-2022.csv"
)
