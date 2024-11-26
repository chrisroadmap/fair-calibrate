#!/usr/bin/env python
# coding: utf-8

"""Convert concentrations of minor GHGs to equivalent emissions."""

# In a departure from v1.4, we are going to do this for all GHGs except CO2, CH4 and
# N2O since it is difficult to reconcile reported emissions with concentrations for
# other gases (and methane is hard enough).

# For harmonization and infilling purposes it only makes sense to cover the species
# that are explicitly modelled in RCMIP.

# In future I think we want to distil this down into one representative HFC, one
# representative ODS and one representative PFC. I don't think this level of
# disaggregation is justified or important and most IAMs don't record these minor
# GHGs so we're just infilling anyway. Going from 43 GHGs to 6 will make fair
# about three times faster.

# The source of the GHG concentration data is Indicators 2023.

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
f.define_time(1750, 2023, 1)
f.define_scenarios(["historical"])
f.define_configs(["historical"])
species = [
    "HFC-134a",
    "HFC-23",
    "HFC-32",
    "HFC-125",
    "HFC-143a",
    "HFC-152a",
    "HFC-227ea",
    "HFC-236fa",
    "HFC-245fa", 
    "HFC-365mfc",
    "HFC-4310mee",
    "NF3",
    "SF6",
    "SO2F2",
    "CF4",
    "C2F6",
    "C3F8",
    "c-C4F8",
    "CFC-12",
    "CFC-11",
    "CFC-113",
    "CFC-114",
    "CFC-115",
    "HCFC-22",
    "HCFC-141b",
    "HCFC-142b",
    "CH3CCl3",
    "CCl4",
    "CH3Cl",
    "CH3Br",
    "CH2Cl2",
    "CHCl3",
    "Halon-1211",
    "Halon-1301",
    "Halon-2402",
    "C4F10",
    "C5F12",
    "C6F14",
    "C7F16",
    "C8F18",
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
    "../../../../../data/concentrations/ghg_concentrations_1750-2023.csv", index_col=0
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
    if specie == "C6F14":
        f.concentration.loc[
            dict(
                timebounds=slice(1751, 2024),
                specie=specie,
                scenario="historical",
                config="historical",
            )
        ] = 0.5 * (
            df_conc_obs.loc[1750:2022, "n-C6F14"].values
            + df_conc_obs.loc[1751:2023, "n-C6F14"].values
            + df_conc_obs.loc[1750:2022, "i-C6F14"].values
            + df_conc_obs.loc[1751:2023, "i-C6F14"].values
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
            timebounds=slice(1751, 2024),
            specie=specie,
            scenario="historical",
            config="historical",
        )
    ] = 0.5 * (
        df_conc_obs.loc[1750:2022, obs_species[specie]].values
        + df_conc_obs.loc[1751:2023, obs_species[specie]].values
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

output = f.emissions[:, 0, 0, :]

df_out = pd.DataFrame(output, index=np.arange(1750, 2023), columns=species)

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/",
    exist_ok=True,
)
df_out.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "minor_ghg_inverse_1750-2022.csv"
)
