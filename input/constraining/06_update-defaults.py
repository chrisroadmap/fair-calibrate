#!/usr/bin/env python
# coding: utf-8

"""Updates defaults"""

# thanks to https://github.com/fracamil/fair-species-configs/blob/main/input/fair-2.1.3/v1.4/all-2022/calibration/17_update-species-configs-properties.py

import os

import numpy as np
import pandas as pd
from fair import __version__
import fair.defaults.data.ar6

from fair_calibrate.parameters import PRIOR_SAMPLES, POSTERIOR_SAMPLES

print("Updating defaults...")

samples = PRIOR_SAMPLES
output_ensemble_size = POSTERIOR_SAMPLES


ch4_lifetime_species = [
    "CH4",
    "N2O",
    "VOC",
    "NOx",
    "Equivalent effective stratospheric chlorine",
]
n_gasboxes = 4
update_landuse = True
update_lapsi = True

df_species_configs = pd.read_csv('../../data/fair_parameters/species_configs_properties_landuse_forcing_irrigation.csv', index_col=0)


# read the methane lifetime calibration file and rename the HC column for easing
# later reinitialization of values
df_methane = pd.read_csv(
    "../../output/calibrations/"
    "CH4_lifetime.csv",
    index_col=0,
)
df_methane = df_methane.rename(
    {"HC": "Equivalent effective stratospheric chlorine"}, axis="columns"
)

# read the landuse emissions scaling if required
if update_landuse == True:
    df_landuse = pd.read_csv(
        "../../output/calibrations/"
        "landuse_scale_factor.csv",
        index_col=0,
    )

# read the lapsi calibration file if required
if update_lapsi == True:
    df_lapsi = pd.read_csv(
        "../../output/calibrations/"
        "lapsi_scale_factor.csv",
        index_col=0,
    )

# CH4: modification of "ch4_lifetime_chemical_sensitivity" according to calibrated values
for specie in ch4_lifetime_species:
    df_species_configs.loc[
        specie, "ch4_lifetime_chemical_sensitivity"
    ] = df_methane.loc["historical_best", specie]

# CH4: modification of "unperturbed_lifetime" according to calibrated value
for gasbox in range(n_gasboxes):
    df_species_configs.loc["CH4", f"unperturbed_lifetime{gasbox}"] = df_methane.loc[
        "historical_best", "base"
    ]

# CH4: modification of "lifetime_temperature_sensitivity" according to calibrated value
df_species_configs.loc["CH4", "lifetime_temperature_sensitivity"] = df_methane.loc[
    "historical_best", "temp"
]

# CO2 AFOLU: modification of "land_use_cumulative_emissions_to_forcing" according to
# calibrated values, if required
if update_landuse == True:
    df_species_configs.loc[
        "CO2 AFOLU", "land_use_cumulative_emissions_to_forcing"
    ] = df_landuse.loc["historical_best", "CO2_AFOLU"]

# BC: modification of "lapsi_radiative_efficiency" according to
# calibrated values, if required
if update_lapsi == True:
    df_species_configs.loc["BC", "lapsi_radiative_efficiency"] = df_lapsi.loc[
        "historical_best", "BC"
    ]

## Volcanic: tune down volcanic efficiency
#df_species_configs.loc["Volcanic", "forcing_efficacy"] = 0.6

df_species_configs.drop(index=["NOx aviation"], inplace=True)
df_species_configs.drop(index=["Contrails"], inplace=True)
df_species_configs.drop(index=["Halon-1202"], inplace=True)


# read the default baseline_emissions file: the baseline emissions for the specified species
# will be updated in the config_species_properties
df_emissions = pd.read_csv(
    "../../output/emissions/"
    "ssps_harmonized_1750-2499.csv"
)

for specie in df_emissions.variable:
    df_species_configs.loc[specie, 'baseline_emissions'] = df_emissions.loc[df_emissions["variable"] == specie, "1750"].values[0]


# it came back, delete it again
df_species_configs.drop(index=["Halon-1202"], inplace=True)

df_species_configs.to_csv(
    "../../output/posteriors/"
    "species_configs_properties.csv",
    na_rep=np.nan
)
