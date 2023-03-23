#!/usr/bin/env python
# coding: utf-8

"""Aerosol-radiation calibration."""
#
# Use the AR6 per-species ERFari calibrations, from Chapter 6 Fig. 6.12. This includes
# contibutions from CH4, N2O and HCs.
#
# The uncertainty is uniform for each specie, a factor of two. This needs to be
# documented in paper.

import os

import numpy as np
import pandas as pd
import pooch
import scipy.stats
from dotenv import load_dotenv

load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

df_emis = pd.read_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/slcf_emissions_1750-2022.csv', index_col=0)
df_conc = pd.read_csv('../../../../../data/concentrations/ghg_concentrations_1750-2022.csv', index_col=0)
for year in range(1751, 1850):
    df_conc.loc[year, :] = np.nan
df_conc.sort_index(inplace=True)
df_conc.interpolate(inplace=True)

emitted_species = [
    "Sulfur",
    "BC",
    "OC",
    "NH3",
    "NOx",
    "VOC",
    "CO",
]

concentration_species = [
    "CH4",
    "N2O",
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
]


hc_species = [
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
]

name_conv = {specie: specie for specie in emitted_species}
name_conv['Sulfur'] = 'SO2'
name_conv['VOC'] = 'NMVOC'

species_out = {}
for ispec, species in enumerate(emitted_species):
    species_out[species] = df_emis[name_conv[species]].values

for ispec, species in enumerate(concentration_species):
    species_out[species] = df_conc[species].values

species_df = pd.DataFrame(species_out, index=range(1750, 2023))

def calculate_eesc(
    concentration,
    fractional_release,
    fractional_release_cfc11,
    cl_atoms,
    br_atoms,
    br_cl_ratio=45,
):
    # EESC is in terms of CFC11-eq
    eesc_out = (
        cl_atoms * (concentration) * fractional_release / fractional_release_cfc11
        + br_cl_ratio
        * br_atoms
        * (concentration)
        * fractional_release
        / fractional_release_cfc11
    ) * fractional_release_cfc11
    return eesc_out


fractional_release = {
    "CFC-11": 0.47,
    "CFC-12": 0.23,
    "CFC-113": 0.29,
    "CFC-114": 0.12,
    "CFC-115": 0.04,
    "HCFC-22": 0.13,
    "HCFC-141b": 0.34,
    "HCFC-142b": 0.17,
    "CCl4": 0.56,
    "CHCl3": 0,
    "CH2Cl2": 0,
    "CH3Cl": 0.44,
    "CH3CCl3": 0.67,
    "CH3Br": 0.6,
    "Halon-1211": 0.62,
    "Halon-1301": 0.28,
    "Halon-2402": 0.65,
}

cl_atoms = {
    "CFC-11": 3,
    "CFC-12": 2,
    "CFC-113": 3,
    "CFC-114": 2,
    "CFC-115": 1,
    "HCFC-22": 1,
    "HCFC-141b": 2,
    "HCFC-142b": 1,
    "CCl4": 4,
    "CHCl3": 3,
    "CH2Cl2": 2,
    "CH3Cl": 1,
    "CH3CCl3": 3,
    "CH3Br": 0,
    "Halon-1211": 1,
    "Halon-1301": 0,
    "Halon-2402": 0,
}

br_atoms = {
    "CFC-11": 0,
    "CFC-12": 0,
    "CFC-113": 0,
    "CFC-114": 0,
    "CFC-115": 0,
    "HCFC-22": 0,
    "HCFC-141b": 0,
    "HCFC-142b": 0,
    "CCl4": 0,
    "CHCl3": 0,
    "CH2Cl2": 0,
    "CH3Cl": 0,
    "CH3CCl3": 0,
    "CH3Br": 1,
    "Halon-1211": 1,
    "Halon-1301": 1,
    "Halon-2402": 2,
}


hc_eesc = {}
total_eesc_2019 = 0
total_eesc_1750 = 0
for species in hc_species:
    hc_eesc[species] = calculate_eesc(
        species_df.loc[2019, species],
        fractional_release[species],
        fractional_release["CFC-11"],
        cl_atoms[species],
        br_atoms[species],
    )
    total_eesc_2019 = total_eesc_2019 + hc_eesc[species]

    hc_eesc[species] = calculate_eesc(
        species_df.loc[1750, species],
        fractional_release[species],
        fractional_release["CFC-11"],
        cl_atoms[species],
        br_atoms[species],
    )
    total_eesc_1750 = total_eesc_1750 + hc_eesc[species]

total_eesc_2019, total_eesc_1750, -0.00808 / (total_eesc_2019 - total_eesc_1750)

# these come from AR6 WG1, yes?
# Yes! from https://github.com/sarambl/AR6_CH6_RCMIPFIGS/blob/master/ar6_ch6_rcmipfigs/data_out/fig6_12_ts15_historic_delta_GSAT/2019_ERF_est.csv
erfari_emitted = pd.Series(
    {
        "Sulfur": -0.234228,
        "BC": 0.144702,
        "OC": -0.072143,
        "NH3": -0.033769,
        "NOx": -0.009166,
        "VOC": -0.002573,
        "CO": 0,
        "CH4": -0.002653,
        "N2O": -0.00209,
        "Equivalent effective stratospheric chlorine": -0.00808,
    }
)

# TODO: probablistic based on Sara's data
# https://github.com/sarambl/AR6_CH6_RCMIPFIGS/blob/master/ar6_ch6_rcmipfigs/data_out/fig6_12_ts15_historic_delta_GSAT/table_std_thornhill_collins_orignames.csv

# erfari radiative efficiency per Mt or ppb or ppt
re = erfari_emitted / (species_df.loc[2019, :] - species_df.loc[1750, :])
re.dropna(inplace=True)

re["Equivalent effective stratospheric chlorine"] = erfari_emitted[
    "Equivalent effective stratospheric chlorine"
] / (total_eesc_2019 - total_eesc_1750)

scalings = scipy.stats.uniform.rvs(
    np.minimum(re * 2, 0),
    np.maximum(re * 2, 0) - np.minimum(re * 2, 0),
    size=(samples, 10),
    random_state=3729329,
)

df = pd.DataFrame(scalings, columns=re.index)

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/",
    exist_ok=True,
)

df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "aerosol_radiation.csv",
    index=False,
)
