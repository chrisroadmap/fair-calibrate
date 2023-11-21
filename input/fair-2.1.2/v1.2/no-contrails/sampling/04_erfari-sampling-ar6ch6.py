#!/usr/bin/env python
# coding: utf-8

"""Aerosol-radiation calibration."""
# In AR6, ERFari was based on emissions to forcing coefficients from Myhre et al (2013)
# https://acp.copernicus.org/articles/13/1853/2013/. At the time, I deemed there not
# sufficient evidence from CMIP6 AerChemMIP models or any other sources to update these.
# The uncertainty ranges from each precursor were expanded slightly compared to Myhre et
# al., in order to reproduce the overall ERFari uncertainty assessment (assumed that
# uncertainties in individual components are uncorrelated).

# Following AR6 and a re-calibration of FaIR, I now use Bill Collins/Terje Bertnsen/
# Sara Blichner/Sophie Szopa's chapter 6 correspondences of emissions or concentrations
# to forcing.

# Rescale to the assessed forcings of -0.3 W/m2 for ERFari 2005-14

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

rcmip_emissions_file = pooch.retrieve(
    url="doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
)

rcmip_concentration_file = pooch.retrieve(
    url=("doi:10.5281/zenodo.4589756/" "rcmip-concentrations-annual-means-v5-1-0.csv"),
    known_hash="md5:0d82c3c3cdd4dd632b2bb9449a5c315f",
)

df_emis = pd.read_csv(rcmip_emissions_file)
df_conc = pd.read_csv(rcmip_concentration_file)

emitted_species = [
    "Sulfur",
    "BC",
    "OC",
    "NH3",
    "NOx",
    "VOC",
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

species_out = {}
for ispec, species in enumerate(emitted_species):
    species_rcmip_name = species.replace("-", "")
    emis_in = (
        df_emis.loc[
            (df_emis["Scenario"] == "ssp245")
            & (df_emis["Variable"].str.endswith("|" + species_rcmip_name))
            & (df_emis["Region"] == "World"),
            "1750":"2100",
        ]
        .interpolate(axis=1)
        .values.squeeze()
    )
    species_out[species] = emis_in

# Adjust NOx for units error in BB
gfed_sectors = [
    "Emissions|NOx|MAGICC AFOLU|Agricultural Waste Burning",
    "Emissions|NOx|MAGICC AFOLU|Forest Burning",
    "Emissions|NOx|MAGICC AFOLU|Grassland Burning",
    "Emissions|NOx|MAGICC AFOLU|Peat Burning",
]

species_out["NOx"] = (
    df_emis.loc[
        (df_emis["Scenario"] == "ssp245")
        & (df_emis["Region"] == "World")
        & (df_emis["Variable"].isin(gfed_sectors)),
        "1750":"2100",
    ]
    .interpolate(axis=1)
    .values.squeeze()
    .sum(axis=0)
    * 46.006
    / 30.006
    + df_emis.loc[
        (df_emis["Scenario"] == "ssp245")
        & (df_emis["Region"] == "World")
        & (df_emis["Variable"] == "Emissions|NOx|MAGICC AFOLU|Agriculture"),
        "1750":"2100",
    ]
    .interpolate(axis=1)
    .values.squeeze()
    + df_emis.loc[
        (df_emis["Scenario"] == "ssp245")
        & (df_emis["Region"] == "World")
        & (df_emis["Variable"] == "Emissions|NOx|MAGICC Fossil and Industrial"),
        "1750":"2100",
    ]
    .interpolate(axis=1)
    .values.squeeze()
)


for ispec, species in enumerate(concentration_species):
    species_rcmip_name = species.replace("-", "")
    conc_in = (
        df_conc.loc[
            (df_conc["Scenario"] == "ssp245")
            & (df_conc["Variable"].str.endswith("|" + species_rcmip_name))
            & (df_conc["Region"] == "World"),
            "1750":"2100",
        ]
        .interpolate(axis=1)
        .values.squeeze()
    )
    species_out[species] = conc_in

species_df = pd.DataFrame(species_out, index=range(1750, 2101))


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
erfari_emitted = pd.Series(
    {
        "Sulfur": -0.234228,
        "BC": 0.144702,
        "OC": -0.072143,
        "NH3": -0.033769,
        "NOx": -0.009166,
        "VOC": -0.002573,
        "CH4": -0.002653,
        "N2O": -0.00209,
        "Equivalent effective stratospheric chlorine": -0.00808,
    }
)

# erfari radiative efficiency per Mt or ppb or ppt
re = erfari_emitted / (species_df.loc[2019, :] - species_df.loc[1750, :])
re.dropna(inplace=True)

re["Equivalent effective stratospheric chlorine"] = erfari_emitted[
    "Equivalent effective stratospheric chlorine"
] / (total_eesc_2019 - total_eesc_1750)

scalings = scipy.stats.uniform.rvs(
    np.minimum(re * 2, 0),
    np.maximum(re * 2, 0) - np.minimum(re * 2, 0),
    size=(samples, 9),
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
