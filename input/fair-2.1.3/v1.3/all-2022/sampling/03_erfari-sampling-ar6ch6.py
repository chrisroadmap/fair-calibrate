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
import scipy.stats
from dotenv import load_dotenv

load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

df_emis = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "slcf_emissions_1750-2022.csv",
    index_col=0,
)
df_conc = pd.read_csv(
    "../../../../../data/concentrations/ghg_concentrations_1750-2022.csv", index_col=0
)
for year in range(1751, 1850):
    df_conc.loc[year, :] = np.nan
df_conc.sort_index(inplace=True)
df_conc.interpolate(inplace=True)

# these are the present day ERFari which comes from AR6 WG1
# source: https://github.com/sarambl/AR6_CH6_RCMIPFIGS/blob/master/ar6_ch6_rcmipfigs/
# data_out/fig6_12_ts15_historic_delta_GSAT/2019_ERF_est.csv
# they sum to exactly -0.22 W/m2, for 2019
# Calculate a radiative efficiency for each species from CEDS and updated
# concentrations.
df_ari_emitted_mean = pd.read_csv(
    "../../../../../data/forcing/table_mean_thornhill_collins_orignames.csv",
    index_col=0,
)
erfari_emitted = pd.Series(df_ari_emitted_mean["Aerosol"])
erfari_emitted.rename_axis(None, inplace=True)
erfari_emitted.rename(
    {"HC": "Equivalent effective stratospheric chlorine", "SO2": "Sulfur"}, inplace=True
)

df_ari_emitted_std = pd.read_csv(
    "../../../../../data/forcing/table_std_thornhill_collins_orignames.csv", index_col=0
)
erfari_emitted_std = pd.Series(df_ari_emitted_std["Aerosol_sd"])
erfari_emitted_std.rename_axis(None, inplace=True)
erfari_emitted_std.rename(
    {"HC": "Equivalent effective stratospheric chlorine", "SO2": "Sulfur"}, inplace=True
)

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
name_conv["Sulfur"] = "SO2"
name_conv["VOC"] = "NMVOC"

species_out = {}
for ispec, species in enumerate(emitted_species):
    species_out[species] = df_emis[name_conv[species]].values

for ispec, species in enumerate(concentration_species):
    species_out[species] = df_conc[species].values[:273]

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
total_eesc = np.zeros(273)
for species in cl_atoms:
    hc_eesc[species] = calculate_eesc(
        species_df.loc[:, species],
        fractional_release[species],
        fractional_release["CFC-11"],
        cl_atoms[species],
        br_atoms[species],
    )
    total_eesc = total_eesc + hc_eesc[species]

# erfari radiative efficiency per Mt or ppb or ppt
re = erfari_emitted / (species_df.loc[2019, :] - species_df.loc[1750, :])
re.dropna(inplace=True)

re["CH4"] = erfari_emitted["CH4"] / (
    species_df.loc[2019, "CH4"] - species_df.loc[1750, "CH4"]
)
re["N2O"] = erfari_emitted["N2O"] / (
    species_df.loc[2019, "N2O"] - species_df.loc[1750, "N2O"]
)
re["Equivalent effective stratospheric chlorine"] = erfari_emitted[
    "Equivalent effective stratospheric chlorine"
] / (total_eesc.loc[2019] - total_eesc.loc[1750])

re_std = erfari_emitted_std / (species_df.loc[2019, :] - species_df.loc[1750, :])
re_std.dropna(inplace=True)
re_std["CH4"] = erfari_emitted_std["CH4"] / (
    species_df.loc[2019, "CH4"] - species_df.loc[1750, "CH4"]
)
re_std["N2O"] = erfari_emitted_std["N2O"] / (
    species_df.loc[2019, "N2O"] - species_df.loc[1750, "N2O"]
)
re_std["Equivalent effective stratospheric chlorine"] = erfari_emitted_std[
    "Equivalent effective stratospheric chlorine"
] / (total_eesc.loc[2019] - total_eesc.loc[1750])

print(re)
print(re_std)

erfari_best = (
    pd.concat(
        (
            (re * species_df)[["BC", "OC", "Sulfur", "NOx", "VOC", "NH3", "CH4", "N2O"]]
            - (
                re
                * species_df.loc[
                    1750, ["BC", "OC", "Sulfur", "NOx", "VOC", "NH3", "CH4", "N2O"]
                ]
            ),
            re["Equivalent effective stratospheric chlorine"]
            * (total_eesc - total_eesc.loc[1750]),
        ),
        axis=1,
    )
    .dropna(axis=1)
    .sum(axis=1)
)

# we need to map the -0.27 +/- 0.57 to -0.3 +/- 0.3 which is the IPCC AR6 assessment
NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)
best_scale = -0.3 / erfari_best.loc[2005:2014].mean()
unc_scale = 0.3 / (
    (erfari_best.loc[2005:2014].mean() / -0.22)
    * np.sqrt((erfari_emitted_std**2).sum())
    * NINETY_TO_ONESIGMA
)

# convert to numpy for efficiency
erfari_re_samples = pd.DataFrame(
    scipy.stats.norm.rvs(
        re * best_scale,
        re_std * unc_scale,
        size=(samples, 9),
        random_state=3729329,
    ),
    columns=re.index,
)[
    [
        "BC",
        "OC",
        "Sulfur",
        "NOx",
        "VOC",
        "NH3",
        "CH4",
        "N2O",
        "Equivalent effective stratospheric chlorine",
    ]
]

print(erfari_re_samples)

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/",
    exist_ok=True,
)

erfari_re_samples.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "aerosol_radiation.csv",
    index=False,
)
