#!/usr/bin/env python
# coding: utf-8

"""Aerosol-radiation calibration."""
# This calibration is currently WITHOUT EESC.

import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import scipy.stats
from dotenv import load_dotenv

load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

# Sampling with updated emissions.
df_emis_message = pd.read_csv(
    "../../../../../data/emissions/message-baseline-2020.csv",
    index_col=0,
)

# and observed concentrations.
df_conc_obs = pd.read_csv(
    "../../../../../data/concentrations/ghg_concentrations_1750-2023.csv", index_col=0
)
for year in range(1751, 1850):
    df_conc_obs.loc[year, :] = np.nan
df_conc_obs.sort_index(inplace=True)
df_conc_obs.interpolate(inplace=True)

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
]

concentration_species = [
    "CH4",
    "N2O",
]

species_out = {}

years_in = df_emis_message.loc[:,'1750.5':].columns.to_numpy().astype(float)
for ispec, species in enumerate(emitted_species):
    raw_data = (
        df_emis_message.loc[
            (df_emis_message["Scenario"] == "baseline")
            & (df_emis_message["Variable"] == species)
            & (df_emis_message["Region"] == "World"),
            "1750.5":"2105.5",
        ]
        .values.squeeze()
    )
    interpolator = interp1d(years_in, raw_data)
    species_out[species] = interpolator(np.arange(1750.5, 2023))

for ispec, species in enumerate(concentration_species):
    raw_data = df_conc_obs[species].values
    interpolator = interp1d(np.arange(1750, 2024), raw_data)
    species_out[species] = interpolator(np.arange(1750.5, 2023))

species_df = pd.DataFrame(species_out, index=np.arange(1750.5, 2023))

# erfari radiative efficiency per Mt or ppb or ppt
re = erfari_emitted / (species_df.loc[2019.5, :] - species_df.loc[1750.5, :])
re.dropna(inplace=True)

re_std = erfari_emitted_std / (species_df.loc[2019.5, :] - species_df.loc[1750.5, :])
re_std.dropna(inplace=True)

print(re)
print(re_std)

erfari_best = (
    pd.concat(
        (
            (re * species_df)[["BC", "OC", "Sulfur", "NOx", "VOC", "NH3", "CH4", "N2O"]]
            - (
                re
                * species_df.loc[
                    1750.5, ["BC", "OC", "Sulfur", "NOx", "VOC", "NH3", "CH4", "N2O"]
                ]
            ),
        ),
        axis=1,
    )
    .dropna(axis=1)
    .sum(axis=1)
)

# we need to map the 2019 mean and stdev to -0.3 +/- 0.3 for 2005-2014 which is the
# IPCC AR6 assessment
NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)
best_scale = -0.3 / erfari_best.loc[2005.5:2014.5].mean()
unc_scale = 0.3 / (
    (erfari_best.loc[2005.5:2014.5].mean() / -0.22)
    * np.sqrt((erfari_emitted_std**2).sum())
    * NINETY_TO_ONESIGMA
)

# convert to numpy for efficiency
erfari_re_samples = pd.DataFrame(
    scipy.stats.norm.rvs(
        re * best_scale,
        re_std * unc_scale,
        size=(samples, 8),
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
    ]
]

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/",
    exist_ok=True,
)

erfari_re_samples.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "aerosol_radiation.csv",
    index=False,
)
