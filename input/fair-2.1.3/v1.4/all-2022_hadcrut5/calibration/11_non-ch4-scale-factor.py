#!/usr/bin/env python
# coding: utf-8

"""Calibrate emissions scale factors for non-CH4."""

# PRIMAP emissions are too low in most cases, so scale them up to maintain consistency
# with observed concentrations. We retain the IPCC's best estimate lifetime.

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import pooch
import scipy.optimize
from dotenv import load_dotenv
from fair import __version__
from fair.fair import DEFAULT_SPECIES_CONFIG_FILE

load_dotenv()

print("Calibrating GHG emissions scale factors...")

ar6_colors = {
    "ssp119": "#00a9cf",
    "ssp126": "#003466",
    "ssp245": "#f69320",
    "ssp370": "#df0000",
    "ssp434": "#2274ae",
    "ssp460": "#b0724e",
    "ssp534-over": "#92397a",
    "ssp585": "#980002",
}

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

assert fair_v == __version__
pl.style.use("../../../../../defaults.mplstyle")

# Temperature data
# Use observations 1850-2022 from IGCC, then use ssp370 projections from IPCC
df_temp = pd.read_csv("../../../../../data/forcing/ssp_strawman_warming.csv")
gmst = df_temp["ssp370"].values


# put this into a simple one box model
def one_box(
    emissions,
    gas_boxes_old,
    airborne_emissions_old,
    burden_per_emission,
    lifetime,
    alpha_lifetime,
    partition_fraction,
    pre_industrial_concentration,
    timestep=1,
    natural_emissions_adjustment=0,
):
    effective_lifetime = alpha_lifetime * lifetime
    decay_rate = timestep / (effective_lifetime)
    decay_factor = np.exp(-decay_rate)
    gas_boxes_new = (
        partition_fraction
        * (emissions - natural_emissions_adjustment)
        * 1
        / decay_rate
        * (1 - decay_factor)
        * timestep
        + gas_boxes_old * decay_factor
    )
    airborne_emissions_new = gas_boxes_new
    concentration_out = (
        pre_industrial_concentration + burden_per_emission * airborne_emissions_new
    )
    return concentration_out, gas_boxes_new, airborne_emissions_new


rcmip_file = pooch.retrieve(
    url="https://zenodo.org/records/4589756/files/"
    "rcmip-concentrations-annual-means-v5-1-0.csv",
    known_hash="md5:0d82c3c3cdd4dd632b2bb9449a5c315f",
    path=datadir,
    progressbar=progress,
)

rcmip_df = pd.read_csv(rcmip_file)


# Find least squares sensible historical fit using best estimate emissions and
# concentrations from our calibration emissions and observed concentrations
df_conc_obs = pd.read_csv(
    "../../../../../data/concentrations/ghg_concentrations_1750-2022.csv", index_col=0
)
for year in range(1751, 1850):
    df_conc_obs.loc[year, :] = np.nan
df_conc_obs.sort_index(inplace=True)
df_conc_obs.interpolate(inplace=True)

# Exclude CO2, CH4, and any gas that is in the observational concentrations but not
# the RCMIP set
species = df_conc_obs.columns
species = [
    specie
    for specie in species
    if specie
    not in [
        "CO2",
        "CH4",
        "CFC-13",
        "i-C6F14",
        "CFC-112",
        "CFC-112a",
        "CFC-113a",
        "CFC-114a",
        "HCFC-133a",
        "HCFC-31",
        "HCFC-124",
        "Halon-1202",
    ]
]

df_emis_obs = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "all_1750-2022.csv"
)

renames = {specie: specie for specie in species}
renames["HFC-43-10mee"] = "HFC-4310mee"
renames["n-C4F10"] = "C4F10"
renames["n-C5F12"] = "C5F12"
renames["n-C6F14"] = "C6F14"

emissions_scalings = {}

for specie in species:
    input_obs = {}
    input_obs[specie] = df_conc_obs[specie].values[:270]  # 1750-2020 timepoints
    input_obs["temp"] = gmst[:270]  # 1750-2020 timepoints

    emis_obs = df_emis_obs.loc[
        df_emis_obs["Variable"] == f"Emissions|{renames[specie]}", "1750":"2019"
    ].values.squeeze()  # 1750-2019 timepoints

    df_defaults = pd.read_csv(DEFAULT_SPECIES_CONFIG_FILE, index_col=0)
    lifetime = df_defaults.loc[renames[specie], "unperturbed_lifetime0"]
    molecular_weight = df_defaults.loc[renames[specie], "molecular_weight"]

    # baselines are 1850! so "base" lifetime out is for 1750!
    baseline_obs = {}
    baseline_obs[specie] = input_obs[specie][100]
    baseline_obs["temp"] = 0

    burden_per_emission = 1 / (5.1352e18 / 1e18 * molecular_weight / 28.97)
    partition_fraction = 1
    pre_industrial_concentration = df_conc_obs[specie].values[0]
    natural_emissions_adjustment = emis_obs[0]

    def find_scale_factor(sf):
        conc_n2o = np.zeros(270)
        gas_boxes = 0
        airborne_emissions = 0

        for i in range(270):
            conc_n2o[i], gas_boxes, airborne_emissions = one_box(
                emis_obs[i] * sf[0],
                gas_boxes,
                airborne_emissions,
                burden_per_emission,
                lifetime,
                1,
                partition_fraction,
                pre_industrial_concentration=pre_industrial_concentration,
                timestep=1,
                natural_emissions_adjustment=natural_emissions_adjustment * sf[0],
            )
        return conc_n2o[-1] - input_obs[specie][-1]
        # return conc_n2o[-2:].mean() - input_obs["N2O"][-2]
        # mean of 2018 and 2019 tbs ; 2018 timepoint

    rootsol = scipy.optimize.root(find_scale_factor, 1.1)

    parameters = {}
    parameters["best_fit"] = {"scale": rootsol.x[0]}

    emissions_scalings[renames[specie]] = parameters["best_fit"]["scale"]

# these are the emissions scaling values that we apply
df = pd.DataFrame(emissions_scalings, index=["historical_best"])
os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/",
    exist_ok=True,
)
df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
    "emissions_scalings.csv"
)
