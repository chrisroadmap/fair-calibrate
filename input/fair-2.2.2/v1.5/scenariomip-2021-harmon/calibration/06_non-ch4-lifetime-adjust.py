#!/usr/bin/env python
# coding: utf-8

"""Calibrate lifetime scale factors for non-CH4."""

# This script compares emissions to concentrations over the historical and
# modifies the IPCC best estimate lifetime if necessary (presuming there is
# some uncertainty in this, and it's not an order of magnitude different).

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



# Find least squares sensible historical fit using best estimate emissions and
# concentrations from our calibration emissions and observed concentrations
df_conc_obs = pd.read_csv(
    "../../../../../data/concentrations/ghg_concentrations_1750-2023.csv", index_col=0
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
    ]
]

df_emis_obs = pd.read_csv(
    f"../../../../../data/emissions/"
    "historical_emissions_1750-2021.csv"
)

renames = {specie: specie for specie in species}
renames["HFC-43-10mee"] = "HFC-4310mee"
renames["n-C4F10"] = "C4F10"
renames["n-C5F12"] = "C5F12"
renames["n-C6F14"] = "C6F14"

lifetime_scalings = {}

for specie in species:
    input_obs = {}
    input_obs[specie] = df_conc_obs[specie].values[:272]  # 1750-2021 timepoints

    emis_obs = df_emis_obs.loc[
        df_emis_obs["variable"] == f"Emissions|{renames[specie]}", "1750":"2021"
    ].values.squeeze()  # 1750-2021 timepoints

    df_defaults = pd.read_csv(DEFAULT_SPECIES_CONFIG_FILE, index_col=0)
    lifetime = df_defaults.loc[renames[specie], "unperturbed_lifetime0"]
    molecular_weight = df_defaults.loc[renames[specie], "molecular_weight"]

    # baselines are 1850, so "base" lifetime out is for 1750
    baseline_obs = {}
    baseline_obs[specie] = input_obs[specie][100]

    burden_per_emission = 1 / (5.1352e18 / 1e18 * molecular_weight / 28.97)
    partition_fraction = 1
    pre_industrial_concentration = df_conc_obs[specie].values[0]
    print(specie)
    natural_emissions_adjustment = emis_obs[0]

    def find_scale_factor(sf):
        conc_projected = np.zeros(272)  # 1751-2022 timebounds
        gas_boxes = 0
        airborne_emissions = 0

        for i in range(272):
            conc_projected[i], gas_boxes, airborne_emissions = one_box(
                emis_obs[i],
                gas_boxes,
                airborne_emissions,
                burden_per_emission,
                lifetime * sf[0],
                1,
                partition_fraction,
                pre_industrial_concentration=pre_industrial_concentration,
                timestep=1,
                natural_emissions_adjustment=natural_emissions_adjustment,
            )
        return conc_projected[-2:].mean() - input_obs[specie][-1]
        # mean of 2021 and 2022 timebounds ; 2021 timepoint

    rootsol = scipy.optimize.root(find_scale_factor, 1)

    parameters = {}
    parameters["best_fit"] = {"scale": rootsol.x[0]}

    lifetime_scalings[renames[specie]] = parameters["best_fit"]["scale"]

df = pd.DataFrame(lifetime_scalings, index=["historical_best"])
os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/",
    exist_ok=True,
)
df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
    "lifetime_scalings.csv"
)
