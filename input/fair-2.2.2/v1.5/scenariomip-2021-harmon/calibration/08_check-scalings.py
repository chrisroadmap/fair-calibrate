#!/usr/bin/env python
# coding: utf-8

"""Calibrate emissions scale factors for non-CH4."""

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import pooch
import xarray as xr
from dotenv import load_dotenv
from fair import __version__
from fair.fair import DEFAULT_SPECIES_CONFIG_FILE
from fair.structure.units import desired_concentration_units

load_dotenv()

print("Sense check of harmonized scalings...")

scenarios = [
    "SSP3 - High Emissions",
    "SSP1 - Low Emissions",
    "SSP1 - Very Low Emissions",
    "SSP2 - Medium Emissions",
    "SSP5 - High Emissions",
]


ar7_colors = {
    "SSP1 - Very Low Emissions": "#00a9cf",
    "SSP1 - Low Emissions": "#003466",
    "SSP2 - Medium Emissions": "#f69320",
    "SSP3 - High Emissions": "#df0000",
    "SSP5 - High Emissions": "#980002",
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
# Use observations 1850-2023 from IGCC, then use ssp370 projections from IPCC
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
        "SO2F2",
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

df_emis = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "scenario_subset_1750-2100.csv",
    index_col=[0, 1, 2, 3, 4]
)

renames = {specie: specie for specie in species}
renames["HFC-43-10mee"] = "HFC-4310mee"
renames["n-C4F10"] = "C4F10"
renames["n-C5F12"] = "C5F12"
renames["n-C6F14"] = "C6F14"

df_lifetime_scalings = pd.read_csv(f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
    "lifetime_scalings.csv",
    index_col = 0
)

for specie in species:
    input_obs = {}
    input_obs[specie] = df_conc_obs[specie].values[:273]  # 1750-2022 timepoints
    input_obs["temp"] = gmst[:273]  # 1750-2022 timepoints

    df_defaults = pd.read_csv(DEFAULT_SPECIES_CONFIG_FILE, index_col=0)
    lifetime = df_defaults.loc[renames[specie], "unperturbed_lifetime0"]
    molecular_weight = df_defaults.loc[renames[specie], "molecular_weight"]

    # baselines are 1850 so "base" lifetime out is for 1750!
    baseline_obs = {}
    baseline_obs[specie] = input_obs[specie][100]
    baseline_obs["temp"] = 0

    burden_per_emission = 1 / (5.1352e18 / 1e18 * molecular_weight / 28.97)
    partition_fraction = 1
    pre_industrial_concentration = df_conc_obs[specie].values[0]

    emis_ssp = (
        df_emis.query(f"scenario=='SSP2 - Medium Emissions' and variable=='{renames[specie]}'").values.squeeze()
    )
    conc_unscaled_hist = np.zeros(273)
    natural_emissions_adjustment = emis_ssp[0]
    gas_boxes = 0
    airborne_emissions = 0

    lifetime_scaling = df_lifetime_scalings.loc['historical_best', renames[specie]]

    for i in range(273):
        conc_unscaled_hist[i], gas_boxes, airborne_emissions = one_box(
            emis_ssp[i],
            gas_boxes,
            airborne_emissions,
            burden_per_emission,
            lifetime,
            1,
            partition_fraction,
            pre_industrial_concentration=pre_industrial_concentration,
            timestep=1,
            natural_emissions_adjustment=natural_emissions_adjustment,
        )

    conc_scaled_hist = np.zeros(273)
    natural_emissions_adjustment = emis_ssp[0]
    gas_boxes = 0
    airborne_emissions = 0

    for i in range(273):
        conc_scaled_hist[i], gas_boxes, airborne_emissions = one_box(
            emis_ssp[i],
            gas_boxes,
            airborne_emissions,
            burden_per_emission,
            lifetime * lifetime_scaling,
            1,
            partition_fraction,
            pre_industrial_concentration=pre_industrial_concentration,
            timestep=1,
            natural_emissions_adjustment=natural_emissions_adjustment,
        )

    print(specie, lifetime, lifetime_scaling)

    # Two panel plot
    if plots:
        os.makedirs(
            f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/minor_ghgs/",
            exist_ok=True,
        )

        fig, ax = pl.subplots(1, 2, figsize=(36 / 2.54, 18 / 2.54))

        ax[0].plot(
            np.arange(1750, 2023),
            input_obs[specie],
            color="k",
            label="Historical best estimate",
            lw=1,
        )
        ax[0].plot(
            np.arange(1750, 2023),
            conc_unscaled_hist[:273],
            color="0.5",
            ls=":",
            lw=1,
            label="Unscaled emissions",
        )
        ax[0].plot(
            np.arange(1750, 2023),
            conc_scaled_hist[:273],
            color="0.5",
            ls="-",
            # color='r',
            lw=1,
            label="Scaled emissions",
        )
        ax[0].set_ylabel(desired_concentration_units[renames[specie]])
        ax[0].set_xlim(1750, 2023)
        ax[0].legend(frameon=False)
        ax[0].set_title(f"(a) {specie} concentration")

        conc_scaled_ssp = {}
        conc_unscaled_ssp = {}
        for ssp in scenarios:
            emis_ssp = df_emis.query(f"scenario=='{ssp}' and variable=='{renames[specie]}'").values.squeeze()
            natural_emissions_adjustment = emis_ssp[0]
            conc_scaled_ssp[ssp] = np.zeros(351)
            gas_boxes = 0
            airborne_emissions = 0
            for i in range(351):
                conc_scaled_ssp[ssp][i], gas_boxes, airborne_emissions = one_box(
                    emis_ssp[i],
                    gas_boxes,
                    airborne_emissions,
                    burden_per_emission,
                    lifetime * lifetime_scaling,
                    1,
                    partition_fraction,
                    pre_industrial_concentration=pre_industrial_concentration,
                    timestep=1,
                    natural_emissions_adjustment=natural_emissions_adjustment,
                )

            natural_emissions_adjustment = emis_ssp[0]
            conc_unscaled_ssp[ssp] = np.zeros(351)
            gas_boxes = 0
            airborne_emissions = 0
            for i in range(351):
                conc_unscaled_ssp[ssp][i], gas_boxes, airborne_emissions = one_box(
                    emis_ssp[i],
                    gas_boxes,
                    airborne_emissions,
                    burden_per_emission,
                    lifetime,
                    1,
                    partition_fraction,
                    pre_industrial_concentration=pre_industrial_concentration,
                    timestep=1,
                    natural_emissions_adjustment=natural_emissions_adjustment,
                )

            ax[1].plot(
                np.arange(1750, 2101),
                conc_unscaled_ssp[ssp],
                label=ssp,
                color=ar7_colors[ssp],
                lw=1,
                ls=':',
            )

            ax[1].plot(
                np.arange(1750, 2101),
                conc_scaled_ssp[ssp],
                label=ssp,
                color=ar7_colors[ssp],
                lw=1,
            )
        ax[1].set_ylabel(desired_concentration_units[renames[specie]])
        ax[1].set_title(f"(b) Best fit {specie} projections")
        ax[1].set_xlim(1750, 2100)
        ax[1].legend(frameon=False)

        fig.tight_layout()
        pl.savefig(
            f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/minor_ghgs/"
            f"{specie}_calibrations.png"
        )
        pl.savefig(
            f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/minor_ghgs/"
            f"{specie}_calibrations.pdf"
        )
        pl.close()

        if specie=='N2O':
            print(conc_unscaled_ssp)
