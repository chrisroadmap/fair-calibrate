#!/usr/bin/env python
# coding: utf-8

"""Calibrate emissions scale factors for non-CH4, non-N2O."""

# PRIMAP/CAT emissions are too low, so scale them up. We retain the IPCC's best
# estimate lifetime.

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
        "Halon-1202",
    ]
]

da_emis_obs = xr.load_dataarray(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "ssps_harmonized_1750-2499.nc"
)

renames = {specie: specie for specie in species}
renames["HFC-43-10mee"] = "HFC-4310mee"
renames["n-C4F10"] = "C4F10"
renames["n-C5F12"] = "C5F12"
renames["n-C6F14"] = "C6F14"

emissions_scalings = {}

for specie in species:
    input_obs = {}
    input_obs[specie] = df_conc_obs[specie].values[:273]  # 1750-2022 timepoints
    input_obs["temp"] = gmst[:273]  # 1750-2022 timepoints

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

    conc_ssp = {}
    for ssp in [
        "ssp119",
        "ssp126",
        "ssp245",
        "ssp370",
        "ssp434",
        "ssp460",
        "ssp534-over",
        "ssp585",
    ]:
        emis_ssp = da_emis_obs.loc[
            dict(
                specie=renames[specie], timepoints=np.arange(1750.5, 2101), scenario=ssp
            )
        ].values.squeeze()
        natural_emissions_adjustment = emis_ssp[0]
        conc_ssp[ssp] = np.zeros(351)
        gas_boxes = 0
        airborne_emissions = 0
        for i in range(351):
            conc_ssp[ssp][i], gas_boxes, airborne_emissions = one_box(
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

    scalings_df = pd.read_csv(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
        "emissions_scalings.csv",
        index_col=0,
    )
    emis_ssp = (
        da_emis_obs.loc[
            dict(
                specie=renames[specie],
                timepoints=np.arange(1750.5, 2101),
                scenario="ssp245",
            )
        ].values.squeeze()
        / scalings_df.loc["historical_best", renames[specie]]
    )
    conc_unscaled_hist = np.zeros(273)
    natural_emissions_adjustment = emis_ssp[0]
    gas_boxes = 0
    airborne_emissions = 0
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

    # Two panel plot
    if plots:
        os.makedirs(
            f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/minor_ghgs/",
            exist_ok=True,
        )

        fig, ax = pl.subplots(1, 2, figsize=(12 / 2.54, 6 / 2.54))

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
            conc_ssp["ssp245"][:273],
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

        for ssp in [
            "ssp119",
            "ssp126",
            "ssp434",
            "ssp534-over",
            "ssp245",
            "ssp460",
            "ssp370",
            "ssp585",
        ]:
            gas = (
                rcmip_df.loc[
                    (rcmip_df["Region"] == "World")
                    & (rcmip_df["Scenario"] == ssp)
                    & (
                        rcmip_df["Variable"].str.endswith(
                            f"|{renames[specie].replace('-', '')}"
                        )
                    ),
                    "1750":"2100",
                ]
                .interpolate(axis=1)
                .squeeze()
            )

            ax[1].plot(
                np.arange(1750, 2101),
                conc_ssp[ssp],
                label=ssp,
                color=ar6_colors[ssp],
                lw=1,
            )
            ax[1].plot(np.arange(1750, 2101), gas, color=ar6_colors[ssp], lw=0.3)
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
