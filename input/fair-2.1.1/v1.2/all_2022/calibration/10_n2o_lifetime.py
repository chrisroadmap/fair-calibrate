#!/usr/bin/env python
# coding: utf-8

"""Calibrate N2O lifetime and natural emissions."""

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import pooch
import scipy.optimize
import scipy.stats
import xarray as xr
from dotenv import load_dotenv
from fair import __version__

load_dotenv()

print("Calibrating N2O lifetime...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

assert fair_v == __version__
pl.style.use("../../../../../defaults.mplstyle")

# Temperature data
# Use observations 1850-2022, then simulate an SSP3-7.0 climate with a linear warming
# rate to 4C in 2100.

df_temp = pd.read_csv("../../../../../data/forcing/AR6_GMST.csv")
gmst = np.zeros(351)
gmst[100:273] = df_temp["gmst"].values
gmst[273:351] = np.linspace(gmst[268:273].mean(), 4, 78)


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
# concentrations (not those from RCMIP)
df_conc_obs = pd.read_csv('../../../../../data/concentrations/ghg_concentrations_1750-2022.csv', index_col=0)
for year in range(1751, 1850):
    df_conc_obs.loc[year, :] = np.nan
df_conc_obs.sort_index(inplace=True)
df_conc_obs.interpolate(inplace=True)

input_obs = {}
input_obs['N2O'] = df_conc_obs['N2O'].values
input_obs['temp'] = gmst[:273]

df_emis_obs = pd.read_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/primap_ceds_gfed_1750-2022.csv')
emis_obs = df_emis_obs.loc[df_emis_obs['Variable']=='Emissions|N2O', '1750':'2022'].values.squeeze()

invect = np.array(
    [input_obs["N2O"], input_obs["temp"]]
)

# baselines are 1850! so "base" lifetime out is for 1750!
baseline_obs = {}
for species in ["N2O"]:
    baseline_obs[species] = input_obs[species][100]
baseline_obs["temp"] = 0

burden_per_emission = 1 / (5.1352e18 / 1e18 * 44.013 / 28.97)
partition_fraction = 1
pre_industrial_concentration = 0#729.2
natural_emissions_adjustment = 0#emis_ch4[0]

def fit_precursors(x, rbase, rnat):
    conc_n2o = np.zeros(273)
    gas_boxes = 270.1 / burden_per_emission
    airborne_emissions = 270.1 / burden_per_emission

    for i in range(273):
        conc_n2o[i], gas_boxes, airborne_emissions = one_box(
            emis_obs[i] + rnat,
            gas_boxes,
            airborne_emissions,
            burden_per_emission,
            rbase,
            1,#lifetime_scaling[i],
            partition_fraction,
            pre_industrial_concentration=0,
            timestep=1,
            natural_emissions_adjustment=0#natural_emissions_adjustment,
        )
    return conc_n2o


# natural bounds from global methane budget (part of GCP)
p, cov = scipy.optimize.curve_fit(
    fit_precursors,
    emis_obs,
    input_obs["N2O"],
    bounds=(  # AerChemMIP min to max range
        (100, 12.5),
        (120, 18.9),
    ),
)

parameters = {}

parameters["best_fit"] = {
    "base": p[0],
    "nat": p[1]
}

# these are the feedback values per ppb / per Mt that go into FaIR
print(parameters["best_fit"])

conc_n2o = np.zeros(273)
gas_boxes = 270.1 / burden_per_emission
airborne_emissions = 270.1 / burden_per_emission

for i in range(273):
    conc_n2o[i], gas_boxes, airborne_emissions = one_box(
        emis_obs[i] + parameters["best_fit"]["nat"],
        gas_boxes,
        airborne_emissions,
        burden_per_emission,
        parameters["best_fit"]["base"],
        1,#lifetime_scaling["best_fit"][i],
        partition_fraction,
        pre_industrial_concentration=0,
        timestep=1,
        natural_emissions_adjustment=0#natural_emissions_adjustment,
    )


# ### Four panel plot
if plots:
    # ar6_colors = {
    #     "ssp119": "#00a9cf",
    #     "ssp126": "#003466",
    #     "ssp245": "#f69320",
    #     "ssp370": "#df0000",
    #     "ssp434": "#2274ae",
    #     "ssp460": "#b0724e",
    #     "ssp534-over": "#92397a",
    #     "ssp585": "#980002",
    # }

    fig, ax = pl.subplots(1, 1, figsize=(3.5, 3.5))
#    fig, ax = pl.subplots(1, 3, figsize=(12, 3.5))

    ax.plot(
        np.arange(1750, 2023), conc_n2o, color="0.5", label="Best fit"
    )
    ax.plot(
        np.arange(1750, 2023), input_obs["N2O"], color="k", label="observations"
    )
    ax.set_ylabel("ppb")
    ax.set_xlim(1750, 2023)
    ax.legend(frameon=False)
    ax.set_title("N$_2$O concentration")

    # for ssp in [
    #     "ssp119",
    #     "ssp126",
    #     "ssp434",
    #     "ssp534-over",
    #     "ssp245",
    #     "ssp460",
    #     "ssp370",
    #     "ssp585",
    # ]:
    #     ax[2].plot(
    #         np.arange(1750, 2101), conc_ch4[ssp], label=ssp, color=ar6_colors[ssp]
    #     )
    # ax[2].set_ylabel("ppb")
    # ax[2].set_title("(c) Best fit CH$_4$ projections")
    # ax[2].set_xlim(1750, 2100)
    # ax[2].legend(frameon=False)

    fig.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "n2o_calibrations.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "n2o_calibrations.pdf"
    )
    pl.close()

# these are the feedback values that go into FaIR
out = np.empty((1, 2))
out[0, 0] = parameters["best_fit"]["base"]
out[0, 1] = parameters["best_fit"]["nat"]

df = pd.DataFrame(
    out,
    columns=["base", "natural_emissions"],
    index=["historical_best"],
)
os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/",
    exist_ok=True
)
df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
    "N2O_lifetime.csv"
)
