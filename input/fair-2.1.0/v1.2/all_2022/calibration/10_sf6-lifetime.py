#!/usr/bin/env python
# coding: utf-8

"""Calibrate SF6 lifetime."""

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

print("Calibrating SF6 lifetime...")

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
# concentrations (not those from RCMIP)
df_conc_obs = pd.read_csv('../../../../../data/concentrations/ghg_concentrations_1750-2022.csv', index_col=0)
for year in range(1751, 1850):
    df_conc_obs.loc[year, :] = np.nan
df_conc_obs.sort_index(inplace=True)
df_conc_obs.interpolate(inplace=True)

input_obs = {}
input_obs['SF6'] = df_conc_obs['SF6'].values

df_emis_obs = pd.read_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/primap_ceds_gfed_1750-2022.csv')
emis_obs = df_emis_obs.loc[df_emis_obs['Variable']=='Emissions|SF6', '1750':'2022'].values.squeeze()

# baselines are 1850! so "base" lifetime out is for 1750!
baseline_obs = {}
for species in ["SF6"]:
    baseline_obs[species] = input_obs[species][100]

burden_per_emission = 1 / (5.1352e18 / 1e18 * 146.06 / 28.97)
partition_fraction = 1
pre_industrial_concentration = 0#729.2
natural_emissions_adjustment = 0#emis_ch4[0]

def fit_precursors(x, rbase):
    conc_sf6 = np.zeros(93)
    gas_boxes = 0
    airborne_emissions = 0

    for i in range(93):
        conc_sf6[i], gas_boxes, airborne_emissions = one_box(
            emis_obs[180+i],
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
    return conc_sf6


# natural bounds from global methane budget (part of GCP)
p, cov = scipy.optimize.curve_fit(
    fit_precursors,
    emis_obs[180:],
    input_obs["SF6"][180:],
)

parameters = {}

parameters["best_fit"] = {
    "base": p[0],
}

# these are the feedback values per ppb / per Mt that go into FaIR
print(parameters["best_fit"])

conc_sf6 = np.zeros(93)
gas_boxes = 0
airborne_emissions = 0

for i in range(93):
    conc_sf6[i], gas_boxes, airborne_emissions = one_box(
        emis_obs[180+i],
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

if plots:
    fig, ax = pl.subplots(1, 1, figsize=(3.5, 3.5))

    ax.plot(
        np.arange(1930, 2023), conc_sf6[:], color="0.5", label="Best fit"
    )
    ax.plot(
        np.arange(1930, 2023), input_obs["SF6"][180:], color="k", label="observations"
    )
    ax.set_ylabel("ppt")
    ax.set_xlim(1930, 2023)
    ax.legend(frameon=False)
    ax.set_title("SF$_6$ concentration")

    fig.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "sf6_calibrations.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "sf6_calibrations.pdf"
    )
    pl.close()

# these are the feedback values that go into FaIR
out = np.empty((1, 1))
out[0, 0] = parameters["best_fit"]["base"]

df = pd.DataFrame(
    out,
    columns=["base"],
    index=["historical_best"],
)
os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/",
    exist_ok=True
)
df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
    "SF6_lifetime.csv"
)
