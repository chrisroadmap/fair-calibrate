#!/usr/bin/env python
# coding: utf-8

"""Calibrate methane lifetime from precursors and climate."""

# # What affects methane chemical lifetime?
#
# - methane
# - VOCs
# - NOx
# - Ozone
# - halocarbons (specifically ODSs) - but we don't have these!
# - N2O
# - climate
#
# Ozone itself is a function of other precursors: we do not include ozone as a direct
# influence on methane lifetime, and restrict ourselves to directly emitted
# anthropogenic species.
#
# Gill Thornhill published two papers on methane lifetime: one on the chemical
# adjustments to lifetime, and one on the climate adjustments. Both effects will be
# included. We will
#
# 1. take AerChemMIP multi-model means from Gill's papers
# 2. run the lifetime relationship to individual AerChemMIP models in Gill's papers
# 3. find a least squares fit with reasonable sensitivies across the historical
#
# NOTE: the NOx emissions issue is not touched here - we take data directly from MESSAGE.

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import pooch
import scipy.optimize
import scipy.stats
import xarray as xr
from dotenv import load_dotenv
from scipy.interpolate import interp1d

load_dotenv()

print("Calibrating methane lifetime...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

# assert fair_v == __version__
pl.style.use("../../../../../defaults.mplstyle")

# Temperature data
# Use observations 1850-2023 from IGCC
df_temp = pd.read_csv("../../../../../data/forcing/IGCC_GMST_1850-2023.csv")
gmst = np.zeros(274)
gmst[100:] = df_temp["gmst"].values

# Gas cycle functions from fair
def alpha_scaling_exp(
    input,
    baseline,
    normalisation,
    beta,
):
    log_lifetime_scaling = 0
#    for species in ["CH4", "N2O", "VOC", "HC", "NOx", "temp"]:
    for species in ["CH4", "N2O", "VOC", "NOx", "temp"]:
        log_lifetime_scaling = log_lifetime_scaling + (
            np.log(
                1
                + (input[species] - baseline[species])
                / normalisation[species]
                * beta[species]
            )
        )
    return np.exp(log_lifetime_scaling)

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
input_obs = {}

df_emis_message = pd.read_csv(
    "../../../../../data/emissions/message-baseline-2020.csv",
    index_col=0,
)

years_in = df_emis_message.loc[:,'1750.5':].columns.to_numpy().astype(float)

for species in ['NOx', 'VOC']:
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
    input_obs[species] = interpolator(np.arange(1750.5, 2024))

raw_data = (
    df_emis_message.loc[
        (df_emis_message["Scenario"] == "baseline")
        & (df_emis_message["Variable"] == "CH4")
        & (df_emis_message["Region"] == "World"),
        "1750.5":"2105.5",
    ]
    .values.squeeze()
)
interpolator = interp1d(years_in, raw_data)
emis_ch4_obs = interpolator(np.arange(1750.5, 2024))

df_conc_obs = pd.read_csv(
    "../../../../../data/concentrations/ghg_concentrations_1750-2023.csv", index_col=0
)
for year in range(1751, 1850):
    df_conc_obs.loc[year, :] = np.nan
df_conc_obs.sort_index(inplace=True)
df_conc_obs.interpolate(inplace=True)

input_obs["CH4"] = df_conc_obs["CH4"].values[:274]
input_obs["N2O"] = df_conc_obs["N2O"].values[:274]
input_obs["temp"] = gmst[:274]

print(input_obs)
print(emis_ch4_obs)

# hc_eesc = {}
# total_eesc = 0

# for species in hc_species:
#     hc_eesc[species] = calculate_eesc(
#         df_conc_obs[species].values,
#         fractional_release[species],
#         fractional_release["CFC-11"],
#         cl_atoms[species],
#         br_atoms[species],
#     )
#     total_eesc = total_eesc + hc_eesc[species]

# total_eesc_1850 = total_eesc[100]
# input_obs["HC"] = total_eesc[:273]

invect = np.array(
    [
        input_obs["CH4"],
        input_obs["NOx"],
        input_obs["VOC"],
#        input_obs["HC"],
        input_obs["N2O"],
        input_obs["temp"],
    ]
)

# normalisation = ppb / ppt / Mt yr-1 increase from 1850 to 2014
normalisation_obs = {}
# for species in ["CH4", "N2O", "VOC", "NOx", "HC"]:
for species in ["CH4", "N2O", "VOC", "NOx"]:
    normalisation_obs[species] = input_obs[species][264] - input_obs[species][100]
normalisation_obs["temp"] = 1

# baselines are 1850! so "base" lifetime out is for 1750!
baseline_obs = {}
# for species in ["CH4", "N2O", "VOC", "NOx", "HC"]:
for species in ["CH4", "N2O", "VOC", "NOx"]:
    baseline_obs[species] = input_obs[species][100]
baseline_obs["temp"] = 0
natural_emissions_adjustment = emis_ch4_obs[0]


# def fit_precursors(x, rch4, rnox, rvoc, rhc, rn2o, rtemp, rbase, rnat):
# def fit_precursors(x, rch4, rnox, rvoc, rhc, rn2o, rtemp, rbase):
def fit_precursors(x, rch4, rnox, rvoc, rn2o, rtemp, rbase):
    conc_ch4 = np.zeros(274)  # 1750-2023 timebounds
    gas_boxes = 0  # should use correct pi value for CMIP6
    airborne_emissions = 0

    params = {}
    params["CH4"] = rch4
    params["NOx"] = rnox
    params["VOC"] = rvoc
    # params["HC"] = rhc
    params["N2O"] = rn2o
    params["temp"] = rtemp

    inp = {}
    inp["CH4"] = x[0]
    inp["NOx"] = x[1]
    inp["VOC"] = x[2]
    # inp["HC"] = x[3]
    inp["N2O"] = x[3] # x[4]
    inp["temp"] = x[4] # x[5]

    lifetime_scaling = alpha_scaling_exp(
        inp,
        baseline_obs,
        normalisation_obs,
        params,
    )

    for i in range(274):
        conc_ch4[i], gas_boxes, airborne_emissions = one_box(
            emis_ch4_obs[i],
            gas_boxes,
            airborne_emissions,
            burden_per_emission,
            rbase,
            lifetime_scaling[i],
            partition_fraction,
            pre_industrial_concentration=pre_industrial_concentration,
            timestep=1,
            natural_emissions_adjustment=natural_emissions_adjustment,
        )
    return conc_ch4


# widen search bounds for methane feedback on basis that CAT/PRIMAP emissions
# are not complete and concentrations would be underestimated with standard
# lifetime
low = np.array([0.18, -0.46, 0.11, -0.075, -0.039, -0.0463, 6.3])
low = np.array([0.18, -0.46, 0.11, -0.039, -0.0463, 6.3])
high = np.array([0.26, -0.25, 0.27, -0.006, -0.012, 0, 13.4])
high = np.array([0.26, -0.25, 0.27, -0.012, 0, 13.4])
gap = (high - low) / 2

# override temperature
gap[4] = 0
#gap[5] = 0

conc_ch4 = {}

burden_per_emission = 1 / (5.1352e18 / 1e18 * 16.043 / 28.97)
conc_ch4["best_fit"] = np.zeros(274)
partition_fraction = 1
pre_industrial_concentration = 729.2
natural_emissions_adjustment = emis_ch4_obs[0]
gas_boxes = 0
airborne_emissions = 0

# natural bounds from global methane budget (part of GCP)
p, cov = scipy.optimize.curve_fit(
    fit_precursors, invect, input_obs["CH4"], bounds=(low - gap, high + gap)
)

parameters = {}
parameters["best_fit"] = {
    "base": p[5], # p[6],
    "CH4": p[0],
    "NOx": p[1],
    "VOC": p[2],
#    "HC": p[3],
    "N2O": p[3], #p[4],
    "temp": p[4], #p[5],
}

# these are the feedback values per ppb / per Mt that go into fair
print(parameters["best_fit"])
#for specie in ["CH4", "NOx", "VOC", "HC", "N2O"]:
for specie in ["CH4", "NOx", "VOC", "N2O"]:
    print(specie, parameters["best_fit"][specie] / normalisation_obs[specie])

lifetime_scaling = {}
lifetime_scaling["best_fit"] = alpha_scaling_exp(
    input_obs,
    baseline_obs,
    normalisation_obs,
    parameters["best_fit"],
)

print(
    "methane lifetime 1750:",
    lifetime_scaling["best_fit"][0] * parameters["best_fit"]["base"],
)
print("methane lifetime 1850:", parameters["best_fit"]["base"])


if plots:
    os.makedirs(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/",
        exist_ok=True,
    )
    pl.plot(
        np.arange(1750, 2024),
        lifetime_scaling["best_fit"] * parameters["best_fit"]["base"],
        label="best_fit",
    )
    pl.legend()
    pl.ylabel("CH4 chemical lifetime (yr)")
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "ch4_chemical_lifetime_best_fit.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "ch4_chemical_lifetime_best_fit.pdf"
    )
    pl.close()

for i in range(274):
    conc_ch4["best_fit"][i], gas_boxes, airborne_emissions = one_box(
        #        emis_ch4_obs[i] + parameters["best_fit"]["nat"],
        emis_ch4_obs[i],
        gas_boxes,
        airborne_emissions,
        burden_per_emission,
        parameters["best_fit"]["base"],
        lifetime_scaling["best_fit"][i],
        partition_fraction,
        pre_industrial_concentration=pre_industrial_concentration,
        timestep=1,
        natural_emissions_adjustment=natural_emissions_adjustment,
    )


# ### Four panel plot
if plots:
    fig, ax = pl.subplots(1, 2, figsize=(12 / 2.54, 6 / 2.54))
    ax[0].plot(
        np.arange(1750, 2024),
        lifetime_scaling["best_fit"] * parameters["best_fit"]["base"],
        color="0.5",
        label="Best fit",
        lw=1,
    )
    ax[0].set_xlim(1750, 2024)
    ax[0].set_ylabel("yr")
    ax[0].set_title("(a) CH$_4$ lifetime")

    ax[1].plot(
        np.arange(1750, 2024), conc_ch4["best_fit"], color="0.5", label="Best fit", lw=1
    )
    ax[1].plot(
        np.arange(1750, 2024), input_obs["CH4"], color="k", label="observations", lw=1
    )
    ax[1].set_ylabel("ppb")
    ax[1].set_xlim(1750, 2024)
    ax[1].legend(frameon=False)
    ax[1].set_title("(b) CH$_4$ concentration")

    fig.tight_layout()
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "methane_calibrations.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "methane_calibrations.pdf"
    )
    pl.close()

# these are the feedback values per ppb / per Mt that go into fair
#out = np.empty((1, 7))
out = np.empty((1, 6))
out[0, 0] = lifetime_scaling["best_fit"][0] * parameters["best_fit"]["base"]
#for i, specie in enumerate(["CH4", "NOx", "VOC", "HC", "N2O"]):
for i, specie in enumerate(["CH4", "NOx", "VOC", "N2O"]):
    out[0, i + 1] = parameters["best_fit"][specie] / normalisation_obs[specie]
out[0, 5] = parameters["best_fit"]["temp"]
# out[0, 6] = parameters["best_fit"]["temp"]
# out[0, 7] = parameters["best_fit"]["nat"]

df = pd.DataFrame(
    out,
#    columns=["base", "CH4", "NOx", "VOC", "HC", "N2O", "temp"],
    columns=["base", "CH4", "NOx", "VOC", "N2O", "temp"],
    index=["historical_best"],
)
os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/",
    exist_ok=True,
)
df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/"
    "CH4_lifetime.csv"
)
