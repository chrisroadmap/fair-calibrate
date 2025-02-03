#!/usr/bin/env python
# coding: utf-8

"""Calibrate methane lifetime from precursors and climate."""

# # What affects methane chemical lifetime?
#
# - methane
# - VOCs
# - NOx
# - Ozone
# - halocarbons (specifically ODSs)
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
# NOTE: we fix an error with NOx unit conversion in RCMIP here, and scale up biomass
# burning emissions by the NO2/NO molecular weight factor.

import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import pooch
import scipy.optimize
import scipy.stats
import xarray as xr
from dotenv import load_dotenv

load_dotenv()

print("Calibrating methane lifetime...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

rcmip_file = pooch.retrieve(
    url="https://zenodo.org/records/4589756/files/"
    "rcmip-concentrations-annual-means-v5-1-0.csv",
    known_hash="md5:0d82c3c3cdd4dd632b2bb9449a5c315f",
    path=datadir,
    progressbar=progress,
)
rcmip_df = pd.read_csv(rcmip_file)

# assert fair_v == __version__
pl.style.use("../../../../../defaults.mplstyle")

# Temperature data
# Use observations 1850-2023 from IGCC, then use ssp370 projections from IPCC
df_temp = pd.read_csv("../../../../../data/forcing/ssp_strawman_warming.csv")
gmst = df_temp["ssp370"].values

# Get emissions and concentrations: from RCMIP for model tuning stage
rcmip_emissions_file = pooch.retrieve(
    url="https://zenodo.org/records/4589756/files/"
    "rcmip-emissions-annual-means-v5-1-0.csv",
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
    path=datadir,
    progressbar=progress,
)

rcmip_concentration_file = pooch.retrieve(
    url=(
        "https://zenodo.org/records/4589756/files/"
        "rcmip-concentrations-annual-means-v5-1-0.csv"
    ),
    known_hash="md5:0d82c3c3cdd4dd632b2bb9449a5c315f",
    path=datadir,
    progressbar=progress,
)

df_emis = pd.read_csv(rcmip_emissions_file)
df_conc = pd.read_csv(rcmip_concentration_file)
input = {}
hc_input = {}

conc_species = ["CH4", "N2O"]
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

for species in conc_species:
    input[species] = (
        df_conc.loc[
            (df_conc["Scenario"] == "ssp370")
            & (df_conc["Variable"].str.endswith(species))
            & (df_conc["Region"] == "World"),
            "1750":"2100",
        ]
        .interpolate(axis=1)
        .values.squeeze()
    )

for species in hc_species:
    species_rcmip_name = species.replace("-", "")
    hc_input[species] = (
        df_conc.loc[
            (df_conc["Scenario"] == "ssp370")
            & (df_conc["Variable"].str.endswith(species_rcmip_name))
            & (df_conc["Region"] == "World"),
            "1750":"2100",
        ]
        .interpolate(axis=1)
        .values.squeeze()
    )

emis_species_units_ok = ["CO", "VOC", "NOx"]
for species in emis_species_units_ok:
    input[species] = (
        df_emis.loc[
            (df_emis["Scenario"] == "ssp370")
            & (df_emis["Variable"].str.endswith(species))
            & (df_emis["Region"] == "World"),
            "1750":"2100",
        ]
        .interpolate(axis=1)
        .values.squeeze()
    )

# NOx emissions: scale up biomass burning
gfed_sectors = [
    "Emissions|NOx|MAGICC AFOLU|Agricultural Waste Burning",
    "Emissions|NOx|MAGICC AFOLU|Forest Burning",
    "Emissions|NOx|MAGICC AFOLU|Grassland Burning",
    "Emissions|NOx|MAGICC AFOLU|Peat Burning",
]
input["NOx"] = (
    df_emis.loc[
        (df_emis["Scenario"] == "ssp370")
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
        (df_emis["Scenario"] == "ssp370")
        & (df_emis["Region"] == "World")
        & (df_emis["Variable"] == "Emissions|NOx|MAGICC AFOLU|Agriculture"),
        "1750":"2100",
    ]
    .interpolate(axis=1)
    .values.squeeze()
    + df_emis.loc[
        (df_emis["Scenario"] == "ssp370")
        & (df_emis["Region"] == "World")
        & (df_emis["Variable"] == "Emissions|NOx|MAGICC Fossil and Industrial"),
        "1750":"2100",
    ]
    .interpolate(axis=1)
    .values.squeeze()
)
input["temp"] = gmst


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
total_eesc = 0

for species in hc_species:
    print(species, hc_input[species])
    hc_eesc[species] = calculate_eesc(
        hc_input[species],
        fractional_release[species],
        fractional_release["CFC-11"],
        cl_atoms[species],
        br_atoms[species],
    )
    total_eesc = total_eesc + hc_eesc[species]

total_eesc_1850 = total_eesc[100]

input["HC"] = total_eesc

# Use 1850 and 2014 emissions or concentrations corresponding to methane lifetime
# changes in Thornhill et al. 2021.
#
# Could we also take into account the fact that there are multiple loss pathways for
# CH4:
# - tropospheric OH loss is 560 Tg/yr
# - chlorine oxidation, 11 Tg/yr, assumed not included in AerChemMIP models
# - stratospheric loss is 31 Tg/yr, assumed not included in AerChemMIP models
# - soil uptake, 30 Tg/yr, not included in AerChemMIP models
#
# Saunois (2020): 90% of sink is OH chemistry in troposphere and is 553 [476–677] Tg
# CH4 yr−1, which is close to the IPCC number of 560, (chapter 5)
#
# Chapter 6 only give time constants for soil uptake and the combined chemistry loss
# (trop OH + chlorine + stratosphere).


def alpha_scaling_exp(
    input,
    baseline,
    normalisation,
    beta,
):
    log_lifetime_scaling = 0
    for species in ["CH4", "N2O", "VOC", "HC", "NOx", "temp"]:
        log_lifetime_scaling = log_lifetime_scaling + (
            np.log(
                1
                + (input[species] - baseline[species])
                / normalisation[species]
                * beta[species]
            )
        )
    return np.exp(log_lifetime_scaling)


# normalisation = ppb / ppt / Mt yr-1 increase from 1850 to 2014
normalisation = {}
for species in ["CH4", "N2O", "VOC", "NOx", "HC"]:
    normalisation[species] = input[species][264] - input[species][100]
    print(species, normalisation[species])
normalisation["temp"] = 1

# baselines are 1850! so "base" lifetime out is for 1750!
baseline = {}
for species in ["CH4", "N2O", "VOC", "NOx", "HC"]:
    baseline[species] = input[species][100]
baseline["temp"] = 0

# Steps 1 and 2

# Get and tune to AerChemMIP models
# MRI and GISS both give pretty good historical emulations
parameters = {}

parameters["AerChemMIP_mean"] = {
    "base": 10.0,
    "CH4": +0.22,
    "NOx": -0.33,
    "VOC": +0.19,
    "HC": -0.037,
    "N2O": -0.02,
    "temp": -0.006,
}

parameters["UKESM"] = {
    "base": 8,
    "CH4": +0.22,
    "NOx": -0.25,
    "VOC": +0.11,
    "HC": -0.049,
    "N2O": -0.012,
    "temp": -0.0408,
}

# we'll exclude BCC and CESM as they don't have VOC expt and that's important.
# We can live with a missing N2O from GFDL and a missing temperature feedback from MRI.

parameters["GFDL"] = {
    "base": 9.6,
    "CH4": +0.21,
    "NOx": -0.33,
    "VOC": +0.15,
    "HC": -0.075,
    "N2O": 0,  # missing
    "temp": -0.0205,
}

parameters["GISS"] = {
    "base": 13.4,
    "CH4": +0.18,
    "NOx": -0.46,
    "VOC": +0.27,
    "HC": -0.006,
    "N2O": -0.039,
    "temp": -0.0333,
}

parameters["MRI"] = {
    "base": 10.1,
    "CH4": +0.22,
    "NOx": -0.26,
    "VOC": +0.21,
    "HC": -0.024,
    "N2O": -0.013,
    "temp": 0,  # missing
}

lifetime_scaling = {}
models = ["UKESM", "GFDL", "GISS", "MRI"]


for model in models:
    print(parameters[model])
    lifetime_scaling[model] = alpha_scaling_exp(
        input,
        baseline,
        normalisation,
        parameters[model],
    )


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


emis_ch4 = (
    df_emis.loc[
        (df_emis["Scenario"] == "ssp370")
        & (df_emis["Variable"].str.endswith("CH4"))
        & (df_emis["Region"] == "World"),
        "1750":"2500",
    ]
    .interpolate(axis=1)
    .values.squeeze()
)

burden_per_emission = 1 / (5.1352e18 / 1e18 * 16.043 / 28.97)
partition_fraction = 1
pre_industrial_concentration = 729.2
natural_emissions_adjustment = emis_ch4[0]

conc_ch4 = {}


for model in models:
    conc_ch4[model] = np.zeros(351)
    gas_boxes = 0  # should use correct pi value for CMIP6
    airborne_emissions = 0
    for i in range(351):
        conc_ch4[model][i], gas_boxes, airborne_emissions = one_box(
            emis_ch4[i],
            gas_boxes,
            airborne_emissions,
            burden_per_emission,
            parameters[model]["base"],
            lifetime_scaling[model][i],
            partition_fraction,
            pre_industrial_concentration,
            timestep=1,
            natural_emissions_adjustment=natural_emissions_adjustment,
        )

if plots:
    for model in models:
        pl.plot(np.arange(1750, 2023), conc_ch4[model][:273], label=model)
    pl.plot(np.arange(1750, 2023), input["CH4"][:273], color="k", label="obs")
    pl.legend()
    os.makedirs(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/", exist_ok=True
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "aerchemmip_tuning_ch4_conc_1750-2023.png"
    )
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "aerchemmip_tuning_ch4_conc_1750-2023.pdf"
    )
    pl.close()

# Step 3

# Find least squares sensible historical fit using best estimate emissions and
# concentrations (not those from RCMIP)
df_emis_obs = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "scenario_subset_1750-2100.csv",
    index_col=0,
)
df_conc_obs = pd.read_csv(
    "../../../../../data/concentrations/ghg_concentrations_1750-2023.csv", index_col=0
)
for year in range(1751, 1850):
    df_conc_obs.loc[year, :] = np.nan
df_conc_obs.sort_index(inplace=True)
df_conc_obs.interpolate(inplace=True)

input_obs = {}
input_obs["CH4"] = df_conc_obs["CH4"].values[:272]
input_obs["N2O"] = df_conc_obs["N2O"].values[:272]
input_obs["VOC"] = df_emis_obs.loc[(df_emis_obs["variable"]=="VOC") & (df_emis_obs["scenario"]=='SSP2 - Medium Emissions'), '1750':'2021'].values.squeeze()
input_obs["NOx"] = df_emis_obs.loc[(df_emis_obs["variable"]=="NOx") & (df_emis_obs["scenario"]=='SSP2 - Medium Emissions'), '1750':'2021'].values.squeeze()
input_obs["temp"] = gmst[:272]

emis_ch4_obs = df_emis_obs.loc[
    (df_emis_obs["variable"] == "CH4") & (df_emis_obs["scenario"] == 'SSP2 - Medium Emissions'), "1750":"2021"
].values.squeeze()

print(emis_ch4_obs)

hc_eesc = {}
total_eesc = 0

for species in hc_species:
    hc_eesc[species] = calculate_eesc(
        df_conc_obs[species].values,
        fractional_release[species],
        fractional_release["CFC-11"],
        cl_atoms[species],
        br_atoms[species],
    )
    total_eesc = total_eesc + hc_eesc[species]

total_eesc_1850 = total_eesc[100]
input_obs["HC"] = total_eesc[:272]

if plots:
    pl.plot(input_obs["HC"])
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "eesc_from_observed_conc.png"
    )
    pl.close()

for key in input_obs:
    print(key, len(input_obs[key]))

invect = np.array(
    [
        input_obs["CH4"],
        input_obs["NOx"],
        input_obs["VOC"],
        input_obs["HC"],
        input_obs["N2O"],
        input_obs["temp"],
    ]
)

# normalisation = ppb / ppt / Mt yr-1 increase from 1850 to 2014
normalisation_obs = {}
for species in ["CH4", "N2O", "VOC", "NOx", "HC"]:
    normalisation_obs[species] = input_obs[species][264] - input_obs[species][100]
normalisation_obs["temp"] = 1

# baselines are 1850! so "base" lifetime out is for 1750!
baseline_obs = {}
for species in ["CH4", "N2O", "VOC", "NOx", "HC"]:
    baseline_obs[species] = input_obs[species][100]
baseline_obs["temp"] = 0
natural_emissions_adjustment = emis_ch4_obs[0]


# def fit_precursors(x, rch4, rnox, rvoc, rhc, rn2o, rtemp, rbase, rnat):
def fit_precursors(x, rch4, rnox, rvoc, rhc, rn2o, rtemp, rbase):
    conc_ch4 = np.zeros(272)  # 1750-2022 timebounds
    gas_boxes = 0  # should use correct pi value for CMIP6
    airborne_emissions = 0

    params = {}
    params["CH4"] = rch4
    params["NOx"] = rnox
    params["VOC"] = rvoc
    params["HC"] = rhc
    params["N2O"] = rn2o
    params["temp"] = rtemp

    inp = {}
    inp["CH4"] = x[0]
    inp["NOx"] = x[1]
    inp["VOC"] = x[2]
    inp["HC"] = x[3]
    inp["N2O"] = x[4]
    inp["temp"] = x[5]

    lifetime_scaling = alpha_scaling_exp(
        inp,
        baseline_obs,
        normalisation_obs,
        params,
    )

    for i in range(272):
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
high = np.array([0.26, -0.25, 0.27, -0.006, -0.012, 0, 13.4])
gap = (high - low) / 2

# override temperature
gap[5] = 0

# override all this time
gap[:] = 0

# natural bounds from global methane budget (part of GCP)
p, cov = scipy.optimize.curve_fit(
    fit_precursors, invect, input_obs["CH4"][:272], bounds=(low - gap, high + gap)
)

parameters["best_fit"] = {
    "base": p[6],
    "CH4": p[0],
    "NOx": p[1],
    "VOC": p[2],
    "HC": p[3],
    "N2O": p[4],
    "temp": p[5],
}

# these are the feedback values per ppb / per Mt that go into FaIR
print(parameters["best_fit"])
for specie in ["CH4", "NOx", "VOC", "HC", "N2O"]:
    print(specie, parameters["best_fit"][specie] / normalisation_obs[specie])

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
    pl.plot(
        np.arange(1750, 2022),
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

conc_ch4["best_fit"] = np.zeros(272)
gas_boxes = 0
airborne_emissions = 0

for i in range(272):
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


# ### Compare the SSP3-7.0 fit to other SSPs

emis_ch4_ssps = {}
emis_nox_ssps = {}
emis_voc_ssps = {}
conc_n2o_ssps = {}
conc_eesc_ssps = {}

ar7_colors = {
    "SSP1 - Very Low Emissions": "#00a9cf",
    "SSP1 - Low Emissions": "#003466",
    "SSP2 - Medium Emissions": "#f69320",
    "SSP3 - High Emissions": "#df0000",
    "SSP5 - High Emissions": "#980002",
}

scenarios = [
    "SSP3 - High Emissions",
    "SSP1 - Low Emissions",
    "SSP1 - Very Low Emissions",
    "SSP2 - Medium Emissions",
    "SSP5 - High Emissions",
]

df_emis = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "scenario_subset_1750-2100.csv",
    index_col=[0, 1, 2, 3, 4]
)

conc_n2o_ssps = {'SSP3 - High Emissions': np.array([270.1       , 270.12010276, 270.14020459, 270.16030548,
       270.18040545, 270.20050451, 270.22060267, 270.24069992,
       270.26079629, 270.28089178, 270.3009864 , 270.32108015,
       270.34117305, 270.3612651 , 270.38135631, 270.40144669,
       270.42153624, 270.44162497, 270.46171289, 270.48180001,
       270.50188633, 270.52197187, 270.54205662, 270.5621406 ,
       270.58222382, 270.60230627, 270.62238797, 270.64246892,
       270.66254914, 270.68262862, 270.70270738, 270.72278542,
       270.74286274, 270.76293936, 270.78301528, 270.8030905 ,
       270.82316504, 270.8432389 , 270.86331208, 270.8833846 ,
       270.90345645, 270.92352765, 270.94359819, 270.96366809,
       270.98373736, 271.00380599, 271.02387399, 271.04394138,
       271.06400814, 271.0840743 , 271.10413986, 271.12420481,
       271.14426917, 271.16433295, 271.18439614, 271.20445875,
       271.2245208 , 271.24458227, 271.26464319, 271.28470355,
       271.30476335, 271.32482262, 271.34488134, 271.36493952,
       271.38499717, 271.4050543 , 271.4251109 , 271.44516699,
       271.46522257, 271.48527763, 271.5053322 , 271.52538626,
       271.54543984, 271.56549292, 271.58554552, 271.60559764,
       271.62564928, 271.64570045, 271.66575115, 271.68580139,
       271.70585118, 271.7259005 , 271.74594938, 271.76599781,
       271.7860458 , 271.80609335, 271.82614046, 271.84618715,
       271.86623341, 271.88627924, 271.90632466, 271.92636966,
       271.94641425, 271.96645844, 271.98650222, 272.0065456 ,
       272.02658859, 272.04663118, 272.06667338, 272.0867152 ,
       272.13732269, 272.21847082, 272.31057988, 272.41518552,
       272.52531447, 272.63992914, 272.74953857, 272.84565887,
       272.9627749 , 273.11989913, 273.29951447, 273.48662153,
       273.65923239, 273.81835415, 273.97996276, 274.15106199,
       274.3246786 , 274.50329977, 274.68440779, 274.85602453,
       275.03414119, 275.22023788, 275.40085636, 275.57697318,
       275.75157908, 275.91818907, 276.06479449, 276.2054029 ,
       276.3470181 , 276.48562702, 276.61323407, 276.73084632,
       276.84644413, 276.95804385, 277.06715611, 277.16926431,
       277.27637712, 277.38098722, 277.48608532, 277.5896878 ,
       277.67479636, 277.7664049 , 277.87701038, 278.00761987,
       278.1532285 , 278.28533284, 278.40243754, 278.5215409 ,
       278.65713876, 278.80973467, 278.96933413, 279.17994347,
       279.45854574, 279.75114009, 280.03274788, 280.31385189,
       280.60644555, 280.92905302, 281.27315989, 281.6317624 ,
       282.02436126, 282.38395515, 282.71155114, 283.06714679,
       283.41676079, 283.72186469, 284.01296081, 284.26656588,
       284.39166122, 284.4607566 , 284.56186473, 284.68946191,
       284.9105495 , 285.13514641, 285.23474313, 285.28534368,
       285.33644758, 285.41404056, 285.49663823, 285.55624072,
       285.67983838, 285.82893331, 285.95951914, 286.08911297,
       286.20421077, 286.35130609, 286.54040867, 286.76950437,
       286.99460209, 287.21570185, 287.46129179, 287.72789292,
       288.19348375, 288.63157808, 288.8616797 , 289.08225967,
       289.23135206, 289.30296016, 289.35355676, 289.45214209,
       289.59023227, 289.74033097, 289.98442366, 290.25300656,
       290.51510432, 290.84170084, 291.18279307, 291.48238124,
       291.7269766 , 291.98957719, 292.30416434, 292.64224988,
       293.01184468, 293.38894668, 293.75503169, 294.14611047,
       294.56070291, 294.97579875, 295.39338166, 295.81497749,
       296.2425699 , 296.6781578 , 297.12375533, 297.58084261,
       298.05143379, 298.53803132, 299.04161177, 299.56418928,
       300.10828157, 300.72385532, 301.48263841, 302.48915086,
       303.36366321, 303.99692544, 304.61993756, 305.35369957,
       305.90996146, 306.36772326, 307.25348494, 308.24624651,
       309.06400798, 309.68151935, 310.10453061, 310.53779177,
       311.11330283, 311.84406378, 312.69307464, 313.40267801,
       314.12072648, 314.78267618, 315.4163347 , 316.04986294,
       316.71828598, 317.39811545, 318.09577208, 318.79450802,
       319.49673375, 320.22171889, 320.92126269, 321.61337434,
       322.33858872, 323.06111471, 323.808466  , 324.55661062,
       325.33057867, 326.15465774, 326.94935179, 327.7229614 ,
       328.50309411, 329.29936877, 330.07418948, 330.87599736,
       331.68410546, 332.49833838, 333.31852235, 334.1444852 ,
       334.97731774, 335.81683116, 336.66283841, 337.51515412,
       338.37359462, 339.23735448, 340.1062632 , 340.98015181,
       341.85885293, 342.74220063, 343.62919619, 344.51968996,
       345.41353369, 346.31058048, 347.21068478, 348.11154202,
       349.01304383, 349.91508281, 350.81755256, 351.72034763,
       352.62259886, 353.52421504, 354.42510579, 355.32518155,
       356.22435358, 357.12258594, 358.01979064, 358.91588053,
       359.81076925, 360.70437122, 361.59635021, 362.48662632,
       363.37512036, 364.26175391, 365.1464492 , 366.02869225,
       366.90841395, 367.78554581, 368.66001997, 369.53176918,
       370.40109791, 371.26793397, 372.13220583, 372.9938426 ,
       373.85277406, 374.70828548, 375.5603187 , 376.40881612,
       377.25372068, 378.09497581, 378.9325846 , 379.76649053,
       380.5966376 , 381.42297028, 382.24543358, 383.07180117,
       383.90203741, 384.73610695, 385.5739748 , 386.41560625,
       387.25972663, 388.10631319, 388.95534342, 389.80679501,
       390.66064584, 391.51788641, 392.37848578, 393.24241328,
       394.10963849, 394.98013134, 395.85301764, 396.72827556,
       397.60588343, 398.48581979, 399.36806338]), 'SSP1 - Low Emissions': np.array([270.1       , 270.12010276, 270.14020459, 270.16030548,
       270.18040545, 270.20050451, 270.22060267, 270.24069992,
       270.26079629, 270.28089178, 270.3009864 , 270.32108015,
       270.34117305, 270.3612651 , 270.38135631, 270.40144669,
       270.42153624, 270.44162497, 270.46171289, 270.48180001,
       270.50188633, 270.52197187, 270.54205662, 270.5621406 ,
       270.58222382, 270.60230627, 270.62238797, 270.64246892,
       270.66254914, 270.68262862, 270.70270738, 270.72278542,
       270.74286274, 270.76293936, 270.78301528, 270.8030905 ,
       270.82316504, 270.8432389 , 270.86331208, 270.8833846 ,
       270.90345645, 270.92352765, 270.94359819, 270.96366809,
       270.98373736, 271.00380599, 271.02387399, 271.04394138,
       271.06400814, 271.0840743 , 271.10413986, 271.12420481,
       271.14426917, 271.16433295, 271.18439614, 271.20445875,
       271.2245208 , 271.24458227, 271.26464319, 271.28470355,
       271.30476335, 271.32482262, 271.34488134, 271.36493952,
       271.38499717, 271.4050543 , 271.4251109 , 271.44516699,
       271.46522257, 271.48527763, 271.5053322 , 271.52538626,
       271.54543984, 271.56549292, 271.58554552, 271.60559764,
       271.62564928, 271.64570045, 271.66575115, 271.68580139,
       271.70585118, 271.7259005 , 271.74594938, 271.76599781,
       271.7860458 , 271.80609335, 271.82614046, 271.84618715,
       271.86623341, 271.88627924, 271.90632466, 271.92636966,
       271.94641425, 271.96645844, 271.98650222, 272.0065456 ,
       272.02658859, 272.04663118, 272.06667338, 272.0867152 ,
       272.13732269, 272.21847082, 272.31057988, 272.41518552,
       272.52531447, 272.63992914, 272.74953857, 272.84565887,
       272.9627749 , 273.11989913, 273.29951447, 273.48662153,
       273.65923239, 273.81835415, 273.97996276, 274.15106199,
       274.3246786 , 274.50329977, 274.68440779, 274.85602453,
       275.03414119, 275.22023788, 275.40085636, 275.57697318,
       275.75157908, 275.91818907, 276.06479449, 276.2054029 ,
       276.3470181 , 276.48562702, 276.61323407, 276.73084632,
       276.84644413, 276.95804385, 277.06715611, 277.16926431,
       277.27637712, 277.38098722, 277.48608532, 277.5896878 ,
       277.67479636, 277.7664049 , 277.87701038, 278.00761987,
       278.1532285 , 278.28533284, 278.40243754, 278.5215409 ,
       278.65713876, 278.80973467, 278.96933413, 279.17994347,
       279.45854574, 279.75114009, 280.03274788, 280.31385189,
       280.60644555, 280.92905302, 281.27315989, 281.6317624 ,
       282.02436126, 282.38395515, 282.71155114, 283.06714679,
       283.41676079, 283.72186469, 284.01296081, 284.26656588,
       284.39166122, 284.4607566 , 284.56186473, 284.68946191,
       284.9105495 , 285.13514641, 285.23474313, 285.28534368,
       285.33644758, 285.41404056, 285.49663823, 285.55624072,
       285.67983838, 285.82893331, 285.95951914, 286.08911297,
       286.20421077, 286.35130609, 286.54040867, 286.76950437,
       286.99460209, 287.21570185, 287.46129179, 287.72789292,
       288.19348375, 288.63157808, 288.8616797 , 289.08225967,
       289.23135206, 289.30296016, 289.35355676, 289.45214209,
       289.59023227, 289.74033097, 289.98442366, 290.25300656,
       290.51510432, 290.84170084, 291.18279307, 291.48238124,
       291.7269766 , 291.98957719, 292.30416434, 292.64224988,
       293.01184468, 293.38894668, 293.75503169, 294.14611047,
       294.56070291, 294.97579875, 295.39338166, 295.81497749,
       296.2425699 , 296.6781578 , 297.12375533, 297.58084261,
       298.05143379, 298.53803132, 299.04161177, 299.56418928,
       300.10828157, 300.72385532, 301.48263841, 302.48915086,
       303.36366321, 303.99692544, 304.61993756, 305.35369957,
       305.90996146, 306.36772326, 307.25348494, 308.24624651,
       309.06400798, 309.68151935, 310.10453061, 310.53779177,
       311.11330283, 311.84406378, 312.69307464, 313.40267801,
       314.12072648, 314.78267618, 315.4163347 , 316.04986294,
       316.71828598, 317.39811545, 318.09577208, 318.79450802,
       319.49673375, 320.22171889, 320.92126269, 321.61337434,
       322.33858872, 323.06111471, 323.808466  , 324.55661062,
       325.33057867, 326.15465774, 326.94935179, 327.7229614 ,
       328.50309411, 329.29936877, 330.07418948, 330.87599736,
       331.66880571, 332.45269669, 333.22775176, 333.9940516 ,
       334.72753863, 335.4285125 , 336.09727015, 336.73410577,
       337.3393109 , 337.92894976, 338.50316452, 339.06209602,
       339.60588384, 340.13466627, 340.65194606, 341.15782823,
       341.65241688, 342.13581514, 342.60812521, 343.07547354,
       343.53790543, 343.99546579, 344.4481991 , 344.89614945,
       345.33816506, 345.77430011, 346.20460833, 346.62914291,
       347.04795658, 347.45904056, 347.86246543, 348.25830113,
       348.64661698, 349.02748164, 349.40096317, 349.76712899,
       350.1260459 , 350.47778012, 350.82239723, 351.16378931,
       351.50198581, 351.83701592, 352.16890856, 352.49769237,
       352.82339574, 353.14604681, 353.46567346, 353.7823033 ,
       354.0959637 , 354.40385012, 354.70601529, 355.00251146,
       355.29339039, 355.57870339, 355.85850129, 356.13283446,
       356.40175279, 356.66530574, 356.92354231, 357.17248471,
       357.41221781, 357.64282571, 357.86439175, 358.0769985 ,
       358.28072779, 358.47566068, 358.66187751, 358.83945787,
       359.00848064, 359.1751946 , 359.33962085, 359.50178026,
       359.66169356, 359.81938123, 359.97486362, 360.12816086,
       360.27929291, 360.42827954, 360.57514034]), 'SSP1 - Very Low Emissions': np.array([270.1       , 270.12010276, 270.14020459, 270.16030548,
       270.18040545, 270.20050451, 270.22060267, 270.24069992,
       270.26079629, 270.28089178, 270.3009864 , 270.32108015,
       270.34117305, 270.3612651 , 270.38135631, 270.40144669,
       270.42153624, 270.44162497, 270.46171289, 270.48180001,
       270.50188633, 270.52197187, 270.54205662, 270.5621406 ,
       270.58222382, 270.60230627, 270.62238797, 270.64246892,
       270.66254914, 270.68262862, 270.70270738, 270.72278542,
       270.74286274, 270.76293936, 270.78301528, 270.8030905 ,
       270.82316504, 270.8432389 , 270.86331208, 270.8833846 ,
       270.90345645, 270.92352765, 270.94359819, 270.96366809,
       270.98373736, 271.00380599, 271.02387399, 271.04394138,
       271.06400814, 271.0840743 , 271.10413986, 271.12420481,
       271.14426917, 271.16433295, 271.18439614, 271.20445875,
       271.2245208 , 271.24458227, 271.26464319, 271.28470355,
       271.30476335, 271.32482262, 271.34488134, 271.36493952,
       271.38499717, 271.4050543 , 271.4251109 , 271.44516699,
       271.46522257, 271.48527763, 271.5053322 , 271.52538626,
       271.54543984, 271.56549292, 271.58554552, 271.60559764,
       271.62564928, 271.64570045, 271.66575115, 271.68580139,
       271.70585118, 271.7259005 , 271.74594938, 271.76599781,
       271.7860458 , 271.80609335, 271.82614046, 271.84618715,
       271.86623341, 271.88627924, 271.90632466, 271.92636966,
       271.94641425, 271.96645844, 271.98650222, 272.0065456 ,
       272.02658859, 272.04663118, 272.06667338, 272.0867152 ,
       272.13732269, 272.21847082, 272.31057988, 272.41518552,
       272.52531447, 272.63992914, 272.74953857, 272.84565887,
       272.9627749 , 273.11989913, 273.29951447, 273.48662153,
       273.65923239, 273.81835415, 273.97996276, 274.15106199,
       274.3246786 , 274.50329977, 274.68440779, 274.85602453,
       275.03414119, 275.22023788, 275.40085636, 275.57697318,
       275.75157908, 275.91818907, 276.06479449, 276.2054029 ,
       276.3470181 , 276.48562702, 276.61323407, 276.73084632,
       276.84644413, 276.95804385, 277.06715611, 277.16926431,
       277.27637712, 277.38098722, 277.48608532, 277.5896878 ,
       277.67479636, 277.7664049 , 277.87701038, 278.00761987,
       278.1532285 , 278.28533284, 278.40243754, 278.5215409 ,
       278.65713876, 278.80973467, 278.96933413, 279.17994347,
       279.45854574, 279.75114009, 280.03274788, 280.31385189,
       280.60644555, 280.92905302, 281.27315989, 281.6317624 ,
       282.02436126, 282.38395515, 282.71155114, 283.06714679,
       283.41676079, 283.72186469, 284.01296081, 284.26656588,
       284.39166122, 284.4607566 , 284.56186473, 284.68946191,
       284.9105495 , 285.13514641, 285.23474313, 285.28534368,
       285.33644758, 285.41404056, 285.49663823, 285.55624072,
       285.67983838, 285.82893331, 285.95951914, 286.08911297,
       286.20421077, 286.35130609, 286.54040867, 286.76950437,
       286.99460209, 287.21570185, 287.46129179, 287.72789292,
       288.19348375, 288.63157808, 288.8616797 , 289.08225967,
       289.23135206, 289.30296016, 289.35355676, 289.45214209,
       289.59023227, 289.74033097, 289.98442366, 290.25300656,
       290.51510432, 290.84170084, 291.18279307, 291.48238124,
       291.7269766 , 291.98957719, 292.30416434, 292.64224988,
       293.01184468, 293.38894668, 293.75503169, 294.14611047,
       294.56070291, 294.97579875, 295.39338166, 295.81497749,
       296.2425699 , 296.6781578 , 297.12375533, 297.58084261,
       298.05143379, 298.53803132, 299.04161177, 299.56418928,
       300.10828157, 300.72385532, 301.48263841, 302.48915086,
       303.36366321, 303.99692544, 304.61993756, 305.35369957,
       305.90996146, 306.36772326, 307.25348494, 308.24624651,
       309.06400798, 309.68151935, 310.10453061, 310.53779177,
       311.11330283, 311.84406378, 312.69307464, 313.40267801,
       314.12072648, 314.78267618, 315.4163347 , 316.04986294,
       316.71828598, 317.39811545, 318.09577208, 318.79450802,
       319.49673375, 320.22171889, 320.92126269, 321.61337434,
       322.33858872, 323.06111471, 323.808466  , 324.55661062,
       325.33057867, 326.15465774, 326.94935179, 327.7229614 ,
       328.50309411, 329.29936877, 330.07418948, 330.87599736,
       331.67485692, 332.47079506, 333.26383848, 334.0540136 ,
       334.79288578, 335.48092353, 336.11859109, 336.70634847,
       337.24465145, 337.75285224, 338.23122575, 338.68004437,
       339.09957801, 339.4900941 , 339.86203299, 340.21556434,
       340.55085624, 340.86807526, 341.16738647, 341.45552017,
       341.73257845, 341.99866244, 342.25387238, 342.49830757,
       342.7323856 , 342.95620106, 343.16984768, 343.37341832,
       343.56700499, 343.75566017, 343.9394289 , 344.1183558 ,
       344.29248509, 344.46186058, 344.62652568, 344.78652342,
       344.94189642, 345.09268691, 345.23893674, 345.38135582,
       345.51997912, 345.65484132, 345.78597676, 345.91341948,
       346.0372032 , 346.15736133, 346.27392699, 346.38693298,
       346.49641181, 346.60244054, 346.70505066, 346.80427341,
       346.90013972, 346.99268023, 347.08192533, 347.1679051 ,
       347.25064937, 347.33018769, 347.40654932, 347.47804419,
       347.54471675, 347.60661103, 347.66377068, 347.71623892,
       347.7640586 , 347.80727217, 347.84592171, 347.88004888,
       347.909695  , 347.93458018, 347.95474791, 347.97024126,
       347.98110293, 347.98737521, 347.98910002, 347.98631888,
       347.97907294, 347.96740298, 347.95134941]), 'SSP2 - Medium Emissions': np.array([270.1       , 270.12010276, 270.14020459, 270.16030548,
       270.18040545, 270.20050451, 270.22060267, 270.24069992,
       270.26079629, 270.28089178, 270.3009864 , 270.32108015,
       270.34117305, 270.3612651 , 270.38135631, 270.40144669,
       270.42153624, 270.44162497, 270.46171289, 270.48180001,
       270.50188633, 270.52197187, 270.54205662, 270.5621406 ,
       270.58222382, 270.60230627, 270.62238797, 270.64246892,
       270.66254914, 270.68262862, 270.70270738, 270.72278542,
       270.74286274, 270.76293936, 270.78301528, 270.8030905 ,
       270.82316504, 270.8432389 , 270.86331208, 270.8833846 ,
       270.90345645, 270.92352765, 270.94359819, 270.96366809,
       270.98373736, 271.00380599, 271.02387399, 271.04394138,
       271.06400814, 271.0840743 , 271.10413986, 271.12420481,
       271.14426917, 271.16433295, 271.18439614, 271.20445875,
       271.2245208 , 271.24458227, 271.26464319, 271.28470355,
       271.30476335, 271.32482262, 271.34488134, 271.36493952,
       271.38499717, 271.4050543 , 271.4251109 , 271.44516699,
       271.46522257, 271.48527763, 271.5053322 , 271.52538626,
       271.54543984, 271.56549292, 271.58554552, 271.60559764,
       271.62564928, 271.64570045, 271.66575115, 271.68580139,
       271.70585118, 271.7259005 , 271.74594938, 271.76599781,
       271.7860458 , 271.80609335, 271.82614046, 271.84618715,
       271.86623341, 271.88627924, 271.90632466, 271.92636966,
       271.94641425, 271.96645844, 271.98650222, 272.0065456 ,
       272.02658859, 272.04663118, 272.06667338, 272.0867152 ,
       272.13732269, 272.21847082, 272.31057988, 272.41518552,
       272.52531447, 272.63992914, 272.74953857, 272.84565887,
       272.9627749 , 273.11989913, 273.29951447, 273.48662153,
       273.65923239, 273.81835415, 273.97996276, 274.15106199,
       274.3246786 , 274.50329977, 274.68440779, 274.85602453,
       275.03414119, 275.22023788, 275.40085636, 275.57697318,
       275.75157908, 275.91818907, 276.06479449, 276.2054029 ,
       276.3470181 , 276.48562702, 276.61323407, 276.73084632,
       276.84644413, 276.95804385, 277.06715611, 277.16926431,
       277.27637712, 277.38098722, 277.48608532, 277.5896878 ,
       277.67479636, 277.7664049 , 277.87701038, 278.00761987,
       278.1532285 , 278.28533284, 278.40243754, 278.5215409 ,
       278.65713876, 278.80973467, 278.96933413, 279.17994347,
       279.45854574, 279.75114009, 280.03274788, 280.31385189,
       280.60644555, 280.92905302, 281.27315989, 281.6317624 ,
       282.02436126, 282.38395515, 282.71155114, 283.06714679,
       283.41676079, 283.72186469, 284.01296081, 284.26656588,
       284.39166122, 284.4607566 , 284.56186473, 284.68946191,
       284.9105495 , 285.13514641, 285.23474313, 285.28534368,
       285.33644758, 285.41404056, 285.49663823, 285.55624072,
       285.67983838, 285.82893331, 285.95951914, 286.08911297,
       286.20421077, 286.35130609, 286.54040867, 286.76950437,
       286.99460209, 287.21570185, 287.46129179, 287.72789292,
       288.19348375, 288.63157808, 288.8616797 , 289.08225967,
       289.23135206, 289.30296016, 289.35355676, 289.45214209,
       289.59023227, 289.74033097, 289.98442366, 290.25300656,
       290.51510432, 290.84170084, 291.18279307, 291.48238124,
       291.7269766 , 291.98957719, 292.30416434, 292.64224988,
       293.01184468, 293.38894668, 293.75503169, 294.14611047,
       294.56070291, 294.97579875, 295.39338166, 295.81497749,
       296.2425699 , 296.6781578 , 297.12375533, 297.58084261,
       298.05143379, 298.53803132, 299.04161177, 299.56418928,
       300.10828157, 300.72385532, 301.48263841, 302.48915086,
       303.36366321, 303.99692544, 304.61993756, 305.35369957,
       305.90996146, 306.36772326, 307.25348494, 308.24624651,
       309.06400798, 309.68151935, 310.10453061, 310.53779177,
       311.11330283, 311.84406378, 312.69307464, 313.40267801,
       314.12072648, 314.78267618, 315.4163347 , 316.04986294,
       316.71828598, 317.39811545, 318.09577208, 318.79450802,
       319.49673375, 320.22171889, 320.92126269, 321.61337434,
       322.33858872, 323.06111471, 323.808466  , 324.55661062,
       325.33057867, 326.15465774, 326.94935179, 327.7229614 ,
       328.50309411, 329.29936877, 330.07418948, 330.87599736,
       331.70007487, 332.54621862, 333.41422709, 334.30390062,
       335.1919027 , 336.0782486 , 336.96295344, 337.84603221,
       338.72749976, 339.59656399, 340.45333816, 341.29793453,
       342.1304643 , 342.95103766, 343.76479054, 344.57178522,
       345.37208343, 346.16574631, 346.95283447, 347.73128743,
       348.50118407, 349.2626025 , 350.01562017, 350.7603138 ,
       351.49644214, 352.22408344, 352.94331519, 353.6542142 ,
       354.35685655, 355.0633076 , 355.77353255, 356.48749694,
       357.20516662, 357.92650776, 358.63760274, 359.33854514,
       360.02942767, 360.7103422 , 361.38137976, 362.04167257,
       362.69131873, 363.3304155 , 363.95905919, 364.57734529,
       365.18536837, 365.78322216, 366.37099952, 366.9487925 ,
       367.51669225, 368.07578294, 368.62614502, 369.1678582 ,
       369.70100146, 370.22565307, 370.74189058, 371.24979082,
       371.74942994, 372.24088338, 372.72422589, 373.19972278,
       373.66744568, 374.12746559, 374.57985287, 375.02467721,
       375.46200768, 375.89191272, 376.31446014, 376.72971713,
       377.13775027, 377.52831364, 377.90156677, 378.25766775,
       378.59677322, 378.91903838, 379.22461704, 379.51366157,
       379.78632297, 380.04275086, 380.28309349]), 'SSP5 - High Emissions': np.array([270.1       , 270.12010276, 270.14020459, 270.16030548,
       270.18040545, 270.20050451, 270.22060267, 270.24069992,
       270.26079629, 270.28089178, 270.3009864 , 270.32108015,
       270.34117305, 270.3612651 , 270.38135631, 270.40144669,
       270.42153624, 270.44162497, 270.46171289, 270.48180001,
       270.50188633, 270.52197187, 270.54205662, 270.5621406 ,
       270.58222382, 270.60230627, 270.62238797, 270.64246892,
       270.66254914, 270.68262862, 270.70270738, 270.72278542,
       270.74286274, 270.76293936, 270.78301528, 270.8030905 ,
       270.82316504, 270.8432389 , 270.86331208, 270.8833846 ,
       270.90345645, 270.92352765, 270.94359819, 270.96366809,
       270.98373736, 271.00380599, 271.02387399, 271.04394138,
       271.06400814, 271.0840743 , 271.10413986, 271.12420481,
       271.14426917, 271.16433295, 271.18439614, 271.20445875,
       271.2245208 , 271.24458227, 271.26464319, 271.28470355,
       271.30476335, 271.32482262, 271.34488134, 271.36493952,
       271.38499717, 271.4050543 , 271.4251109 , 271.44516699,
       271.46522257, 271.48527763, 271.5053322 , 271.52538626,
       271.54543984, 271.56549292, 271.58554552, 271.60559764,
       271.62564928, 271.64570045, 271.66575115, 271.68580139,
       271.70585118, 271.7259005 , 271.74594938, 271.76599781,
       271.7860458 , 271.80609335, 271.82614046, 271.84618715,
       271.86623341, 271.88627924, 271.90632466, 271.92636966,
       271.94641425, 271.96645844, 271.98650222, 272.0065456 ,
       272.02658859, 272.04663118, 272.06667338, 272.0867152 ,
       272.13732269, 272.21847082, 272.31057988, 272.41518552,
       272.52531447, 272.63992914, 272.74953857, 272.84565887,
       272.9627749 , 273.11989913, 273.29951447, 273.48662153,
       273.65923239, 273.81835415, 273.97996276, 274.15106199,
       274.3246786 , 274.50329977, 274.68440779, 274.85602453,
       275.03414119, 275.22023788, 275.40085636, 275.57697318,
       275.75157908, 275.91818907, 276.06479449, 276.2054029 ,
       276.3470181 , 276.48562702, 276.61323407, 276.73084632,
       276.84644413, 276.95804385, 277.06715611, 277.16926431,
       277.27637712, 277.38098722, 277.48608532, 277.5896878 ,
       277.67479636, 277.7664049 , 277.87701038, 278.00761987,
       278.1532285 , 278.28533284, 278.40243754, 278.5215409 ,
       278.65713876, 278.80973467, 278.96933413, 279.17994347,
       279.45854574, 279.75114009, 280.03274788, 280.31385189,
       280.60644555, 280.92905302, 281.27315989, 281.6317624 ,
       282.02436126, 282.38395515, 282.71155114, 283.06714679,
       283.41676079, 283.72186469, 284.01296081, 284.26656588,
       284.39166122, 284.4607566 , 284.56186473, 284.68946191,
       284.9105495 , 285.13514641, 285.23474313, 285.28534368,
       285.33644758, 285.41404056, 285.49663823, 285.55624072,
       285.67983838, 285.82893331, 285.95951914, 286.08911297,
       286.20421077, 286.35130609, 286.54040867, 286.76950437,
       286.99460209, 287.21570185, 287.46129179, 287.72789292,
       288.19348375, 288.63157808, 288.8616797 , 289.08225967,
       289.23135206, 289.30296016, 289.35355676, 289.45214209,
       289.59023227, 289.74033097, 289.98442366, 290.25300656,
       290.51510432, 290.84170084, 291.18279307, 291.48238124,
       291.7269766 , 291.98957719, 292.30416434, 292.64224988,
       293.01184468, 293.38894668, 293.75503169, 294.14611047,
       294.56070291, 294.97579875, 295.39338166, 295.81497749,
       296.2425699 , 296.6781578 , 297.12375533, 297.58084261,
       298.05143379, 298.53803132, 299.04161177, 299.56418928,
       300.10828157, 300.72385532, 301.48263841, 302.48915086,
       303.36366321, 303.99692544, 304.61993756, 305.35369957,
       305.90996146, 306.36772326, 307.25348494, 308.24624651,
       309.06400798, 309.68151935, 310.10453061, 310.53779177,
       311.11330283, 311.84406378, 312.69307464, 313.40267801,
       314.12072648, 314.78267618, 315.4163347 , 316.04986294,
       316.71828598, 317.39811545, 318.09577208, 318.79450802,
       319.49673375, 320.22171889, 320.92126269, 321.61337434,
       322.33858872, 323.06111471, 323.808466  , 324.55661062,
       325.33057867, 326.15465774, 326.94935179, 327.7229614 ,
       328.50309411, 329.29936877, 330.07418948, 330.87599736,
       331.6815506 , 332.49081498, 333.30375661, 334.12034192,
       334.95102101, 335.79566518, 336.65414689, 337.52633978,
       338.41211863, 339.29679611, 340.18038229, 341.06288712,
       341.94432048, 342.82469217, 343.7036862 , 344.58131516,
       345.45759152, 346.33252763, 347.20613573, 348.07618572,
       348.9427101 , 349.80574106, 350.6653105 , 351.52145005,
       352.37331489, 353.22094406, 354.06437626, 354.9036498 ,
       355.73880266, 356.56644627, 357.3866492 , 358.1994794 ,
       359.00500419, 359.80329031, 360.59305115, 361.37436459,
       362.14730776, 362.91195711, 363.66838838, 364.41501451,
       365.15192505, 365.87920873, 366.59695346, 367.30524636,
       368.00358361, 368.69205612, 369.37075399, 370.03976648,
       370.69918205, 371.34819   , 371.9868854 , 372.61536242,
       373.23371437, 373.84203373, 374.44173022, 375.03288259,
       375.61556887, 376.18986638, 376.75585172, 377.31630071,
       377.87126392, 378.42079144, 378.96493291, 379.50373752,
       380.03727978, 380.56560774, 381.08876902, 381.60681081,
       382.11977985, 382.62686258, 383.12811275, 383.62358362,
       384.11332797, 384.59739811, 385.07523394, 385.5468924 ,
       386.01242991, 386.47190235, 386.92536513])}


eesc_noddy = np.zeros(351)
eesc_noddy[:272] = input_obs["HC"]
eesc_noddy[272:] = eesc_noddy[271] * np.exp(-np.arange(79)/80)

for ssp in scenarios:
    emis_ch4_ssps[ssp] = df_emis.query(f"scenario=='{ssp}' and variable=='CH4'").values.squeeze()
    emis_nox_ssps[ssp] = df_emis.query(f"scenario=='{ssp}' and variable=='NOx'").values.squeeze()
    emis_voc_ssps[ssp] = df_emis.query(f"scenario=='{ssp}' and variable=='VOC'").values.squeeze()
    conc_eesc_ssps[ssp] = eesc_noddy

    conc_ch4[ssp] = np.zeros(351)
    conc_ch4[ssp][0] = 729.2
    gas_boxes = 0
    airborne_emissions = 0
    norm = {}
    norm["CH4"] = normalisation_obs["CH4"]
    norm["N2O"] = conc_n2o_ssps[ssp][264] - conc_n2o_ssps[ssp][100]
    norm["VOC"] = normalisation_obs["VOC"]
    norm["NOx"] = normalisation_obs["NOx"]
    norm["HC"] = conc_eesc_ssps[ssp][264] - conc_eesc_ssps[ssp][100]
    norm["temp"] = 1

    bl = {}
    bl["CH4"] = baseline_obs["CH4"]
    bl["VOC"] = baseline_obs["VOC"]
    bl["NOx"] = baseline_obs["NOx"]
    bl["N2O"] = conc_n2o_ssps[ssp][100]
    bl["HC"] = conc_eesc_ssps[ssp][100]
    bl["temp"] = 0

    for i in range(351):
        inp = {}
        inp["CH4"] = conc_ch4[ssp][i - 1]
        inp["N2O"] = conc_n2o_ssps[ssp][i]
        inp["HC"] = conc_eesc_ssps[ssp][i]
        inp["NOx"] = emis_nox_ssps[ssp][i]
        inp["VOC"] = emis_voc_ssps[ssp][i]
        inp["temp"] = gmst[i]

        ls = alpha_scaling_exp(
            inp,
            bl,
            norm,
            parameters["best_fit"],
        )
        conc_ch4[ssp][i], gas_boxes, airborne_emissions = one_box(
            #            emis_ch4_ssps[ssp][i]+parameters["best_fit"]["nat"],
            emis_ch4_ssps[ssp][i],
            gas_boxes,
            airborne_emissions,
            burden_per_emission,
            parameters["best_fit"]["base"],
            ls,
            partition_fraction,
            pre_industrial_concentration=pre_industrial_concentration,
            timestep=1,
            natural_emissions_adjustment=natural_emissions_adjustment,
        )

# ### Four panel plot
if plots:
    fig, ax = pl.subplots(1, 3, figsize=(18 / 2.54, 6 / 2.54))
    for model in models:
        ax[0].plot(
            np.arange(1750, 2101),
            lifetime_scaling[model] * parameters[model]["base"],
            label=model,
            lw=1,
        )
    ax[0].plot(
        np.arange(1750, 2022),
        lifetime_scaling["best_fit"] * parameters["best_fit"]["base"],
        color="0.5",
        label="Best fit",
        lw=1,
    )
    ax[0].set_xlim(1750, 2100)
    ax[0].set_ylabel("yr")
    ax[0].set_title("(a) CH$_4$ lifetime")

    #    for model in models:
    #        ax[1].plot(np.arange(1750, 2101), conc_ch4[model], label=model)
    ax[1].plot(
        np.arange(1750, 2022), conc_ch4["best_fit"], color="0.5", label="Best fit", lw=1
    )
    ax[1].plot(
        np.arange(1750, 2022), input_obs["CH4"], color="k", label="observations", lw=1
    )
    ax[1].set_ylabel("ppb")
    ax[1].set_xlim(1750, 2022)
    ax[1].legend(frameon=False)
    ax[1].set_title("(b) CH$_4$ concentration")

    for ssp in scenarios:
        ax[2].plot(
            np.arange(1750, 2101), conc_ch4[ssp], label=ssp, color=ar7_colors[ssp], lw=1
        )
    ax[2].set_ylabel("ppb")
    ax[2].set_title("(c) CH$_4$ projections")
    ax[2].set_xlim(1750, 2100)
    ax[2].legend(frameon=False)

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

# these are the feedback values per ppb / per Mt that go into FaIR
out = np.empty((1, 7))
out[0, 0] = lifetime_scaling["best_fit"][0] * parameters["best_fit"]["base"]
for i, specie in enumerate(["CH4", "NOx", "VOC", "HC", "N2O"]):
    out[0, i + 1] = parameters["best_fit"][specie] / normalisation_obs[specie]
out[0, 6] = parameters["best_fit"]["temp"]
# out[0, 7] = parameters["best_fit"]["nat"]

df = pd.DataFrame(
    out,
    columns=["base", "CH4", "NOx", "VOC", "HC", "N2O", "temp"],
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
