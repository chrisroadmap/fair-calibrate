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
# Use observations 1850-2022 from IGCC, then use ssp370 projections from IPCC
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
    "slcf_emissions_1750-2022.csv",
    index_col=0,
)
df_conc_obs = pd.read_csv(
    "../../../../../data/concentrations/ghg_concentrations_1750-2022.csv", index_col=0
)
for year in range(1751, 1850):
    df_conc_obs.loc[year, :] = np.nan
df_conc_obs.sort_index(inplace=True)
df_conc_obs.interpolate(inplace=True)

input_obs = {}
input_obs["CH4"] = df_conc_obs["CH4"].values[:273]
input_obs["N2O"] = df_conc_obs["N2O"].values[:273]
input_obs["VOC"] = df_emis_obs["NMVOC"].values[:273]
input_obs["NOx"] = df_emis_obs["NOx"].values[:273]
input_obs["temp"] = gmst[:273]

df_ch4emis_obs = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "all_scaled_1750-2022.csv"
)
emis_ch4_obs = df_ch4emis_obs.loc[
    df_ch4emis_obs["variable"] == "Emissions|CH4", "1750":"2022"
].values.squeeze()

emis_ch4_obs = emis_ch4_obs

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
input_obs["HC"] = total_eesc[:273]

if plots:
    pl.plot(input_obs["HC"])
    pl.savefig(
        f"../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/"
        "eesc_from_observed_conc.png"
    )
    pl.close()

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
    conc_ch4 = np.zeros(273)  # 1750-2022 timebounds
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

    for i in range(273):
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

# natural bounds from global methane budget (part of GCP)
p, cov = scipy.optimize.curve_fit(
    fit_precursors, invect, input_obs["CH4"], bounds=(low - gap, high + gap)
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
        np.arange(1750, 2023),
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

conc_ch4["best_fit"] = np.zeros(273)
gas_boxes = 0
airborne_emissions = 0

emis_ch4_obs = df_ch4emis_obs.loc[
    df_ch4emis_obs["variable"] == "Emissions|CH4", "1750":"2022"
].values.squeeze()

for i in range(273):
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
    da_emissions = xr.load_dataarray(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
        "ssps_harmonized_1750-2499.nc"
    )

    da = da_emissions.loc[dict(config="unspecified", scenario=ssp, specie="CH4")][
        :351, ...
    ]

    emis_ch4_ssps[ssp] = da_emissions.loc[
        dict(config="unspecified", scenario=ssp, specie="CH4")
    ].data[:351]
    emis_nox_ssps[ssp] = da_emissions.loc[
        dict(config="unspecified", scenario=ssp, specie="NOx")
    ].data[:351]
    emis_voc_ssps[ssp] = da_emissions.loc[
        dict(config="unspecified", scenario=ssp, specie="VOC")
    ].data[:351]
    conc_n2o_ssps[ssp] = (
        df_conc.loc[
            (df_conc["Scenario"] == ssp)
            & (df_conc["Variable"].str.endswith("N2O"))
            & (df_conc["Region"] == "World"),
            "1750":"2100",
        ]
        .interpolate(axis=1)
        .values.squeeze()
    )
    total_eesc = 0
    for species in hc_species:
        species_rcmip_name = species.replace("-", "")
        tempinput = (
            df_conc.loc[
                (df_conc["Scenario"] == ssp)
                & (df_conc["Variable"].str.endswith(species_rcmip_name))
                & (df_conc["Region"] == "World"),
                "1750":"2100",
            ]
            .interpolate(axis=1)
            .values.squeeze()
        )
        tempeesc = calculate_eesc(
            tempinput,
            fractional_release[species],
            fractional_release["CFC-11"],
            cl_atoms[species],
            br_atoms[species],
        )
        total_eesc = total_eesc + tempeesc

    conc_eesc_ssps[ssp] = total_eesc

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

    fig, ax = pl.subplots(1, 3, figsize=(18 / 2.54, 6 / 2.54))
    for model in models:
        ax[0].plot(
            np.arange(1750, 2101),
            lifetime_scaling[model] * parameters[model]["base"],
            label=model,
            lw=1,
        )
    ax[0].plot(
        np.arange(1750, 2023),
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
        np.arange(1750, 2023), conc_ch4["best_fit"], color="0.5", label="Best fit", lw=1
    )
    ax[1].plot(
        np.arange(1750, 2023), input_obs["CH4"], color="k", label="observations", lw=1
    )
    ax[1].set_ylabel("ppb")
    ax[1].set_xlim(1750, 2023)
    ax[1].legend(frameon=False)
    ax[1].set_title("(b) CH$_4$ concentration")

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
                & (rcmip_df["Variable"].str.endswith("|CH4")),
                "1750":"2100",
            ]
            .interpolate(axis=1)
            .squeeze()
        )

        ax[2].plot(
            np.arange(1750, 2101), conc_ch4[ssp], label=ssp, color=ar6_colors[ssp], lw=1
        )
        ax[2].plot(np.arange(1750, 2101), gas, color=ar6_colors[ssp], lw=0.3)
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
