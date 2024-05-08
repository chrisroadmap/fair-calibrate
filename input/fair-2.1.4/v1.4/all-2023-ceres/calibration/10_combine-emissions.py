#!/usr/bin/env python
# coding: utf-8

import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

print("Combining GCP, CEDS, PRIMAP, GFED data into one file...")

# the purpose of this really is to make a pyam friendly format to use for harmonisation
# TODO: include the decomposition of PFCs and HFCs
cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

update = pd.DataFrame(columns=range(1750, 2023))

# CO2 from Global Carbon Project
co2_df = pd.read_csv(
    "../../../../../data/emissions/gcp_v2023_co2_1750-2022.csv", index_col=0
)

# SLCF pre-processed from CEDS and GFED - should put code in
slcf_df = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "slcf_emissions_1750-2022.csv",
    index_col=0,
)

# PRIMAP for CH4 FFI+AG, N2O FFI+AG, SF6, NF3
primap_df = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "primap-histtp-2.5_1750-2022.csv",
    index_col=0,
)

# Split of HFCs and PFCs
split_df = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "hfcs_pfcs_split_1750-2022.csv",
    index_col=0,
)

# GFED
gfed41s_df = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "gfed4.1s_1997-2022.csv",
    index_col=0,
)

# pre-1997 biomass burning
bb_df = pd.read_csv(
    "../../../../../data/emissions/global-biomass-burning-emissions.csv", index_col=0
)

# inverse Montreal
inv_df = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "minor_ghg_inverse_1750-2022.csv"
)

# PRIMAP does not include biomass burning but does include agriculture. Therefore, CH4
# and N2O emissions from BB need to be added to the emissions from PRIMAP.

# The CMIP6 RCMIP datasets have breakdowns by sector and should match the
# cmip6_biomass totals, so they can be used from Zeb's data.
# The exception is N2O, which does not have this granularity in RCMIP.
# Assumptions for CH4 and N2O: 2022 fossil+agriculture is same as 2021.
# For SF6 and NF3 also assume 2021 emissions in 2022.

# co2
update.loc["Emissions|CO2|Energy and Industrial Processes", 1750:2022] = co2_df.loc[
    "1750":"2022", "fossil emissions excluding carbonation"
].values
update.loc["Emissions|CO2|AFOLU", 1750:2022] = co2_df.loc[
    "1750":"2022", "land-use change emissions"
].values

# ch4
update.loc["Emissions|CH4", 1750:1996] = (
    primap_df.loc["CH4", "1750":"1996"].values + bb_df.loc[1750:1996, "CH4"].values
)
update.loc["Emissions|CH4", 1997:2022] = (
    primap_df.loc["CH4", "1997":"2022"].values + gfed41s_df.loc[1997:2022, "CH4"].values
)

# n2o
update.loc["Emissions|N2O", 1750:1996] = (
    primap_df.loc["N2O", "1750":"1996"].values + bb_df.loc[1750:1996, "N2O"].values
)
update.loc["Emissions|N2O", 1997:2022] = (
    primap_df.loc["N2O", "1997":"2022"].values + gfed41s_df.loc[1997:2022, "N2O"].values
)

# NF3 and SF6
update.loc["Emissions|SF6", 1750:2022] = primap_df.loc["SF6", "1750":"2022"].values
update.loc["Emissions|NF3", 1750:2022] = primap_df.loc["NF3", "1750":"2022"].values

# SLCFs: already calculated
species = ["Sulfur", "CO", "VOC", "NOx", "BC", "OC", "NH3"]

names = {specie: specie for specie in species}
names.update({"VOC": "NMVOC", "Sulfur": "SO2"})

for specie in species:
    update.loc[f"Emissions|{specie}", 1750:2022] = slcf_df[
        names[specie]
    ].values.squeeze()

# Montreal GHGs + SO2F2
species_minor = [
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
    "SO2F2",
]
for specie in species_minor:
    update.loc[f"Emissions|{specie}", 1750:2022] = inv_df[specie].values.squeeze()

# HFC/PFC split
species_split = [
    "HFC-125",
    "HFC-134a",
    "HFC-143a",
    "HFC-152a",
    "HFC-227ea",
    "HFC-23",
    "HFC-236fa",
    "HFC-245fa",
    "HFC-32",
    "HFC-365mfc",
    "HFC-4310mee",
    "C2F6",
    "C3F8",
    "C4F10",
    "C5F12",
    "C6F14",
    "C7F16",
    "C8F18",
    "CF4",
    "c-C4F8",
]

for specie in species_split:
    update.loc[f"Emissions|{specie}", 1750:2022] = split_df.loc[
        split_df.index.str.endswith(specie.replace("-", "")), :
    ].values.squeeze()


units = [
    "Gt CO2/yr",
    "Gt CO2/yr",
    "Mt CH4/yr",
    "Mt N2O/yr",
    "kt SF6/yr",
    "kt NF3/yr",
    "Mt SO2/yr",
    "Mt CO/yr",
    "Mt VOC/yr",
    "Mt NO2/yr",
    "Mt BC/yr",
    "Mt OC/yr",
    "Mt NH3/yr",
]
units = units + [
    f'kt {specie.replace("-", "")}/yr' for specie in species_minor + species_split
]

update = update.rename_axis("Variable").reset_index(level=0)

update.insert(loc=0, column="Model", value="Historical")
update.insert(loc=1, column="Scenario", value="GCP+CEDS+PRIMAP+GFED")
update.insert(loc=2, column="Region", value="World")
update.insert(loc=4, column="Unit", value=units)
print(update)

os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/",
    exist_ok=True,
)

update.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "all_1750-2022.csv",
    index=False,
)
