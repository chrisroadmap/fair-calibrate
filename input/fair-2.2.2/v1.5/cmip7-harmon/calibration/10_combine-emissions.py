#!/usr/bin/env python
# coding: utf-8

import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

print("Combining GCP, CEDS, PRIMAP, GFED data into one file...")

# the purpose of this really is to make a pyam friendly format to use for harmonisation
cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

update = pd.DataFrame(columns=range(1750, 2025))

# CO2 from Global Carbon Project
co2_df = pd.read_csv(
    "../../../../../data/emissions/gcp_v2024_co2_1750-2023_prelim_2024.csv", index_col=0
)

# my reconstruction combination of GFED and CEDS for CH4 FFI+AG
ch4_df = pd.read_csv(
    "../../../../../data/emissions/CH4_reconstructed_1750-2023.csv", index_col=0
)

# SLCF pre-processed from CEDS and GFED
slcf_df = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "slcf_emissions_1750-2022.csv",
    index_col=0,
)

# PRIMAP for N2O FFI+AG
primap_df = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "primap-histtp-2.6_1750-2023.csv",
    index_col=0,
)

# GFED
gfed41s_df = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "gfed4.1s_1997-2023.csv",
    index_col=0,
)

# pre-1997 biomass burning
bb_df = pd.read_csv(
    "../../../../../data/emissions/global-biomass-burning-emissions.csv", index_col=0
)

# inverse minor GHGs
inv_df = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "minor_ghg_inverse_1750-2022.csv"
)

# PRIMAP does not include biomass burning but does include agriculture. Therefore, CH4
# and N2O emissions from BB need to be added to the emissions from PRIMAP.

# The CMIP6 RCMIP datasets have breakdowns by sector and should match the
# cmip6_biomass totals, so they can be used from Zeb's data.
# The exception is N2O, which does not have this granularity in RCMIP.

# co2
update.loc["Emissions|CO2|Energy and Industrial Processes", 1750:2024] = co2_df.loc[
    "1750":"2024", "fossil emissions excluding carbonation"
].values
update.loc["Emissions|CO2|AFOLU", 1750:2024] = co2_df.loc[
    "1750":"2024", "land-use change emissions"
].values

# ch4
update.loc["Emissions|CH4", 1750:1996] = (
    ch4_df.loc[1750:1996, "reconstructed"].values + bb_df.loc[1750:1996, "CH4"].values
)
update.loc["Emissions|CH4", 1997:2023] = (
    ch4_df.loc[1997:2023, "reconstructed"].values + gfed41s_df.loc[1997:2023, "CH4"].values
)

# n2o
update.loc["Emissions|N2O", 1750:1996] = (
    primap_df.loc["N2O", "1750":"1996"].values + bb_df.loc[1750:1996, "N2O"].values
)
update.loc["Emissions|N2O", 1997:2023] = (
    primap_df.loc["N2O", "1997":"2023"].values + gfed41s_df.loc[1997:2023, "N2O"].values
)

# SLCFs: already calculated
species = ["Sulfur", "CO", "VOC", "NOx", "BC", "OC", "NH3"]

names = {specie: specie for specie in species}
names.update({"VOC": "NMVOC", "Sulfur": "SO2"})

for specie in species:
    update.loc[f"Emissions|{specie}", 1750:2022] = slcf_df[
        names[specie]
    ].values.squeeze()

# Minor GHGs
species_minor = [
    "HFC-134a",
    "HFC-23",
    "HFC-32",
    "HFC-125",
    "HFC-143a",
    "HFC-152a",
    "HFC-227ea",
    "HFC-236fa",
    "HFC-245fa",
    "HFC-365mfc",
    "HFC-4310mee",
    "NF3",
    "SF6",
    "SO2F2",
    "CF4",
    "C2F6",
    "C3F8",
    "c-C4F8",
    "CFC-12",
    "CFC-11",
    "CFC-113",
    "CFC-114",
    "CFC-115",
    "HCFC-22",
    "HCFC-141b",
    "HCFC-142b",
    "CH3CCl3",
    "CCl4",
    "CH3Cl",
    "CH3Br",
    "CH2Cl2",
    "CHCl3",
    "Halon-1211",
    "Halon-1301",
    "Halon-2402",
    "C4F10",
    "C5F12",
    "C6F14",
    "C7F16",
    "C8F18",
]
for specie in species_minor:
    update.loc[f"Emissions|{specie}", 1750:2022] = inv_df[specie].values.squeeze()

units = [
    "Gt CO2/yr",
    "Gt CO2/yr",
    "Mt CH4/yr",
    "Mt N2O/yr",
    "Mt SO2/yr",
    "Mt CO/yr",
    "Mt VOC/yr",
    "Mt NO2/yr",
    "Mt BC/yr",
    "Mt OC/yr",
    "Mt NH3/yr",
]
units = units + [
    f'kt {specie.replace("-", "")}/yr' for specie in species_minor
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
    "all_1750-2024.csv",
    index=False,
)
