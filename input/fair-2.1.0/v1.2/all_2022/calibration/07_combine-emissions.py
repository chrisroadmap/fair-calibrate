#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as pl
import os
from pathlib import PurePath

import h5py
import numpy as np
from dotenv import load_dotenv
import pooch
import pandas as pd

load_dotenv()

print("Combining CEDS, PRIMAP and GFED data into one file...")

# the purpose of this really is to make a pyam friendly format to use for harmonisation

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

update = pd.DataFrame(columns=range(1750, 2023))

# RCMIP
rcmip_emissions_file = pooch.retrieve(
    url="doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
)
emis_df = pd.read_csv(rcmip_emissions_file)

# SLCF pre-processed from CEDS and GFED - should put code in
slcf_df = pd.read_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/slcf_emissions_1750-2022.csv', index_col=0)

# PRIMAP for CH4, N2O, SF6 and NF3
primap24_df = pd.read_csv('../../../../../data/emissions/primap-hist-2.4.2_1750-2021.csv', index_col=0)

# GFED
gfed41s_df = pd.read_csv('../../../../../data/emissions/gfed4.1s_1997-2022.csv', index_col=0)

# biomass burning (from open climate data)
# https://github.com/openclimatedata/global-biomass-burning-emissions
bb_df = pd.read_csv('../../../../../data/emissions/global-biomass-burning-emissions.csv', index_col=0)

# PRIMAP does not include biomass burning but does include agriculture.
# CEDS is the same.
# The CMIP6 RCMIP datasets have breakdowns by sector and should match the 
# cmip6_biomass totals, so they can be used from Zeb's data.
# The exception is N2O, which does not have this granularity in RCMIP.
# Assumptions for CH4 and N2O: 2022 fossil+agriculture is same as 2021.
# For SF6 and NF3 also assume 2021 emissions in 2022.

# ch4
update.loc['Emissions|CH4', 1750:1996] = (
    primap24_df.loc['CH4', '1750':'1996'].values.squeeze() +
    bb_df.loc[1750:1996, 'CH4'].values.squeeze()
)
update.loc['Emissions|CH4', 1997:2021] = (
    primap24_df.loc['CH4', '1997':].values.squeeze() +
    gfed41s_df.loc[1997:2021, 'CH4'].values.squeeze()
)
update.loc['Emissions|CH4', 2022] = (
    primap24_df.loc['CH4', '2021'] +
    gfed41s_df.loc[2022, 'CH4']
)

# n2o
update.loc['Emissions|N2O', 1750:1996] = (
    primap24_df.loc['N2O', '1750':'1996'].values.squeeze() +
    bb_df.loc[1750:1996, 'N2O'].values.squeeze()
)
update.loc['Emissions|N2O', 1997:2021] = (
    primap24_df.loc['N2O', '1997':].values.squeeze() +
    gfed41s_df.loc[1997:2021, 'N2O'].values.squeeze()
)
update.loc['Emissions|N2O', 2022] = (
    primap24_df.loc['N2O', '2021'] +
    gfed41s_df.loc[2022, 'N2O']
)


# sf6
update.loc['Emissions|SF6', 1750:2021] = (
    primap24_df.loc['SF6', '1750':'2021'].values.squeeze() 
)
update.loc['Emissions|SF6', 2022] = primap24_df.loc['SF6', '2021']

# nf3
update.loc['Emissions|NF3', 1750:2021] = (
    primap24_df.loc['NF3', '1750':'2021'].values.squeeze() 
)
update.loc['Emissions|NF3', 2022] = primap24_df.loc['NF3', '2021']


# SLCFs: already calculated
species = ['Sulfur', 'CO', 'VOC', 'NOx', 'BC', 'OC', 'NH3']

names = {specie: specie for specie in species}
names.update({'VOC': 'NMVOC', 'Sulfur': 'SO2'})


for specie in species:
    update.loc[f"Emissions|{specie}", 1750:2022] = slcf_df[names[specie]].values.squeeze()


update = update.rename_axis('Variable').reset_index(level=0)

update.insert(loc=0, column="Model", value = "History")
update.insert(loc=1, column="Scenario", value = "PRIMAP+CEDS+GFED")
update.insert(loc=2, column="Region", value = "World")
update.insert(loc=4, column="Unit", value = ["Mt CH4/yr", "Mt N2O/yr", "kt SF6/yr", "kt NF3/yr", "Mt SO2/yr",
    "Mt CO/yr", "Mt VOC/yr", "Mt NO2/yr", "Mt BC/yr", "Mt OC/yr", "Mt NH3/yr"])
print(update)

os.makedirs(f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/", exist_ok=True)

update.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "primap_ceds_gfed_1750-2022.csv",
    index=False
)
