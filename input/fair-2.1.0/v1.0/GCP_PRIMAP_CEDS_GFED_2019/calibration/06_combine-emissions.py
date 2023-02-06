#!/usr/bin/env python
# coding: utf-8

# Note: from time to time, the PRIMAP data will be updated and hence these files will
# no longer be current. And it should be a new minor release of the calibration if
# data is updated.

# Here, we use v2.4.
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

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")

update = pd.DataFrame(columns=range(1750, 2022))

rcmip_emissions_file = pooch.retrieve(
    url="doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
)

# RCMIP
emis_df = pd.read_csv(rcmip_emissions_file)

# PRIMAP
primap24_df = pd.read_csv('../../../../../data/emissions/primap-hist-2.4_1750-2021.csv', index_col=0)

# GFED
gfed41s_df = pd.read_csv('../../../../../data/emissions/gfed4.1s_1997-2021.csv', index_col=0)
n2o_biomass = pd.read_csv('../../../../../data/emissions/vua_n2o_biomass_burning_global_totals_1750-2015.csv', index_col=0)

# ch4
rcmip = emis_df.loc[
    (emis_df['Scenario']=='historical')&
    (emis_df['Region']=='World')&
    (emis_df['Variable'].str.startswith(f'Emissions|CH4|')),
:]
ceds_rcmip = [f'Emissions|CH4|MAGICC AFOLU|Agriculture', f'Emissions|CH4|MAGICC Fossil and Industrial']
uva_rcmip = [
    f'Emissions|CH4|MAGICC AFOLU|Agricultural Waste Burning',
    f'Emissions|CH4|MAGICC AFOLU|Forest Burning',
    f'Emissions|CH4|MAGICC AFOLU|Grassland Burning',
    f'Emissions|CH4|MAGICC AFOLU|Peat Burning'
]

update.loc['CH4', 1750:1996] = (
    primap24_df.loc['CH4', '1750':'1996'].values.squeeze() +
    rcmip.loc[rcmip['Variable'].isin(uva_rcmip), '1750':'1996'].interpolate(axis=1).sum().values.squeeze()#[:-4]
)

update.loc['CH4', 1997:2021] = (
    primap24_df.loc['CH4', '1997':'2021'].values.squeeze() +
    gfed41s_df.loc['CH4', '1997':'2021'].values.squeeze()
)

# n2o
# biomass burning emissions calculated previously using a very similar method to this:
# https://github.com/openclimatedata/global-biomass-burning-emissions

update.loc['N2O', 1750:1996] = (
    primap24_df.loc['N2O', '1750':'1996'].values.squeeze() +
    n2o_biomass.loc[1750:1996].values.squeeze()
)
update.loc['N2O', 1997:2021] = (
    primap24_df.loc['N2O', '1997':].values.squeeze() +
    gfed41s_df.loc['N2O', '1997':'2021'].values.squeeze()
)

# SLCFs = CEDS + VUA
species = ['Sulfur', 'CO', 'VOC', 'NOx', 'BC', 'OC', 'NH3']

# don't use NMHC field from GFED... don't know what it actually represents
gfed41s_df.loc['NMVOC'] = pd.concat((gfed41s_df.loc['C2H6':'C3H6O'], gfed41s_df.loc['C2H6S':])).sum()


ceds_convert = {specie: 1/1000 for specie in species}
gfed_convert = {specie: 1 for specie in species}
gfed_convert['NOx'] = 46.006/30.006
ceds_names = {specie: specie for specie in species}
ceds_names.update({'VOC': 'NMVOC', 'Sulfur': 'SO2'})


for specie in species:
    df_ceds_latest = pd.read_csv(f'../../../../../data/emissions/ceds_v20210421/{ceds_names[specie]}_global_CEDS_emissions_by_sector_2021_04_21.csv')
    df_ceds_latest.drop(columns=['em', 'sector', 'units'], inplace=True)
    rcmip = emis_df.loc[
        (emis_df['Scenario']=='ssp245')&
        (emis_df['Region']=='World')&
        (emis_df['Variable'].str.startswith(f'Emissions|{specie}|')),
    :]
    ceds_rcmip = [f'Emissions|{specie}|MAGICC AFOLU|Agriculture', f'Emissions|{specie}|MAGICC Fossil and Industrial']
    uva_rcmip = [
        f'Emissions|{specie}|MAGICC AFOLU|Agricultural Waste Burning',
        f'Emissions|{specie}|MAGICC AFOLU|Forest Burning',
        f'Emissions|{specie}|MAGICC AFOLU|Grassland Burning',
        f'Emissions|{specie}|MAGICC AFOLU|Peat Burning'
    ]
    update.loc[specie, 1750:1996] = (
        df_ceds_latest.sum().values[:247] * ceds_convert[specie] +
        rcmip.loc[rcmip['Variable'].isin(uva_rcmip), '1750':'1996'].interpolate(axis=1).sum().values.squeeze() * gfed_convert[specie]
    )
    update.loc[specie, 1997:2019] = (
        df_ceds_latest.sum().values[247:] * ceds_convert[specie] +
        gfed41s_df.loc[ceds_names[specie], '1997':'2019'].values.squeeze() * gfed_convert[specie]
    )
    update.loc[specie, 2020:2021] = (
        df_ceds_latest.sum().values[-1] * ceds_convert[specie] +
        gfed41s_df.loc[ceds_names[specie], '2020':'2021'].values.squeeze() * gfed_convert[specie]
    )


print(update)
# update.T.plot()
# pl.show()

os.makedirs(f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/", exist_ok=True)

update.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/"
    "primap_ceds_gfed_1750-2021.csv"
)
