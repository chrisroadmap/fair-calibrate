#!/usr/bin/env python
# coding: utf-8

# Note: from time to time, the PRIMAP data will be updated and hence these files will
# no longer be current. And it should be a new minor release of the calibration if
# data is updated.

# Here, we use v2.4.

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

print(update)





# df_out = pd.DataFrame([ch4, n2o], columns = ["CH4", "N2O"], index=np.arange(1750, 2022))
#
# os.makedirs(
#     f"../../../../../data/emissions/",
#     exist_ok=True,
# )
#
# df_out.to_csv('../../../../../data/emissions/primap-hist-2.4_1750-2021.csv')
