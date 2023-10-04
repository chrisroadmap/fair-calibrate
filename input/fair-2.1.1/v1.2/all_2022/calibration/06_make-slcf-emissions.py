#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import pooch
import h5py
import matplotlib.pyplot as pl
from dotenv import load_dotenv

load_dotenv()

pl.style.use("../../../../../defaults.mplstyle")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")
datadir = os.getenv("DATADIR")

species = ['BC', 'OC', 'SO2', 'NOx', 'CO', 'NMVOC', 'NH3']

slcf_df = pd.DataFrame(columns = species, index=np.arange(1750, 2023, dtype=int))
ceds_df = pd.DataFrame(columns = species, index=np.arange(1750, 2020, dtype=int))

for specie in species:
    ceds_df.loc[:, specie] = 0.001 * pd.read_csv(f'../../../../../data/emissions/ceds/v20210421/{specie}_global_CEDS_emissions_by_sector_2021_04_21.csv').sum()['X1750':].values

gfed41s_df = pd.read_csv('../../../../../data/emissions/gfed4.1s_1997-2022.csv', index_col=0)

bb_df = pd.read_csv('../../../../../data/emissions/global-biomass-burning-emissions.csv', index_col=0)

gfed_convert = {specie: 1 for specie in species}
gfed_convert['NOx'] = 46.006/30.006
rcmip_specie = {specie: specie for specie in species}
rcmip_specie['NMVOC'] = 'VOC'

covid_df = pd.read_csv('../../../../../data/emissions/forster_et_al_2020_two_year_blip_emissions_ratios.csv', index_col=0)
covid_df

for specie in species:
    slcf_df.loc[1750:1996, specie] = (
        ceds_df.loc[1750:1996, specie]
    ) + bb_df.loc[1750:1996, specie].values.squeeze() * gfed_convert[specie]

    slcf_df.loc[1997:2019, specie] = (
        ceds_df.loc[1997:2019, specie]
    ) + gfed41s_df.loc[1997:2019, specie].values.squeeze() * gfed_convert[specie]

    slcf_df.loc[2020:2022, specie] = (
        ceds_df.loc[2019, specie] * covid_df.loc[2020:2022, rcmip_specie[specie]]
    ) + gfed41s_df.loc[2020:2022, specie].values.squeeze() * gfed_convert[specie]


os.makedirs(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions', exist_ok=True)
slcf_df.to_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/slcf_emissions_1750-2022.csv')
