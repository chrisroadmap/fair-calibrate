#!/usr/bin/env python
# coding: utf-8

# Note: from time to time, the PRIMAP data will be updated and hence these files will
# no longer be current. And it should be a new minor release of the calibration if
# data is updated.

# Here, we use v2.5 of PRIMAP. We use Third Party emissions as country reported is
# biased low = the countries are telling porky pies

# PRIMAP will be used for CH4, N2O, SF6, NF3, HFCs and PFCs.

import os
from pathlib import PurePath

import numpy as np
from dotenv import load_dotenv
import pooch
import pandas as pd

load_dotenv()

print("Getting PRIMAP data...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
datadir = os.getenv("DATADIR")
progress = os.getenv("PROGRESS", "False").lower() in ("true", "1", "t")

# PRIMAP-hist emissions: newest version
primap25 = pooch.retrieve(
    url="https://zenodo.org/records/10006301/files/Guetschow_et_al_2023b-PRIMAP-hist_v2.5_final_no_rounding_15-Oct-2023.csv",
    known_hash="md5:e4ddeeb06d9cff9c7e16fc320797d2f1",
    progressbar=progress,
    path=datadir,
)

primap25_df = pd.read_csv(primap25)

for scenario in ['HISTTP']:
    ch4 = primap25_df.loc[
        (primap25_df['scenario (PRIMAP-hist)']==scenario)&
        (primap25_df['entity']=='CH4')&
        (primap25_df['category (IPCC2006_PRIMAP)']=='M.0.EL')&
        (primap25_df['area (ISO3)']=='EARTH'),
    '1750':].values.squeeze()

    n2o = primap25_df.loc[
        (primap25_df['scenario (PRIMAP-hist)']==scenario)&
        (primap25_df['entity']=='N2O')&
        (primap25_df['category (IPCC2006_PRIMAP)']=='M.0.EL')&
        (primap25_df['area (ISO3)']=='EARTH'),
    '1750':].values.squeeze()

    sf6 = primap25_df.loc[
        (primap25_df['scenario (PRIMAP-hist)']==scenario)&
        (primap25_df['entity']=='SF6')&
        (primap25_df['category (IPCC2006_PRIMAP)']=='M.0.EL')&
        (primap25_df['area (ISO3)']=='EARTH'),
    '1750':].values.squeeze()

    nf3 = primap25_df.loc[
        (primap25_df['scenario (PRIMAP-hist)']==scenario)&
        (primap25_df['entity']=='NF3')&
        (primap25_df['category (IPCC2006_PRIMAP)']=='M.0.EL')&
        (primap25_df['area (ISO3)']=='EARTH'),
    '1750':].values.squeeze()

    pfcs = primap25_df.loc[
        (primap25_df['scenario (PRIMAP-hist)']==scenario)&
        (primap25_df['entity']=='PFCS (AR6GWP100)')&
        (primap25_df['category (IPCC2006_PRIMAP)']=='M.0.EL')&
        (primap25_df['area (ISO3)']=='EARTH'),
    '1750':].values.squeeze()

    hfcs = primap25_df.loc[
        (primap25_df['scenario (PRIMAP-hist)']==scenario)&
        (primap25_df['entity']=='HFCS (AR6GWP100)')&
        (primap25_df['category (IPCC2006_PRIMAP)']=='M.0.EL')&
        (primap25_df['area (ISO3)']=='EARTH'),
    '1750':].values.squeeze()

    df_out = pd.DataFrame([ch4/1e3, n2o/1e3, sf6, nf3, pfcs, hfcs], index = ["CH4", "N2O", "SF6", "NF3", "PFCs", "HFCs"], columns=np.arange(1750, 2023))

    os.makedirs(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions', exist_ok=True)
    df_out.to_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/emissions/primap-{scenario.lower()}-2.5_1750-2022.csv')
