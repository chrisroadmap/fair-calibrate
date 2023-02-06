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

print("Getting PRIMAP data...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")

# PRIMAP-hist emissions: newest version
primap24 = pooch.retrieve(
    "https://zenodo.org/record/7179775/files/Guetschow-et-al-2022-PRIMAP-hist_v2.4_no_rounding_11-Oct-2022.csv",
    known_hash="md5:3a40d9abb47fe688295af36301424679",
    progressbar=True
)
primap24_df = pd.read_csv(primap24)

ch4 = primap24_df.loc[
    (primap24_df['scenario (PRIMAP-hist)']=='HISTTP')&
    (primap24_df['entity']=='CH4')&
    (primap24_df['category (IPCC2006_PRIMAP)']=='M.0.EL')&
    (primap24_df['area (ISO3)']=='EARTH'),
'1750':].values.squeeze()

n2o = primap24_df.loc[
    (primap24_df['scenario (PRIMAP-hist)']=='HISTTP')&
    (primap24_df['entity']=='N2O')&
    (primap24_df['category (IPCC2006_PRIMAP)']=='M.0.EL')&
    (primap24_df['area (ISO3)']=='EARTH'),
'1750':].values.squeeze()

df_out = pd.DataFrame([ch4/1000, n2o/1000], index = ["CH4", "N2O"], columns=np.arange(1750, 2022))

os.makedirs(
    f"../../../../../data/emissions/",
    exist_ok=True,
)

df_out.to_csv('../../../../../data/emissions/primap-hist-2.4_1750-2021.csv')