#!/usr/bin/env python
# coding: utf-8

"""Put APRP results in CSV files."""

import glob
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv

warnings.simplefilter('ignore')

load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")

print("Tabulating aerosol...")

datadir = f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/aerosol'

resultdirs = glob.glob(f'{datadir}/*/*')

for model_run in resultdirs:
    model = model_run.split('/')[-2]
    run = model_run.split('/')[-1]
    data = {}
    for var in ['ERF', 'ERFariSW', 'ERFaciSW', 'ERFariLW', 'ERFaciLW', 'albedo']:
        ds = xr.load_dataset(f'{model_run}/{var}.nc')
        data[var] = ds[var].data
        #time = ds['time'].data
        time = ds['year'].data

    data['ERFari'] = data['ERFariSW'] + data['ERFariLW']
    data['ERFaci'] = data['ERFaciSW'] + data['ERFaciLW']
    data['residual'] = data['ERF'] - data['ERFari'] - data['ERFaci'] - data['albedo']
    pd.DataFrame(data, index=time).to_csv(f'{model_run}/{model}_{run}_aerosol_forcing.csv')
