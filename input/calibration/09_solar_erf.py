#!/usr/bin/env python
# coding: utf-8

"""Making the solar forcing time series on fair timebounds"""

# Correlate cumulative CO2 AFOLU emissions to land use forcing in the present

import os

import numpy as np
import pandas as pd
import pooch
from dotenv import load_dotenv
from netCDF4 import Dataset

import matplotlib.pyplot as pl

load_dotenv()

print("Making solar forcing ERF time series...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
datadir = os.getenv("DATADIR")

# using CMIP6-Plus right now as it has future
filepath_future = pooch.retrieve(
    url = "https://esgf-node.ornl.gov/thredds/fileServer/user_pub_work/input4MIPs/CMIP6Plus/ScenarioMIP/SOLARIS-HEPPA/SOLARIS-HEPPA-ScenarioMIP-4-6-a002/atmos/mon/multiple/gn/v20251001/multiple_input4MIPs_solar_ScenarioMIP_SOLARIS-HEPPA-ScenarioMIP-4-6-a002_gn_202201-229912.nc",
    known_hash = "97756cb0b1524eaa1ca66a94f883728a54dffe98b6ee66557b24837948715f1d",
)

filepath_historical = pooch.retrieve(
    url = "https://esgf1.dkrz.de/thredds/fileServer/input4mips/input4MIPs/CMIP7/CMIP/SOLARIS-HEPPA/SOLARIS-HEPPA-CMIP-4-6/atmos/mon/multiple/gn/v20250219//multiple_input4MIPs_solar_CMIP_SOLARIS-HEPPA-CMIP-4-6_gn_185001-202312.nc",
    known_hash="536498e442ed4c279ce91eeecabf1908a25142b3dc9a2d0825eb1121886b4581"
)

nc = Dataset(filepath_historical)
tsi = nc.variables["tsi"][:]
time_bnds = nc.variables["time_bnds"]
days_in_month = np.diff(time_bnds[:], axis=1).squeeze()
tsi_ann_mean_1850_2023 = np.zeros(len(tsi)//12)
nc.close()

# pretty sure this can be less clunky
for year in range(len(tsi_ann_mean_1850_2023)):
    tsi_ann_mean_1850_2023[year] = np.average(tsi[year*12:year*12+12], weights=days_in_month[year*12:year*12+12])


nc = Dataset(filepath_future)
tsi = nc.variables["tsi"][:]
time_bnds = nc.variables["time_bnds"]
days_in_month = np.diff(time_bnds[:], axis=1).squeeze()
tsi_ann_mean_2022_2299 = np.zeros(len(tsi)//12)
nc.close()


# pretty sure this can be less clunky
for year in range(len(tsi_ann_mean_2022_2299)):
    tsi_ann_mean_2022_2299[year] = np.average(tsi[year*12:year*12+12], weights=days_in_month[year*12:year*12+12])


# solar forcing
# option: do what cmip has always done and take mean of 1850-1873
# but why not be consistent with volcanic and take mean of full historical
# last solar cycle ended in 2019
# so take mean 1850-2019
# indices 0:170
tsi_ann_mean_1850_2299 = np.concatenate((tsi_ann_mean_1850_2023, tsi_ann_mean_2022_2299[2:]))

solar_erf = (tsi_ann_mean_1850_2299 - np.mean(tsi_ann_mean_1850_2299[0:170])) * 0.25 * 0.71 * 0.72  # Chapter 7 AR6 WG1 - 0.72 is the tropospheric adjustment

# put on fair timebounds and cut when it gets back to zero
solar_erf_trimmed_extended = np.zeros(752)
solar_erf_trimmed_extended[101:551] = solar_erf

df_solar = pd.DataFrame(index=np.arange(1750, 2502), columns=['erf'], data=solar_erf_trimmed_extended)
df_solar.index.name = "year"

# save out
os.makedirs(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/forcing/",
    exist_ok=True,
)
df_solar.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/forcing/"
    "solar_forcing_timebounds.csv"
)
