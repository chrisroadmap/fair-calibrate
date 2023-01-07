#!/usr/bin/env python
# coding: utf-8

"""ERF scaling factors"""
#
# Based on AR6 Chapter 7 ERF uncertainty
#
# We do not modify forcing scale factors for ozone and aerosols, because we adjust the precursor species to span the forcing uncertainty this way.

import os

import pandas as pd
import scipy.stats
import numpy as np
import matplotlib.pyplot as pl

from fair import __version__
from dotenv import load_dotenv

load_dotenv()

print("Doing forcing uncertainty sampling...")


cal_v = os.getenv('CALIBRATION_VERSION')
fair_v = os.getenv('FAIR_VERSION')
constraint_set = os.getenv('CONSTRAINT_SET')
samples = int(os.getenv("PRIOR_SAMPLES"))

assert fair_v == __version__

NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)
NINETY_TO_ONESIGMA

forcing_u90 = {
#    'CO2': 0.12,      # CO2
    'CH4': 0.20,      # CH4: updated value from etminan 2016
    'N2O': 0.14,      # N2O
    'minorGHG': 0.19,      # other WMGHGs
    'Stratospheric water vapour': 1.00,
    'Contrails' : 0.70,      # contrails approx - half-normal
    'Light absorbing particles on snow and ice': 1.25,      # bc on snow - half-normal
    'Land use': 0.50,      # land use change
    'Volcanic': 5.0/20.0,  # needs to be way bigger?
    'solar_amplitude': 0.50,
    'solar_trend': 0.07,
}

seedgen = 380133900
scalings = {}
for forcer in forcing_u90:
    scalings[forcer] = scipy.stats.norm.rvs(1, forcing_u90[forcer]/NINETY_TO_ONESIGMA, size=samples, random_state=seedgen)
    seedgen=seedgen+112

## LAPSI is asymmetric Gaussian. We can just scale the half of the distribution above/below best estimate
scalings['Light absorbing particles on snow and ice'][scalings['Light absorbing particles on snow and ice']<1] = 0.08/0.1*(scalings['Light absorbing particles on snow and ice'][scalings['Light absorbing particles on snow and ice']<1]-1) + 1

## so is contrails - the benefits of doing this are tiny :)
scalings['Contrails'][scalings['Contrails']<1] = 0.0384/0.0406*(scalings['Contrails'][scalings['Contrails']<1]-1) + 1

## Solar trend is absolute, not scaled
scalings['solar_trend'] = scalings['solar_trend'] - 1

# take CO2 scaling from 4xCO2 generated from the EBMs
df_ebm = pd.read_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/climate_response_ebm3.csv')

scalings['CO2'] = np.array(1 + 0.563*(df_ebm['F_4xCO2'].mean() - df_ebm['F_4xCO2'])/df_ebm['F_4xCO2'].mean())

df_out = pd.DataFrame(scalings, columns=scalings.keys())

df_out.to_csv(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/forcing_scaling.csv', index=False)
