#!/usr/bin/env python
# coding: utf-8

"""ERF scaling factors"""
#
# Based on AR6 Chapter 7 ERF uncertainty
#
# We do not modify forcing scale factors for ozone and aerosols, because we adjust the
# precursor species to span the forcing uncertainty this way.

import os

import pandas as pd
import scipy.stats
from dotenv import load_dotenv
from fair import __version__
from sklearn.preprocessing import QuantileTransformer

load_dotenv()

print("Doing forcing uncertainty sampling...")


cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))

assert fair_v == __version__

NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)
NINETY_TO_ONESIGMA

forcing_u90 = {
    "CH4": 0.20,  # CH4: updated value from etminan 2016
    "N2O": 0.14,  # N2O
    "minorGHG": 0.19,  # other WMGHGs
    "Stratospheric water vapour": 1.00,
    "Land use": 0.50,  # land use change
    "Volcanic": 5.0 / 20.0,  # needs to be way bigger?
    "solar_amplitude": 0.50,
    "solar_trend": 0.07,
}

seedgen = 380133900
scalings = {}
for forcer in forcing_u90:
    scalings[forcer] = scipy.stats.norm.rvs(
        1, forcing_u90[forcer] / NINETY_TO_ONESIGMA, size=samples, random_state=seedgen
    )
    seedgen = seedgen + 112


def opt(x, q05_desired, q50_desired, q95_desired):
    "x is (a, loc, scale) in that order."
    q05, q50, q95 = scipy.stats.skewnorm.ppf(
        (0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2]
    )
    return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)


# Asymmetric distributions we use skew-normal, fitting quantiles
lapsi_params = scipy.optimize.root(opt, [1, 1, 1], args=(0, 1, 2.25)).x
contrails_params = scipy.optimize.root(opt, [1, 1, 1], args=(19 / 57, 1, 98 / 57)).x

scalings["Light absorbing particles on snow and ice"] = scipy.stats.skewnorm.rvs(
    lapsi_params[0],
    loc=lapsi_params[1],
    scale=lapsi_params[2],
    size=samples,
    random_state=3701584,
)

# Solar trend is absolute, not scaled
scalings["solar_trend"] = scalings["solar_trend"] - 1

# CO2 scaling is quantile mapping from ERF 4xCO2 and +/- 12%
df_ebm = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "climate_response_ebm3.csv"
)

qt = QuantileTransformer(output_distribution="normal", random_state=70601701)
f4xco2 = df_ebm["F_4xCO2"].values.reshape(-1, 1)
trans = qt.fit(f4xco2).transform(f4xco2)
trans = 1 + trans * 0.12 / NINETY_TO_ONESIGMA
scalings["CO2"] = trans.squeeze()

df_out = pd.DataFrame(scalings, columns=scalings.keys())

df_out.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "forcing_scaling.csv",
    index=False,
)
