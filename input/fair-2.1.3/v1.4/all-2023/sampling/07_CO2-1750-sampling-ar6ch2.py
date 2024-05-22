#!/usr/bin/env python
# coding: utf-8

"""Sampling CO2 concentration in 1750"""
#
# 1750 concentration was 278.3 ppm +/- 2.9 ppm (data provided by Jinho Ahn, IPCC AR6 WG1
# Ch2).

# The X2019 scale has shifted this marginally upwards. I'm not convinced the new value
# makes sense, but we use it anyway.

import os

import pandas as pd
import scipy.stats
from dotenv import load_dotenv
from fair import __version__

load_dotenv(override=True)

print("Sampling 1750 CO2 concentration...")

# Get environment variables
load_dotenv(override=True)

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))

assert fair_v == __version__

NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)
co2_1750_conc = scipy.stats.norm.rvs(
    size=samples, loc=278.377857, scale=2.9 / NINETY_TO_ONESIGMA, random_state=1067061
)

df = pd.DataFrame({"co2_concentration": co2_1750_conc})

df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "co2_concentration_1750.csv",
    index=False,
)
