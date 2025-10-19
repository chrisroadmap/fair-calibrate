#!/usr/bin/env python
# coding: utf-8

"""Sampling CO2 concentration in 1750"""

# 1750 concentration is given by the CMIP7 concentration data
# we'll use the same uncertainty range.

import os

import pandas as pd
import scipy.stats
from dotenv import load_dotenv
from fair import __version__

load_dotenv()

print("Sampling 1750 CO2 concentration...")

# Get environment variables
load_dotenv()

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))

assert fair_v == __version__

# extrapolate the CMIP7 concs backwards
NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)
co2_1750_conc = scipy.stats.norm.rvs(
    size=samples, loc=278.0, scale=2.9 / NINETY_TO_ONESIGMA, random_state=1067061
)

df = pd.DataFrame({"co2_concentration": co2_1750_conc})

df.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "co2_concentration_1750.csv",
    index=False,
)
