#!/usr/bin/env python
# coding: utf-8

"""Takes constrained runs and dumps parameters into the output file"""

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fair import __version__

load_dotenv()

print("Dumping output...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))

#assert fair_v == __version__

df_cc = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "carbon_cycle.csv"
)
df_cr = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "climate_response_ebm3.csv"
)
df_aci = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "aerosol_cloud.csv"
)
df_ari = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "aerosol_radiation.csv"
)
df_ozone = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/ozone.csv"
)
df_scaling = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "forcing_scaling.csv"
)
df_1750co2 = pd.read_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
    "co2_concentration_1750.csv"
)

valid_all = np.loadtxt(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "runids_rmse_reweighted_pass.csv"
).astype(
    np.int64
)  # [:1000]
valid_all

seed = 1355763 + 399 * valid_all
seed

params_out = pd.concat(
    (
        df_cr.loc[valid_all, :],
        df_cc.loc[valid_all, :],
        df_ari.loc[valid_all, :],
        df_aci.loc[valid_all, :],
        df_ozone.loc[valid_all, :],
        df_scaling.loc[valid_all, :],
        df_1750co2.loc[valid_all, :],
        pd.Series(seed, index=valid_all),
    ),
    axis=1,
)

pd.Series(seed, index=valid_all)

params_out.columns = [
    "gamma",
    "c1",
    "c2",
    "c3",
    "kappa1",
    "kappa2",
    "kappa3",
    "epsilon",
    "sigma_eta",
    "sigma_xi",
    "F_4xCO2",
    "r0",
    "rU",
    "rT",
    "rA",
    'ari BC',
    'ari OC',
    'ari Sulfur',
    'ari NOx',
    'ari VOC',
    'ari NH3',
    'ari CH4',
    'ari N2O',
    'ari Equivalent effective stratospheric chlorine',
    "shape Sulfur",
    "shape BC",
    "shape OC",
    "beta",
    "o3 CH4",
    "o3 N2O",
    "o3 Equivalent effective stratospheric chlorine",
    "o3 CO",
    "o3 VOC",
    "o3 NOx",
    "scale CH4",
    "scale N2O",
    "scale minorGHG",
    "scale Stratospheric water vapour",
    "scale Contrails",
    "scale Light absorbing particles on snow and ice",
    "scale Land use",
    "scale Volcanic",
    "solar_amplitude",
    "solar_trend",
    "scale CO2",
    "co2_concentration_1750",
    "seed",
]

params_out.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "calibrated_constrained_parameters.csv"
)
