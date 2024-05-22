#!/usr/bin/env python
# coding: utf-8

"""Takes constrained runs and dumps parameters into the output file"""

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fair import __version__

load_dotenv(override=True)

print("Dumping output...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
samples = int(os.getenv("PRIOR_SAMPLES"))
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))

assert fair_v == __version__

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

# concatenate each param dataframe and prefix with its model element to avoid
# namespace conflicts (and a bit of user intuivity)
params_out = pd.concat(
    (
        df_cr.loc[valid_all, :].rename(columns=lambda x: "clim_" + x),
        df_cc.loc[valid_all, :].rename(columns=lambda x: "cc_" + x),
        df_ari.loc[valid_all, :].rename(columns=lambda x: "ari_" + x),
        df_aci.loc[valid_all, :].rename(columns=lambda x: "aci_" + x),
        df_ozone.loc[valid_all, :].rename(columns=lambda x: "o3_" + x),
        df_scaling.loc[valid_all, :].rename(columns=lambda x: "fscale_" + x),
        df_1750co2.loc[valid_all, :].rename(
            columns={"co2_concentration": "cc_co2_concentration_1750"}
        ),
        pd.DataFrame(seed, index=valid_all, columns=["seed"]),
    ),
    axis=1,
)

params_out.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "calibrated_constrained_parameters.csv"
)
