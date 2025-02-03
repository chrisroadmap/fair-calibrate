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
)
valid_all

seed = 1355763 + 399 * valid_all
seed

# concatenate each param dataframe and prefix with its model element to avoid
# namespace conflicts (and a bit of user intuivity)

# this is where we create consistent v2.2 calibration files
print(df_cr.columns)
print(df_cc.columns)
print(df_ari.columns)
print(df_aci.columns)
print(df_ozone.columns)
print(df_scaling.columns)
print(df_1750co2.columns)

cr_renames = {
    "gamma": "gamma_autocorrelation",
    "c1": "ocean_heat_capacity[0]",
    "c2": "ocean_heat_capacity[1]",
    "c3": "ocean_heat_capacity[2]",
    "kappa1": "ocean_heat_transfer[0]",
    "kappa2": "ocean_heat_transfer[1]",
    "kappa3": "ocean_heat_transfer[2]",
    "epsilon": "deep_ocean_efficacy",
    "sigma_eta": "sigma_eta",
    "sigma_xi": "sigma_xi",
    "F_4xCO2": "forcing_4co2",
}
cc_renames = {
    "r0": "iirf_0[CO2]",
    "rU": "iirf_uptake[CO2]",
    "rT": "iirf_temperature[CO2]",
    "rA": "iirf_airborne[CO2]",
}
ari_renames = {
    "BC": "erfari_radiative_efficiency[BC]",
    "OC": "erfari_radiative_efficiency[OC]",
    "Sulfur": "erfari_radiative_efficiency[Sulfur]",
    "NOx": "erfari_radiative_efficiency[NOx]",
    "VOC": "erfari_radiative_efficiency[VOC]",
    "NH3": "erfari_radiative_efficiency[NH3]",
    "CH4": "erfari_radiative_efficiency[CH4]",
    "N2O": "erfari_radiative_efficiency[N2O]",
    "Equivalent effective stratospheric chlorine": "erfari_radiative_efficiency[Equivalent effective stratospheric chlorine]",
}
aci_renames = {
    "shape_so2": "aci_shape[Sulfur]",
    "shape_bc": "aci_shape[BC]",
    "shape_oc": "aci_shape[OC]",
    "beta": "aci_scale",
}
ozone_renames = {
    "CH4": "ozone_radiative_efficiency[CH4]",
    "N2O": "ozone_radiative_efficiency[N2O]",
    "Equivalent effective stratospheric chlorine": "ozone_radiative_efficiency[Equivalent effective stratospheric chlorine]",
    "CO": "ozone_radiative_efficiency[CO]",
    "VOC": "ozone_radiative_efficiency[VOC]",
    "NOx": "ozone_radiative_efficiency[NOx]"
}
scaling_renames = {
    "CH4": "forcing_scale[CH4]",
    "N2O": "forcing_scale[N2O]",
    "Stratospheric water vapour": "forcing_scale[Stratospheric water vapour]",
    "Land use": "forcing_scale[Land use]",
    "Volcanic": "forcing_scale[Volcanic]",
    "solar_amplitude": "forcing_scale[Solar]",
    "Light absorbing particles on snow and ice": "forcing_scale[Light absorbing particles on snow and ice]",
    "CO2": "forcing_scale[CO2]",
}

minor_ghgs = [
    "forcing_scale[CFC-11]",
    "forcing_scale[CFC-12]",
    "forcing_scale[CFC-113]",
    "forcing_scale[CFC-114]",
    "forcing_scale[CFC-115]",
    "forcing_scale[HCFC-22]",
    "forcing_scale[HCFC-141b]",
    "forcing_scale[HCFC-142b]",
    "forcing_scale[CCl4]",
    "forcing_scale[CHCl3]",
    "forcing_scale[CH2Cl2]",
    "forcing_scale[CH3Cl]",
    "forcing_scale[CH3CCl3]",
    "forcing_scale[CH3Br]",
    "forcing_scale[Halon-1202]",
    "forcing_scale[Halon-1211]",
    "forcing_scale[Halon-1301]",
    "forcing_scale[Halon-2402]",
    "forcing_scale[CF4]",
    "forcing_scale[C2F6]",
    "forcing_scale[C3F8]",
    "forcing_scale[c-C4F8]",
    "forcing_scale[C4F10]",
    "forcing_scale[C5F12]",
    "forcing_scale[C6F14]",
    "forcing_scale[C7F16]",
    "forcing_scale[C8F18]",
    "forcing_scale[NF3]",
    "forcing_scale[SF6]",
    "forcing_scale[SO2F2]",
    "forcing_scale[HFC-125]",
    "forcing_scale[HFC-134a]",
    "forcing_scale[HFC-143a]",
    "forcing_scale[HFC-152a]",
    "forcing_scale[HFC-227ea]",
    "forcing_scale[HFC-32]",
    "forcing_scale[HFC-365mfc]",
    "forcing_scale[HFC-4310mee]",
    "forcing_scale[HFC-23]",
    "forcing_scale[HFC-236fa]",
    "forcing_scale[HFC-245fa]",
]
# seed, stochastic_run = TRUE, use_seed = TRUE

params_out = pd.concat(
    (
        df_cr.loc[valid_all, :].rename(columns=cr_renames),
        df_cc.loc[valid_all, :].rename(columns=cc_renames),
        df_ari.loc[valid_all, :].rename(columns=ari_renames),
        df_aci.loc[valid_all, :].rename(columns=aci_renames),
        df_ozone.loc[valid_all, :].rename(columns=ozone_renames),
        df_scaling.loc[valid_all, :].rename(columns=scaling_renames),
        pd.DataFrame((df_scaling.loc[valid_all, "minorGHG"].values * np.ones((41, 1))).T, index=valid_all, columns=minor_ghgs),
        df_1750co2.loc[valid_all, :].rename(
            columns={"co2_concentration": "baseline_concentration[CO2]"}
        ),
        pd.DataFrame(seed, index=valid_all, columns=["seed"]),
        pd.DataFrame(True, index=valid_all, columns=["stochastic_run", "use_seed"])
    ),
    axis=1,
)

params_out.drop(columns=['minorGHG'], inplace=True)

params_out.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "calibrated_constrained_parameters.csv"
)
