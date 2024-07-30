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

seed = 1355763 + 399 * valid_all

# little choice but to do this manually and painfully
df_cr_rename = {
    'gamma':  'gamma_autocorrelation',
    'c1': 'ocean_heat_capacity[0]',
    'c2': 'ocean_heat_capacity[1]',
    'c3': 'ocean_heat_capacity[2]',
    'kappa1': 'ocean_heat_transfer[0]',
    'kappa2': 'ocean_heat_transfer[1]',
    'kappa3': 'ocean_heat_transfer[2]',
    'epsilon': 'deep_ocean_efficacy',
    'F_4xCO2': 'forcing_4co2'
}

df_cc_rename = {
    'r0': 'iirf_0[CO2]',
    'rU': 'iirf_uptake[CO2]',
    'rT': 'iirf_temperature[CO2]',
    'rA': 'iirf_airborne[CO2]'
}

df_ari_rename = {
    'BC': 'erfari_radiative_efficiency[BC]',
    'OC': 'erfari_radiative_efficiency[OC]',
    'Sulfur': 'erfari_radiative_efficiency[Sulfur]',
    'NOx': 'erfari_radiative_efficiency[NOx]',
    'VOC': 'erfari_radiative_efficiency[VOC]',
    'NH3': 'erfari_radiative_efficiency[NH3]',
    'CH4': 'erfari_radiative_efficiency[CH4]',
    'N2O': 'erfari_radiative_efficiency[N2O]'
}

df_aci_rename = {
    'shape_so2': 'aci_shape[Sulfur]',
    'shape_bc': 'aci_shape[BC]',
    'shape_oc': 'aci_shape[OC]',
    'beta': 'aci_scale'
}

df_ozone_rename = {
    'CH4': 'ozone_radiative_efficiency[CH4]',
    'N2O': 'ozone_radiative_efficiency[N2O]',
    'VOC': 'ozone_radiative_efficiency[VOC]',
    'NOx': 'ozone_radiative_efficiency[NOx]',
    'CO': 'ozone_radiative_efficiency[CO]'
}

df_scaling_rename = {
    'CH4': 'forcing_scale[CH4]',
    'N2O': 'forcing_scale[N2O]',
    'Stratospheric water vapour': 'forcing_scale[Stratospheric water vapour]',
    'Land use': 'forcing_scale[Land use]',
    'solar_amplitude': 'forcing_scale[Solar]',
    'Volcanic': 'forcing_scale[Volcanic]',
    'Light absorbing particles on snow and ice': 'forcing_scale[Light absorbing particles on snow and ice]',
    'CO2': 'forcing_scale[CO2]',
}

# concatenate each param dataframe and rename accoring to fair 2.2.0 conventions
params_out = pd.concat(
    (
        df_cr.loc[valid_all, :].rename(columns=df_cr_rename),
        df_cc.loc[valid_all, :].rename(columns=df_cc_rename),
        df_ari.loc[valid_all, :].rename(columns=df_ari_rename),
        df_aci.loc[valid_all, :].rename(columns=df_aci_rename),
        df_ozone.loc[valid_all, :].rename(columns=df_ozone_rename),
        df_scaling.loc[valid_all, :].rename(columns=df_scaling_rename),
        pd.DataFrame(
            df_scaling.loc[valid_all, 'minorGHG'].values[:, None] * np.ones((output_ensemble_size, 8)),
            index=valid_all,
            columns = [
                'forcing_scale[HFC-125]',
                'forcing_scale[HFC-134a]',
                'forcing_scale[HFC-143a]',
                'forcing_scale[HFC-227ea]',
                'forcing_scale[HFC-32]',
                'forcing_scale[HFC-4310mee]',
                'forcing_scale[HFC-245fa]',
                'forcing_scale[SF6]'
            ]
        ),
        df_1750co2.loc[valid_all, :].rename(
            columns={"co2_concentration": "baseline_concentration[CO2]"}
        ),
        pd.DataFrame(seed, index=valid_all, columns=["seed"]),
        pd.DataFrame([True] * np.ones((output_ensemble_size, 1)), index=valid_all, columns=["stochastic_run"]),
        pd.DataFrame([True] * np.ones((output_ensemble_size, 1)), index=valid_all, columns=["use_seed"]),
    ),
    axis=1,
)

params_out.drop(columns=['minorGHG'], inplace=True)

params_out.to_csv(
    f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
    "calibrated_constrained_parameters.csv"
)
