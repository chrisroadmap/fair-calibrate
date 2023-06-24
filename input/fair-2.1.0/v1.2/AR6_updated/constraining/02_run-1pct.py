#!/usr/bin/env python
# coding: utf-8

"""Run 1pct CO2 concentration driven runs where RMSE passes"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fair import __version__
from parallel_1pct import run_fair
from utils import _parallel_process

if __name__ == "__main__":
    print("Running 1pctCO2 scenarios...")
    load_dotenv()

    cal_v = os.getenv("CALIBRATION_VERSION")
    fair_v = os.getenv("FAIR_VERSION")
    constraint_set = os.getenv("CONSTRAINT_SET")
    samples = int(os.getenv("PRIOR_SAMPLES"))
    batch_size = int(os.getenv("BATCH_SIZE"))
    WORKERS = int(os.getenv("WORKERS"))

    # number of processors
    WORKERS = min(multiprocessing.cpu_count(), WORKERS)

    assert fair_v == __version__

    df_cc = pd.read_csv(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
        "carbon_cycle.csv"
    )
    df_cr = pd.read_csv(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
        "climate_response_ebm3.csv"
    )
    df_scaling = pd.read_csv(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/priors/"
        "forcing_scaling.csv"
    )

    seedgen = 1355763
    seedstep = 399

    # we also only want to run ensembles that passed RMSE test
    rmse_pass = np.loadtxt(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/"
        "runids_rmse_pass.csv"
    ).astype(int)

    # we only care about temperature and airborne fraction in years 70 and 140
    temp_out = np.ones((2, len(rmse_pass))) * np.nan
    af_out = np.ones((2, len(rmse_pass))) * np.nan

    config = []
    for ibatch, batch_start in enumerate(range(0, len(rmse_pass), batch_size)):
        config.append({})
        batch_end = min(batch_start + batch_size, len(rmse_pass))
        config[ibatch]["batch_start"] = batch_start
        config[ibatch]["batch_end"] = batch_end
        config[ibatch]["c1"] = df_cr.loc[rmse_pass[batch_start:batch_end], "c1"].values
        config[ibatch]["c2"] = df_cr.loc[rmse_pass[batch_start:batch_end], "c2"].values
        config[ibatch]["c3"] = df_cr.loc[rmse_pass[batch_start:batch_end], "c3"].values
        config[ibatch]["kappa1"] = df_cr.loc[
            rmse_pass[batch_start:batch_end], "kappa1"
        ].values
        config[ibatch]["kappa2"] = df_cr.loc[
            rmse_pass[batch_start:batch_end], "kappa2"
        ].values
        config[ibatch]["kappa3"] = df_cr.loc[
            rmse_pass[batch_start:batch_end], "kappa3"
        ].values
        config[ibatch]["epsilon"] = df_cr.loc[
            rmse_pass[batch_start:batch_end], "epsilon"
        ].values
        config[ibatch]["gamma"] = df_cr.loc[
            rmse_pass[batch_start:batch_end], "gamma"
        ].values
        config[ibatch]["forcing_4co2"] = df_cr.loc[
            rmse_pass[batch_start:batch_end], "F_4xCO2"
        ]
        config[ibatch]["iirf_0"] = df_cc.loc[
            rmse_pass[batch_start:batch_end], "r0"
        ].values.squeeze()
        config[ibatch]["iirf_airborne"] = df_cc.loc[
            rmse_pass[batch_start:batch_end], "rA"
        ].values.squeeze()
        config[ibatch]["iirf_uptake"] = df_cc.loc[
            rmse_pass[batch_start:batch_end], "rU"
        ].values.squeeze()
        config[ibatch]["iirf_temperature"] = df_cc.loc[
            rmse_pass[batch_start:batch_end], "rT"
        ].values.squeeze()
        config[ibatch]["scaling_CO2"] = df_scaling.loc[
            rmse_pass[batch_start:batch_end], "CO2"
        ].values.squeeze()
        config[ibatch]["scaling_CH4"] = df_scaling.loc[
            rmse_pass[batch_start:batch_end], "CH4"
        ].values.squeeze()
        config[ibatch]["scaling_N2O"] = df_scaling.loc[
            rmse_pass[batch_start:batch_end], "N2O"
        ].values.squeeze()

    parallel_process_kwargs = dict(
        func=run_fair,
        configuration=config,
        config_are_kwargs=False,
    )

    with ProcessPoolExecutor(WORKERS) as pool:
        res = _parallel_process(
            **parallel_process_kwargs,
            pool=pool,
        )

    for ibatch, batch_start in enumerate(range(0, len(rmse_pass), batch_size)):
        batch_end = min(batch_start + batch_size, len(rmse_pass))
        temp_out[:, batch_start:batch_end] = res[ibatch][0]
        af_out[:, batch_start:batch_end] = res[ibatch][1]

    os.makedirs(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/",
        exist_ok=True,
    )
    np.save(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "temperature_1pctCO2_y70_y140.npy",
        temp_out,
        allow_pickle=True,
    )
    np.save(
        f"../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/"
        "airborne_fraction_1pctCO2_y70_y140.npy",
        af_out,
        allow_pickle=True,
    )
