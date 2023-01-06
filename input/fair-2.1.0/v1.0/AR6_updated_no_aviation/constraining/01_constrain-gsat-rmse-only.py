#!/usr/bin/env python
# coding: utf-8

"""First constraint: RMSE < 0.16 K"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import time
import scipy.stats
import scipy.optimize
from tqdm.auto import tqdm

from fair.energy_balance_model import EnergyBalanceModel
from fair import __version__

print("Doing RMSE constraint...")

cal_v = os.getenv('CALIBRATION_VERSION')
fair_v = os.getenv('FAIR_VERSION')
constraint_set = os.getenv('CONSTRAINT_SET')
samples = int(os.getenv("PRIOR_SAMPLES"))
plots = os.getenv("PLOTS", 'False').lower() in ('true', '1', 't')

assert fair_v == __version__

if plots:
    pl.rcParams['font.size'] = 16
    pl.rcParams['font.family'] = 'Arial'
    pl.rcParams['ytick.direction'] = 'in'
    pl.rcParams['ytick.minor.visible'] = True
    pl.rcParams['ytick.major.right'] = True
    pl.rcParams['ytick.right'] = True
    pl.rcParams['xtick.direction'] = 'in'
    pl.rcParams['xtick.minor.visible'] = True
    pl.rcParams['xtick.major.top'] = True
    pl.rcParams['xtick.top'] = True
    pl.rcParams['axes.spines.top'] = True
    pl.rcParams['axes.spines.bottom'] = True
    pl.rcParams['figure.dpi'] = 150
    os.makedirs(f'../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/', exist_ok=True)


temp_in = np.load(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_runs/temperature_1850-2101.npy')

df_gmst = pd.read_csv('../../../../../data/forcing/AR6_GMST.csv')
gmst = df_gmst['gmst'].values

def rmse(obs, mod):
    return np.sqrt(np.sum((obs-mod)**2)/len(obs))

weights = np.ones(52)
weights[0] = 0.5
weights[-1] = 0.5

rmse_temp = np.zeros((samples))

if plots:
    fig, ax = pl.subplots(figsize=(5, 5))
    ax.fill_between(
        np.arange(1850, 2102),
        np.min(temp_in-np.average(temp_in[:52, :], weights=weights, axis=0), axis=1),
        np.max(temp_in-np.average(temp_in[:52, :], weights=weights, axis=0), axis=1),
        color='#000000',
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850, 2102),
        np.percentile(temp_in-np.average(temp_in[:52, :], weights=weights, axis=0), 5, axis=1),
        np.percentile(temp_in-np.average(temp_in[:52, :], weights=weights, axis=0), 95, axis=1),
        color='#000000',
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850, 2102),
        np.percentile(temp_in-np.average(temp_in[:52, :], weights=weights, axis=0), 16, axis=1),
        np.percentile(temp_in-np.average(temp_in[:52, :], weights=weights, axis=0), 84, axis=1),
        color='#000000',
        alpha=0.2,
    )
    ax.plot(
        np.arange(1850, 2102),
        np.median(temp_in-np.average(temp_in[:52, :], weights=weights, axis=0), axis=1),
        color='#000000',
    )
    ax.plot(np.arange(1850.5, 2021), gmst, color='b')

    ax.set_xlim(1850,2100)
    ax.set_ylim(-1, 5)
    ax.set_ylabel('°C relative to 1850-1900')
    ax.axhline(0, color='k', ls=":", lw=0.5)
    pl.tight_layout()
    #pl.title('Temperature anomaly - unconstrained')
    pl.savefig(f'../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/prior_ssp245.png')
    pl.close()

for i in tqdm(range(samples)):
    rmse_temp[i] = rmse(gmst[:171], temp_in[:171,i]-np.average(temp_in[:52, i], weights=weights, axis=0))

accept_temp=(rmse_temp<0.16)
print("Passing RMSE constraint:", np.sum(accept_temp))
valid_temp = np.arange(samples, dtype=int)[accept_temp]

if plots:
    fig, ax = pl.subplots(figsize=(5,5))
    ax.fill_between(
        np.arange(1850, 2102),
        np.min(temp_in[:,accept_temp]-np.average(temp_in[:52, accept_temp], weights=weights, axis=0), axis=1),
        np.max(temp_in[:,accept_temp]-np.average(temp_in[:52, accept_temp], weights=weights, axis=0), axis=1),
        color='#000000',
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850.5, 2102),
        np.percentile(temp_in[:,accept_temp]-np.average(temp_in[:52, accept_temp], weights=weights, axis=0), 5, axis=1),
        np.percentile(temp_in[:,accept_temp]-np.average(temp_in[:52, accept_temp], weights=weights, axis=0), 95, axis=1),
        color='#000000',
        alpha=0.2,
    )
    ax.fill_between(
        np.arange(1850.5, 2102),
        np.percentile(temp_in[:,accept_temp]-np.average(temp_in[:52, accept_temp], weights=weights, axis=0), 16, axis=1),
        np.percentile(temp_in[:,accept_temp]-np.average(temp_in[:52, accept_temp], weights=weights, axis=0), 84, axis=1),
        color='#000000',
        alpha=0.2,
    )
    ax.plot(
        np.arange(1850.5, 2102),
        np.median(temp_in[:,accept_temp]-np.average(temp_in[:52, accept_temp], weights=weights, axis=0), axis=1),
        color='#000000',
    )

    ax.plot(np.arange(1850.5, 2021), gmst, color='b')

    ax.set_xlim(1850,2100)
    ax.set_ylim(-1, 5)
    ax.set_ylabel('°C relative to 1850-1900')
    ax.axhline(0, color='k', ls=":", lw=0.5)
    pl.tight_layout()
    pl.savefig(f'../../../../../plots/fair-{fair_v}/v{cal_v}/{constraint_set}/post_rsme_ssp245.png')
    pl.close()

valid_temp = np.arange(samples, dtype=int)[accept_temp]
os.makedirs(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors', exist_ok=True)
np.savetxt(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/posteriors/runids_rmse_pass.csv', valid_temp)
