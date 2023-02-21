#!/usr/bin/env python
# coding: utf-8

"""Get ERFari and ERFaci from climate model output."""

import glob
import os
import warnings

from climateforcing.aprp import aprp
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fair import __version__
import iris
import iris.coord_categorisation
import iris.analysis.cartography
from iris.util import equalise_attributes
from tqdm.auto import tqdm

warnings.simplefilter('ignore')

load_dotenv()

print("Calculating APRP breakdown...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")
plots = os.getenv("PLOTS", "False").lower() in ("true", "1", "t")
assert fair_v == __version__
pl.style.use("../../../../../defaults.mplstyle")

datadir = '../../../../../data/cmip6/'

model_list = []
for filepath in glob.glob(datadir + '*.nc'):
    model_list.append(filepath.split('/')[-1].split('_')[2])
models = list(set(model_list))

runs_piclim_control = {}
runs_piclim_histaer = {}
runs_histsst_piaer = {}
runs_histsst = {}
for model in models:
    runids_piclim_control_list = []
    runids_piclim_histaer_list = []
    runids_histsst_piaer_list = []
    runids_histsst_list = []
    for filepath in glob.glob(f'{datadir}*{model}*.nc'):
        expt = filepath.split('/')[-1].split('_')[3]
        runid = filepath.split('/')[-1].split('_')[4]
        if expt=='piClim-control':
            runids_piclim_control_list.append(runid)
        elif expt=='piClim-histaer':
            runids_piclim_histaer_list.append(runid)
        elif expt=='histSST-piAer':
            runids_histsst_piaer_list.append(runid)
        elif expt=='histSST':
            runids_histsst_list.append(runid)
    runs_piclim_control[model] = list(set(runids_piclim_control_list))
    runs_piclim_histaer[model] = list(set(runids_piclim_histaer_list))
    runs_histsst_piaer[model] = list(set(runids_histsst_piaer_list))
    runs_histsst[model] = list(set(runids_histsst_list))

print(runs_piclim_control)
print(runs_piclim_histaer)
print(runs_histsst_piaer)
print(runs_histsst)

# from email correspondence with Ron Miller 19 May 2020: use r?i1p1f2 from GISS.
# from email correspondence with Dirk Olivie 19 October 2020: use r?i1p2f1 from NorESM.

# RFMIP-style runs
for model in models:
    print(model)
    print()
    print('control: ')
    for run in runs_piclim_control[model]:
        print('-', run)
    print()
    print('histaer: ')
    for run in runs_piclim_histaer[model]:
        print('-', run)
    print()
    print()

varlist = [
            "rsdt",
            "rsus",
            "rsds",
            "clt",
            "rsdscs",
            "rsuscs",
            "rsut",
            "rsutcs",
            "rlut",
            "rlutcs",
        ]

longnames = {
    'rsdt': 'toa_incoming_shortwave_flux',
    'rlut': 'toa_outgoing_longwave_flux',
    'rsut': 'toa_outgoing_shortwave_flux',
    'rlutcs': 'toa_outgoing_longwave_flux_assuming_clear_sky',
    'rsds': 'surface_downwelling_shortwave_flux_in_air',
    'rsus': 'surface_upwelling_shortwave_flux_in_air',
    'rsutcs': 'toa_outgoing_shortwave_flux_assuming_clear_sky',
    'clt': 'cloud_area_fraction',
    'rsdscs': 'surface_downwelling_shortwave_flux_in_air_assuming_clear_sky',
    'rsuscs': 'surface_upwelling_shortwave_flux_in_air_assuming_clear_sky'
}

component_longnames = {
    'ERF' : 'Effective radiative forcing',
    'ERFariSW': 'Shortwave effective radiative forcing due to aerosol-radiation interactions',
    'ERFaciSW': 'Shortwave effective radiative forcing due to aerosol-cloud interactions',
    'albedo'  : 'Shortwave effective radiative forcing due to surface albedo',
    'ERFariLW': 'Longwave effective radiative forcing due to aerosol-radiation interactions',
    'ERFaciLW': 'Longwave effective radiative forcing due to aerosol-cloud interactions'
}


def rfmip():
    base = {}
    for run in runs_piclim_control[model][0:1]:
        for var in varlist:
            base[var] = iris.load(f"{datadir}/{var}_Amon_{model}_piClim-control_{run}_*.nc")
            equalise_attributes(base[var])
            base[var] = base[var].concatenate()[0]

    # calculate aprp for each ensemble member
    for run in tqdm(runs_piclim_histaer[model], desc='run', leave=False):
        outdir = f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/aerosol/{model}/{run}'
        os.makedirs(outdir, exist_ok=True)
        pert = {}
        for var in varlist:
            pert[var] = iris.load(f"{datadir}/{var}_Amon_{model}_piClim-histaer_{run}_*.nc")
            equalise_attributes(pert[var])
            pert[var] = pert[var].concatenate()[0]
        pert_monthlen=pert[var].coord('time').bounds[:,1] - pert[var].coord('time').bounds[:,0]

        nrpt = int(np.ceil(pert['rsdt'].shape[0] / base['rsdt'].shape[0]))
        nmonths_pert = pert['rsdt'].shape[0]
        nmonths_base = base['rsdt'].shape[0]
        nyears_pert = nmonths_pert//12
        nyears_base = nmonths_base//12
        nlat = base['rsdt'].shape[1]
        nlon = base['rsdt'].shape[2]

        outvars = ['ERFariSW', 'ERFaciSW', 'ERFariLW', 'ERFaciLW', 'albedo']
        results = {}
        interim = {}
        for var in outvars:
            results[var] = np.ones((nmonths_pert, nlat, nlon)) * np.nan
        results['ERF'] = np.ones((nmonths_pert, nlat, nlon)) * np.nan

        for i in tqdm(range(nyears_pert), desc='model years', leave=False):
            pert_slice = {key: value.data[i*12:i*12+12,...] for key, value in pert.items()}
            for var in outvars:
                interim[var] = np.ones((nyears_base, 12, nlat, nlon)) * np.nan
            interim['ERF'] = np.ones((nyears_base, 12, nlat, nlon)) * np.nan
            for j in tqdm(range(nyears_base), desc='base years', leave=False):
                base_slice = {key: value.data[j*12:j*12+12,...] for key, value in base.items()}
                interim['ERF'][j, ...] = (
                    (
                        pert_slice['rsdt'] - pert_slice['rsut'] - pert_slice['rlut']
                    ) - (
                        base_slice['rsdt'] - base_slice['rsut'] - base_slice['rlut']
                    )
                )
                aprp_output = aprp(base_slice, pert_slice, longwave=True)
                for var in outvars:
                    interim[var][j, ...] = aprp_output[var]
            for var in outvars + ['ERF']:
                results[var][12*i:12*i+12, ...] = np.mean(interim[var], axis=0)

        for component in results:
            cube = iris.cube.Cube(
                results[component],
                var_name = component,
                long_name = component_longnames[component],
                units = 'W m-2',
                dim_coords_and_dims=[(pert['rsdt'].coord('time'), 0), (pert['rsdt'].coord('latitude'), 1), (pert['rsdt'].coord('longitude'), 2)]
            )

            iris.coord_categorisation.add_year(cube, 'time')
            cube_year = cube.aggregated_by('year', iris.analysis.MEAN)
            if not cube_year.coord('latitude').has_bounds():
                cube_year.coord('latitude').guess_bounds()
            if not cube_year.coord('longitude').has_bounds():
                cube_year.coord('longitude').guess_bounds()
            grid_areas = iris.analysis.cartography.area_weights(cube_year)
            cube_gmym = cube_year.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
            iris.save(cube_gmym, f"{outdir}/{component}.nc")



def aerchemmip():
    base = {}
    for run in runs_histsst_piaer[model][0:1]:
        for var in varlist:
            base[var] = iris.load(f"{datadir}/{var}_Amon_{model}_histSST-piAer_{run}_*.nc")
            equalise_attributes(base[var])
            base[var] = base[var].concatenate()[0]

    # calculate aprp for each ensemble member
    for run in tqdm(runs_histsst[model], desc='run', leave=False):
        outdir = f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/aerosol/{model}/{run}'
        os.makedirs(outdir, exist_ok=True)
        pert = {}
        for var in varlist:
            pert[var] = iris.load(f"{datadir}/{var}_Amon_{model}_histSST_{run}_*.nc")
            equalise_attributes(pert[var])
            pert[var] = pert[var].concatenate()[0]
        pert_monthlen=pert[var].coord('time').bounds[:,1] - pert[var].coord('time').bounds[:,0]

        nmonths_pert = pert['rsdt'].shape[0]
        nyears_pert = nmonths_pert//12
        nlat = base['rsdt'].shape[1]
        nlon = base['rsdt'].shape[2]

        outvars = ['ERFariSW', 'ERFaciSW', 'ERFariLW', 'ERFaciLW', 'albedo']
        results = {}
        interim = {}
        for var in outvars:
            results[var] = np.ones((nmonths_pert, nlat, nlon)) * np.nan


        base_slice = {key: value.data for key, value in base.items()}
        pert_slice = {key: value.data for key, value in pert.items()}

        results['ERF'] = (
            (
                pert_slice['rsdt'] - pert_slice['rsut'] - pert_slice['rlut']
            ) - (
                base_slice['rsdt'] - base_slice['rsut'] - base_slice['rlut']
            )
        )

        aprp_output = aprp(base_slice, pert_slice, longwave=True)
        for var in outvars:
            results[var] = aprp_output[var]

        for component in results:
            cube = iris.cube.Cube(
                results[component],
                var_name = component,
                long_name = component_longnames[component],
                units = 'W m-2',
                dim_coords_and_dims=[(pert['rsdt'].coord('time'), 0), (pert['rsdt'].coord('latitude'), 1), (pert['rsdt'].coord('longitude'), 2)]
            )

            iris.coord_categorisation.add_year(cube, 'time')
            cube_year = cube.aggregated_by('year', iris.analysis.MEAN)
            if not cube_year.coord('latitude').has_bounds():
                cube_year.coord('latitude').guess_bounds()
            if not cube_year.coord('longitude').has_bounds():
                cube_year.coord('longitude').guess_bounds()
            grid_areas = iris.analysis.cartography.area_weights(cube_year)
            cube_gmym = cube_year.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
            iris.save(cube_gmym, f"{outdir}/{component}.nc")


for model in tqdm(['IPSL-CM6A-LR', 'GISS-E2-1-G', 'UKESM1-0-LL'], desc='Models'):
    # if more than one control ensemble member, concatenate them
    if len(runs_piclim_control[model])>0:
        rfmip()
    else:
        aerchemmip()
