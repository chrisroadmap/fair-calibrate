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
from iris.util import equalise_attributes, unify_time_units
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

# from email correspondence with Ron Miller 19 May 2020: use r?i1p1f2 from GISS.
# from email correspondence with Dirk Olivie 19 October 2020: use r?i1p2f1 from NorESM.

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


# move away from dicts because of mutability issues
def rfmip():
    run = runs_piclim_control[model][0]
    clt_base = iris.load(f"{datadir}/clt_Amon_{model}_piClim-control_{run}_*.nc")
    equalise_attributes(clt_base)
    unify_time_units(clt_base)
    clt_base = clt_base.concatenate_cube()
    rsdt_base = iris.load(f"{datadir}/rsdt_Amon_{model}_piClim-control_{run}_*.nc")
    equalise_attributes(rsdt_base)
    unify_time_units(rsdt_base)
    rsdt_base = rsdt_base.concatenate_cube()
    rsus_base = iris.load(f"{datadir}/rsus_Amon_{model}_piClim-control_{run}_*.nc")
    equalise_attributes(rsus_base)
    unify_time_units(rsus_base)
    rsus_base = rsus_base.concatenate_cube()
    rsds_base = iris.load(f"{datadir}/rsds_Amon_{model}_piClim-control_{run}_*.nc")
    equalise_attributes(rsds_base)
    unify_time_units(rsds_base)
    rsds_base = rsds_base.concatenate_cube()
    rsdscs_base = iris.load(f"{datadir}/rsdscs_Amon_{model}_piClim-control_{run}_*.nc")
    equalise_attributes(rsdscs_base)
    unify_time_units(rsdscs_base)
    rsdscs_base = rsdscs_base.concatenate_cube()
    rsut_base = iris.load(f"{datadir}/rsut_Amon_{model}_piClim-control_{run}_*.nc")
    equalise_attributes(rsut_base)
    unify_time_units(rsut_base)
    rsut_base = rsut_base.concatenate_cube()
    rsutcs_base = iris.load(f"{datadir}/rsutcs_Amon_{model}_piClim-control_{run}_*.nc")
    equalise_attributes(rsutcs_base)
    unify_time_units(rsutcs_base)
    rsutcs_base = rsutcs_base.concatenate_cube()
    rlut_base = iris.load(f"{datadir}/rlut_Amon_{model}_piClim-control_{run}_*.nc")
    equalise_attributes(rlut_base)
    unify_time_units(rlut_base)
    rlut_base = rlut_base.concatenate_cube()
    rlutcs_base = iris.load(f"{datadir}/rlutcs_Amon_{model}_piClim-control_{run}_*.nc")
    equalise_attributes(rlutcs_base)
    unify_time_units(rlutcs_base)
    rlutcs_base = rlutcs_base.concatenate_cube()
    rsuscs_base = iris.load(f"{datadir}/rsuscs_Amon_{model}_piClim-control_{run}_*.nc")
    equalise_attributes(rsuscs_base)
    unify_time_units(rsuscs_base)
    rsuscs_base = rsuscs_base.concatenate_cube()

    # calculate aprp for each ensemble member
    for run in tqdm(runs_piclim_histaer[model], desc='run', leave=False):
        outdir = f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/aerosol/{model}/{run}'
        os.makedirs(outdir, exist_ok=True)
        clt_pert = iris.load(f"{datadir}/clt_Amon_{model}_piClim-histaer_{run}_*.nc")
        equalise_attributes(clt_pert)
        unify_time_units(clt_pert)
        clt_pert = clt_pert.concatenate_cube()
        rsdt_pert = iris.load(f"{datadir}/rsdt_Amon_{model}_piClim-histaer_{run}_*.nc")
        equalise_attributes(rsdt_pert)
        unify_time_units(rsdt_pert)
        rsdt_pert = rsdt_pert.concatenate_cube()
        rsus_pert = iris.load(f"{datadir}/rsus_Amon_{model}_piClim-histaer_{run}_*.nc")
        equalise_attributes(rsus_pert)
        unify_time_units(rsus_pert)
        rsus_pert = rsus_pert.concatenate_cube()
        rsds_pert = iris.load(f"{datadir}/rsds_Amon_{model}_piClim-histaer_{run}_*.nc")
        equalise_attributes(rsds_pert)
        unify_time_units(rsds_pert)
        rsds_pert = rsds_pert.concatenate_cube()
        rsdscs_pert = iris.load(f"{datadir}/rsdscs_Amon_{model}_piClim-histaer_{run}_*.nc")
        equalise_attributes(rsdscs_pert)
        unify_time_units(rsdscs_pert)
        rsdscs_pert = rsdscs_pert.concatenate_cube()
        rsut_pert = iris.load(f"{datadir}/rsut_Amon_{model}_piClim-histaer_{run}_*.nc")
        equalise_attributes(rsut_pert)
        unify_time_units(rsut_pert)
        rsut_pert = rsut_pert.concatenate_cube()
        rsutcs_pert = iris.load(f"{datadir}/rsutcs_Amon_{model}_piClim-histaer_{run}_*.nc")
        equalise_attributes(rsutcs_pert)
        unify_time_units(rsutcs_pert)
        rsutcs_pert = rsutcs_pert.concatenate_cube()
        rlut_pert = iris.load(f"{datadir}/rlut_Amon_{model}_piClim-histaer_{run}_*.nc")
        equalise_attributes(rlut_pert)
        unify_time_units(rlut_pert)
        rlut_pert = rlut_pert.concatenate_cube()
        rlutcs_pert = iris.load(f"{datadir}/rlutcs_Amon_{model}_piClim-histaer_{run}_*.nc")
        equalise_attributes(rlutcs_pert)
        unify_time_units(rlutcs_pert)
        rlutcs_pert = rlutcs_pert.concatenate_cube()
        rsuscs_pert = iris.load(f"{datadir}/rsuscs_Amon_{model}_piClim-histaer_{run}_*.nc")
        equalise_attributes(rsuscs_pert)
        unify_time_units(rsuscs_pert)
        rsuscs_pert = rsuscs_pert.concatenate_cube()

        pert_nmonths = rsdt_pert.shape[0]
        base_nmonths = rsdt_base.shape[0]
        pert_nyears = pert_nmonths//12
        base_nyears = base_nmonths//12
        nlat = rsdt_base.shape[1]
        nlon = rsdt_base.shape[2]

        outvars = ['ERFariSW', 'ERFaciSW', 'ERFariLW', 'ERFaciLW', 'albedo']
        results = {}
        for var in outvars:
            results[var] = np.ones((pert_nmonths, nlat, nlon)) * np.nan
        results['ERF'] = np.ones((pert_nmonths, nlat, nlon)) * np.nan

        for i in tqdm(range(pert_nyears), desc='model years', leave=False):
            interim = {}
            for var in outvars:
                interim[var] = np.ones((base_nyears, 12, nlat, nlon)) * np.nan
            interim['ERF'] = np.ones((base_nyears, 12, nlat, nlon)) * np.nan
            for j in tqdm(range(base_nyears), desc='base years', leave=False):
                base_slice = {
                    "clt": clt_base.data[j*12:j*12+12,...],
                    "rsdt": rsdt_base.data[j*12:j*12+12,...],
                    "rsus": rsus_base.data[j*12:j*12+12,...],
                    "rsds": rsds_base.data[j*12:j*12+12,...],
                    "rsdscs": rsdscs_base.data[j*12:j*12+12,...],
                    "rsut": rsut_base.data[j*12:j*12+12,...],
                    "rsutcs": rsutcs_base.data[j*12:j*12+12,...],
                    "rlut": rlut_base.data[j*12:j*12+12,...],
                    "rlutcs": rlutcs_base.data[j*12:j*12+12,...],
                    "rsuscs": rsuscs_base.data[j*12:j*12+12,...],
                }
                pert_slice = {
                    "clt": clt_pert.data[i*12:i*12+12,...],
                    "rsdt": rsdt_pert.data[i*12:i*12+12,...],
                    "rsus": rsus_pert.data[i*12:i*12+12,...],
                    "rsds": rsds_pert.data[i*12:i*12+12,...],
                    "rsdscs": rsdscs_pert.data[i*12:i*12+12,...],
                    "rsut": rsut_pert.data[i*12:i*12+12,...],
                    "rsutcs": rsutcs_pert.data[i*12:i*12+12,...],
                    "rlut": rlut_pert.data[i*12:i*12+12,...],
                    "rlutcs": rlutcs_pert.data[i*12:i*12+12,...],
                    "rsuscs": rsuscs_pert.data[i*12:i*12+12,...],
                }
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
                dim_coords_and_dims=[(rsdt_pert.coord('time'), 0), (rsdt_pert.coord('latitude'), 1), (rsdt_pert.coord('longitude'), 2)]
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
    run = runs_histsst_piaer[model][0]
    clt_base = iris.load(f"{datadir}/clt_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(clt_base)
    unify_time_units(clt_base)
    clt_base = clt_base.concatenate_cube()
    rsdt_base = iris.load(f"{datadir}/rsdt_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rsdt_base)
    unify_time_units(rsdt_base)
    rsdt_base = rsdt_base.concatenate_cube()
    rsus_base = iris.load(f"{datadir}/rsus_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rsus_base)
    unify_time_units(rsus_base)
    rsus_base = rsus_base.concatenate_cube()
    rsds_base = iris.load(f"{datadir}/rsds_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rsds_base)
    unify_time_units(rsds_base)
    rsds_base = rsds_base.concatenate_cube()
    rsdscs_base = iris.load(f"{datadir}/rsdscs_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rsdscs_base)
    unify_time_units(rsdscs_base)
    rsdscs_base = rsdscs_base.concatenate_cube()
    rsut_base = iris.load(f"{datadir}/rsut_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rsut_base)
    unify_time_units(rsut_base)
    rsut_base = rsut_base.concatenate_cube()
    rsutcs_base = iris.load(f"{datadir}/rsutcs_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rsutcs_base)
    unify_time_units(rsutcs_base)
    rsutcs_base = rsutcs_base.concatenate_cube()
    rlut_base = iris.load(f"{datadir}/rlut_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rlut_base)
    unify_time_units(rlut_base)
    rlut_base = rlut_base.concatenate_cube()
    rlutcs_base = iris.load(f"{datadir}/rlutcs_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rlutcs_base)
    unify_time_units(rlutcs_base)
    rlutcs_base = rlutcs_base.concatenate_cube()
    rsuscs_base = iris.load(f"{datadir}/rsuscs_Amon_{model}_histSST-piAer_{run}_*.nc")
    equalise_attributes(rsuscs_base)
    unify_time_units(rsuscs_base)
    rsuscs_base = rsuscs_base.concatenate_cube()

    # calculate aprp for each ensemble member
    for run in tqdm(runs_histsst[model], desc='run', leave=False):
        outdir = f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/calibrations/aerosol/{model}/{run}'
        os.makedirs(outdir, exist_ok=True)
        clt_pert = iris.load(f"{datadir}/clt_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(clt_pert)
        unify_time_units(clt_pert)
        clt_pert = clt_pert.concatenate_cube()
        rsdt_pert = iris.load(f"{datadir}/rsdt_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rsdt_pert)
        unify_time_units(rsdt_pert)
        rsdt_pert = rsdt_pert.concatenate_cube()
        rsus_pert = iris.load(f"{datadir}/rsus_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rsus_pert)
        unify_time_units(rsus_pert)
        rsus_pert = rsus_pert.concatenate_cube()
        rsds_pert = iris.load(f"{datadir}/rsds_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rsds_pert)
        unify_time_units(rsds_pert)
        rsds_pert = rsds_pert.concatenate_cube()
        rsdscs_pert = iris.load(f"{datadir}/rsdscs_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rsdscs_pert)
        unify_time_units(rsdscs_pert)
        rsdscs_pert = rsdscs_pert.concatenate_cube()
        rsut_pert = iris.load(f"{datadir}/rsut_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rsut_pert)
        unify_time_units(rsut_pert)
        rsut_pert = rsut_pert.concatenate_cube()
        rsutcs_pert = iris.load(f"{datadir}/rsutcs_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rsutcs_pert)
        unify_time_units(rsutcs_pert)
        rsutcs_pert = rsutcs_pert.concatenate_cube()
        rlut_pert = iris.load(f"{datadir}/rlut_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rlut_pert)
        unify_time_units(rlut_pert)
        rlut_pert = rlut_pert.concatenate_cube()
        rlutcs_pert = iris.load(f"{datadir}/rlutcs_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rlutcs_pert)
        unify_time_units(rlutcs_pert)
        rlutcs_pert = rlutcs_pert.concatenate_cube()
        rsuscs_pert = iris.load(f"{datadir}/rsuscs_Amon_{model}_histSST_{run}_*.nc")
        equalise_attributes(rsuscs_pert)
        unify_time_units(rsuscs_pert)
        rsuscs_pert = rsuscs_pert.concatenate_cube()

        pert_nmonths = rsdt_pert.shape[0]
        pert_nyears = pert_nmonths//12
        nlat = rsdt_base.shape[1]
        nlon = rsdt_base.shape[2]

        outvars = ['ERFariSW', 'ERFaciSW', 'ERFariLW', 'ERFaciLW', 'albedo']
        results = {}

        for var in outvars:
            results[var] = np.ones((pert_nmonths, nlat, nlon)) * np.nan

        base_slice = {
            "clt": clt_base.data,
            "rsdt": rsdt_base.data,
            "rsus": rsus_base.data,
            "rsds": rsds_base.data,
            "rsdscs": rsdscs_base.data,
            "rsut": rsut_base.data,
            "rsutcs": rsutcs_base.data,
            "rlut": rlut_base.data,
            "rlutcs": rlutcs_base.data,
            "rsuscs": rsuscs_base.data,
        }
        pert_slice = {
            "clt": clt_pert.data,
            "rsdt": rsdt_pert.data,
            "rsus": rsus_pert.data,
            "rsds": rsds_pert.data,
            "rsdscs": rsdscs_pert.data,
            "rsut": rsut_pert.data,
            "rsutcs": rsutcs_pert.data,
            "rlut": rlut_pert.data,
            "rlutcs": rlutcs_pert.data,
            "rsuscs": rsuscs_pert.data,
        }

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
                dim_coords_and_dims=[(rsdt_pert.coord('time'), 0), (rsdt_pert.coord('latitude'), 1), (rsdt_pert.coord('longitude'), 2)]
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


for model in tqdm(["MRI-ESM2-0", "GFDL-ESM4", "GFDL-CM4", "EC-Earth3"], desc='Models'):
    # what to do about ec-earth, which is a huge model?
    # GFDL-ESM4 seemed to struggle
    # I think MRI-ESM would too
    if len(runs_piclim_control[model])>0:
        rfmip()
    else:
        aerchemmip()
