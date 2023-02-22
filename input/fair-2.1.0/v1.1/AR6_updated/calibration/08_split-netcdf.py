#!/usr/bin/env python
# coding: utf-8

"""Split netCDF files into years."""

# why? this will make smaller memory requirements in the APRP process.

import glob
import os
import warnings

import numpy as np
import iris
import iris.coord_categorisation
import iris.analysis.cartography
from iris.util import equalise_attributes
from tqdm.auto import tqdm
from dotenv import load_dotenv

warnings.simplefilter('ignore')

load_dotenv()

print("Splitting netCDF files...")

cal_v = os.getenv("CALIBRATION_VERSION")
fair_v = os.getenv("FAIR_VERSION")
constraint_set = os.getenv("CONSTRAINT_SET")

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

# move away from dicts because of mutability issues
def rfmip():
    for run in tqdm(runs_piclim_control[model][0:1], desc='base run', leave=False):
        outdir = f"{datadir}/{model}/{run}/piClim-control/"
        os.makedirs(outdir, exist_ok=True)
        for var in tqdm(varlist, desc="variables", leave=False):
            base = iris.load(f"{datadir}/{var}_Amon_{model}_piClim-control_{run}_*.nc")
            equalise_attributes(base)
            base = base.concatenate()[0]
            iris.coord_categorisation.add_year(base, 'time')
            years = np.unique(base.coord("year").points)
            for year in years:
                con = iris.Constraint(year=year)
                base_1yr = base.extract(con)
                iris.save(base_1yr, f"{outdir}/{var}_Amon_{model}_piClim-control_{run}_{year}.nc")

    for run in tqdm(runs_piclim_histaer[model], desc='pert run', leave=False):
        outdir = f"{datadir}/{model}/{run}/piClim-histaer/"
        os.makedirs(outdir, exist_ok=True)
        for var in tqdm(varlist, desc="variables", leave=False):
            pert = iris.load(f"{datadir}/{var}_Amon_{model}_piClim-histaer_{run}_*.nc")
            equalise_attributes(pert)
            pert = pert.concatenate()[0]
            iris.coord_categorisation.add_year(pert, 'time')
            years = np.unique(pert.coord("year").points)
            for year in years:
                con = iris.Constraint(year=year)
                pert_1yr = pert.extract(con)
                iris.save(pert_1yr, f"{outdir}/{var}_Amon_{model}_piClim-histaer_{run}_{year}.nc")



def aerchemmip():
    for run in tqdm(runs_histsst_piaer[model][0:1], desc='base run', leave=False):
        outdir = f"{datadir}/{model}/{run}/histSST-piAer/"
        os.makedirs(outdir, exist_ok=True)
        for var in tqdm(varlist, desc="variables", leave=False):
            base = iris.load(f"{datadir}/{var}_Amon_{model}_histSST-piAer_{run}_*.nc")
            equalise_attributes(base)
            base = base.concatenate()[0]
            iris.coord_categorisation.add_year(base, 'time')
            years = np.unique(base.coord("year").points)
            for year in years:
                con = iris.Constraint(year=year)
                base_1yr = base.extract(con)
                iris.save(base_1yr, f"{outdir}/{var}_Amon_{model}_histSST-piAer_{run}_{year}.nc")

    for run in tqdm(runs_histsst[model], desc='run', leave=False):
        outdir = f"{datadir}/{model}/{run}/piClim-histaer/"
        os.makedirs(outdir, exist_ok=True)
        for var in tqdm(varlist, desc="variables", leave=False):
            pert = iris.load(f"{datadir}/{var}_Amon_{model}_histSST_{run}_*.nc")
            equalise_attributes(pert)
            pert = pert.concatenate()[0]
            iris.coord_categorisation.add_year(pert, 'time')
            years = np.unique(pert.coord("year").points)
            for year in years:
                con = iris.Constraint(year=year)
                pert_1yr = pert.extract(con)
                iris.save(pert_1yr, f"{outdir}/{var}_Amon_{model}_histSST_{run}_{year}.nc")


for model in tqdm(['UKESM1-0-LL', 'HadGEM3-GC31-LL'], desc='Models'):
    # if more than one control ensemble member, concatenate them
    if 'EC-Earth3' in model:
        continue
    if len(runs_piclim_control[model])>0:
        rfmip()
    else:
        aerchemmip()
