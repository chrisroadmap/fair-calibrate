#!/usr/bin/env python
# coding: utf-8

"""Make emissions binary file."""
#
# SSPs from RCMIP

import os
from fair import FAIR
from fair.interface import initialise, fill
from fair.io import read_properties

from fair import __version__
from dotenv import load_dotenv

load_dotenv()


cal_v = os.getenv('CALIBRATION_VERSION')
fair_v = os.getenv('FAIR_VERSION')
constraint_set = os.getenv('CONSTRAINT_SET')
samples = int(os.getenv("PRIOR_SAMPLES"))

assert fair_v == __version__

print("Making SSP concentration binary...")

scenarios = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585']

species, properties = read_properties()

ghgs = ['CO2', 'CH4', 'N2O', 'CFC-11', 'CFC-12', 'CFC-113', 'CFC-114', 'CFC-115', 'HCFC-22', 'HCFC-141b', 'HCFC-142b', 'CCl4', 'CHCl3', 'CH2Cl2', 'CH3Cl', 'CH3CCl3', 'CH3Br', 'Halon-1211', 'Halon-1301', 'Halon-2402', 'CF4', 'C2F6', 'C3F8', 'c-C4F8', 'C4F10', 'C5F12', 'C6F14', 'C7F16', 'C8F18', 'NF3', 'SF6', 'SO2F2', 'HFC-125', 'HFC-134a', 'HFC-143a', 'HFC-152a', 'HFC-227ea', 'HFC-23', 'HFC-236fa', 'HFC-245fa', 'HFC-32', 'HFC-365mfc', 'HFC-4310mee']

for specie in ghgs:
    properties[specie]['input_mode'] = 'concentration'

species.remove('CO2 FFI')
species.remove('CO2 AFOLU')
species.remove('Halon-1202')

f = FAIR(ch4_method='thornhill2021')
f.define_time(1750, 2500, 1)
f.define_configs(['unspecified'])
f.define_scenarios(scenarios)
f.define_species(species, properties)

f.allocate()
f.fill_from_rcmip()

os.makedirs(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/concentration/', exist_ok=True)

f.concentration.to_netcdf(f'../../../../../output/fair-{fair_v}/v{cal_v}/{constraint_set}/concentration/ssp_concentration_1750-2500.nc')
