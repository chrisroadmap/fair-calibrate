import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from fair21 import SpeciesID, SpeciesConfig, Category, Config, Species, RunMode, Scenario, ClimateResponse, FAIR
from fair21.defaults import species_config_from_default

species_ids = {
    'co2': SpeciesID('CO2', Category.CO2, run_mode=RunMode.CONCENTRATION),
    'ch4': SpeciesID('CH4', Category.CH4, run_mode=RunMode.CONCENTRATION),
    'n2o': SpeciesID('N2O', Category.N2O, run_mode=RunMode.CONCENTRATION),
}

scenario_species = ['CO2']
species_to_include = scenario_species #+ ['CH4', 'N2O']

scenarios_to_include = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585']
scenarios = []

df_conc = pd.read_csv('data/rcmip/rcmip-concentrations-annual-means-v5-1-0.csv')

for iscen, scenario in enumerate(scenarios_to_include):
    list_of_species = []
    for ispec, species in enumerate(scenario_species):
        species_rcmip_name = species.replace("-", "")
        conc_in = df_conc.loc[
            (df_conc['Scenario']==scenario) & (df_conc['Variable'].str.endswith("|"+species_rcmip_name)) & (df_conc['Region']=='World'), '1750':'2100'
        ].interpolate(axis=1).values.squeeze()

        list_of_species.append(Species(species_ids[species.lower()], concentration=conc_in))
    #list_of_species.append(Species(species_ids['ch4'], concentration=np.ones(351)*729.2))
    #list_of_species.append(Species(species_ids['n2o'], concentration=np.ones(351)*270.1))
    scenarios.append(Scenario(scenario, list_of_species))

df = pd.read_csv("data/calibration/4xCO2_cummins_ebm3.csv")
models = df['model'].unique()

params = {}

configs = []

for imodel, model in enumerate(models):
    for run in df.loc[df['model']==model, 'run']:
        condition = (df['model']==model) & (df['run']==run)
        config_name = f"{model}_{run}"
        climate_response = ClimateResponse(
            ocean_heat_capacity = df.loc[condition, 'C1':'C3'].values.squeeze(),
            ocean_heat_transfer = df.loc[condition, 'kappa1':'kappa3'].values.squeeze(),
            deep_ocean_efficacy = df.loc[condition, 'epsilon'].values[0],
            gamma_autocorrelation = df.loc[condition, 'gamma'].values[0],
            sigma_eta = df.loc[condition, 'sigma_eta'].values[0],
            sigma_xi = df.loc[condition, 'sigma_xi'].values[0],
            stochastic_run = True
            )
        # HUGE TODO is to make it easy to change one entry e.g. emissions to concentrations driven mode
        species_config = [
            SpeciesConfig(
                species_id = SpeciesID(name='CO2', category=Category.CO2, run_mode=RunMode.CONCENTRATION),
                molecular_weight = 44.009,
                lifetime = np.array([1e9, 394.4, 36.54, 4.304]),
                partition_fraction = np.array([0.2173, 0.2240, 0.2824, 0.2763]),
                radiative_efficiency = 1.3344985680386619e-05,
                iirf_0=29,
                iirf_airborne=0.000819,
                iirf_cumulative=0.00846,
                iirf_temperature=4.0,
                baseline_concentration = 278.3,
                tropospheric_adjustment = 0.05
            ),
            SpeciesConfig(
                species_id = SpeciesID('CH4', Category.CH4, run_mode=RunMode.CONCENTRATION),
                molecular_weight = 16.043,
                lifetime = 8.25,
                radiative_efficiency = 0.00038864402860869495,
                iirf_airborne = 0.00032,
                iirf_temperature = -0.3,
                baseline_concentration = 729.2,
                tropospheric_adjustment = -0.14,
                ozone_radiative_efficiency = 1.75e-4,
                h2o_stratospheric_factor = 0.091914639065882,
            ),
            SpeciesConfig(
                species_id = SpeciesID('N2O', Category.N2O, run_mode=RunMode.CONCENTRATION),
                molecular_weight = 44.013,
                lifetime = 109,
                radiative_efficiency = 0.00319550741640458,
                baseline_concentration = 270.1,
                tropospheric_adjustment = 0.07,
                ozone_radiative_efficiency = 7.1e-4,
            ),
        ]
        configs.append(Config(config_name, climate_response, species_config))

import time
start = time.time()
fair = FAIR(scenarios, configs, time=np.arange(1750.5, 2101))
fair.run()
end = time.time()
print (f"{len(scenarios) * len(configs)} ensemble members in {end - start}s.")

pl.plot(np.arange(1750.5, 2101), fair.concentration_array[:, :, 0, 0, 0])
pl.show()
pl.plot(np.arange(1750.5, 2101), fair.emissions_array[:, 7, :, 0, 0])
pl.show()
pl.plot(np.arange(1750.5, 2101), fair.temperature[:, 7, :, 0, 0])
pl.show()
