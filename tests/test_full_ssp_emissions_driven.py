# this will eventally be split up and become the testing Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from fair21 import SpeciesID, Category, Config, Species, RunMode, Scenario, ClimateResponse, FAIR
from fair21.defaults import species_config_from_default

# top level
species_ids = {
    'co2_ffi': SpeciesID('CO2 fossil fuel and industrial', Category.CO2_FFI),
    'co2_afolu': SpeciesID('CO2 AFOLU', Category.CO2_AFOLU),
    'co2': SpeciesID('CO2', Category.CO2),
    'ch4': SpeciesID('CH4', Category.CH4),
    'n2o': SpeciesID('N2O', Category.N2O),
    'cfc-11': SpeciesID('CFC-11', Category.CFC_11),
    'cfc-12': SpeciesID('CFC-12', Category.OTHER_HALOGEN),
    'cfc-113': SpeciesID('CFC-113', Category.OTHER_HALOGEN),
    'cfc-114': SpeciesID('CFC-114', Category.OTHER_HALOGEN),
    'cfc-115': SpeciesID('CFC-115', Category.OTHER_HALOGEN),
    'ccl4': SpeciesID('CCl4', Category.OTHER_HALOGEN),
    'chcl3': SpeciesID('CHCl3', Category.OTHER_HALOGEN),
    'ch2cl2': SpeciesID('CH2Cl2', Category.OTHER_HALOGEN),
    'ch3cl': SpeciesID('CH3Cl', Category.OTHER_HALOGEN),
    'ch3ccl3': SpeciesID('CH3CCl3', Category.OTHER_HALOGEN),
    'ch3br': SpeciesID('CH3Br', Category.OTHER_HALOGEN),
    'halon-1211': SpeciesID('Halon-1211', Category.OTHER_HALOGEN),
    'halon-1301': SpeciesID('Halon-1301', Category.OTHER_HALOGEN),
    'halon-2402': SpeciesID('Halon-2402', Category.OTHER_HALOGEN),
    'cf4': SpeciesID('CF4', Category.F_GAS),
    'c2f6': SpeciesID('C2F6', Category.F_GAS),
    'c3f8': SpeciesID('C3F8', Category.F_GAS),
    'c-c4f8': SpeciesID('C-C4F8', Category.F_GAS),
    'c4f10': SpeciesID('C4F10', Category.F_GAS),
    'c5f12': SpeciesID('C5F12', Category.F_GAS),
    'c6f14': SpeciesID('C6F14', Category.F_GAS),
    'c7f16': SpeciesID('C7F16', Category.F_GAS),
    'c8f18': SpeciesID('C8F18', Category.F_GAS),
    'hfc-125': SpeciesID('HFC-125', Category.F_GAS),
    'hfc-134a': SpeciesID('HFC-134a', Category.F_GAS),
    'hfc-143a': SpeciesID('HFC-143a', Category.F_GAS),
    'hfc-152a': SpeciesID('HFC-152a', Category.F_GAS),
    'hfc-227ea': SpeciesID('HFC-227ea', Category.F_GAS),
    'hfc-23': SpeciesID('HFC-23', Category.F_GAS),
    'hfc-236fa': SpeciesID('HFC-236fa', Category.F_GAS),
    'hfc-245fa': SpeciesID('HFC-245fa', Category.F_GAS),
    'hfc-32': SpeciesID('HFC-32', Category.F_GAS),
    'hfc-365mfc': SpeciesID('HFC-365mfc', Category.F_GAS),
    'hfc-4310mee': SpeciesID('HFC-4310mee', Category.F_GAS),
    'nf3': SpeciesID('NF3', Category.F_GAS),
    'sf6': SpeciesID('SF6', Category.F_GAS),
    'so2f2': SpeciesID('SO2F2', Category.F_GAS),
    'ozone': SpeciesID('Ozone', Category.OZONE),
    'sulfur': SpeciesID('Sulfur', Category.SULFUR),
    'bc': SpeciesID('BC', Category.BC),
    'oc': SpeciesID('OC', Category.OC),
    'nh3': SpeciesID('NH3', Category.OTHER_AEROSOL),
    'voc': SpeciesID('VOC', Category.SLCF_OZONE_PRECURSOR),
    'co': SpeciesID('CO', Category.SLCF_OZONE_PRECURSOR),
    'nox': SpeciesID('NOx', Category.SLCF_OZONE_PRECURSOR),
    'aviation nox': SpeciesID('Aviation NOx', Category.AVIATION_NOX),
    'contrails': SpeciesID('Contrails', Category.CONTRAILS),
    'aerosol-cloud interactions': SpeciesID('Aerosol-Cloud Interactions', Category.AEROSOL_CLOUD_INTERACTIONS),
    'lapsi': SpeciesID('Light absorbing particles on snow and ice', Category.LAPSI),
    'h2o stratospheric': SpeciesID('H2O Stratospheric', Category.H2O_STRATOSPHERIC),
    'land use': SpeciesID('Land Use', Category.LAND_USE),
    'solar': SpeciesID('Solar', Category.SOLAR),
    'volcanic': SpeciesID('Volcanic', Category.VOLCANIC)
}


# time for some verifiable scenarios
# this looks like it's case sensitive - yes it will be
emitted_species = [
    'CO2_FFI', 'CO2_AFOLU', 'CH4', 'N2O',
    'Sulfur', 'BC', 'OC', 'NH3', 'NOx', 'VOC', 'CO',
    'CFC-11', 'CFC-12', 'CFC-113', 'CFC-114', 'CFC-115',
    'CCl4', 'CHCl3', 'CH2Cl2', 'CH3Cl', 'CH3CCl3', 'CH3Br',
    'Halon-1211', 'Halon-1301', 'Halon-2402',
    'CF4', 'C2F6', 'C3F8', 'c-C4F8', 'C4F10', 'C5F12', 'C6F14', 'C7F16', 'C8F18',
    'NF3', 'SF6', 'SO2F2',
    'HFC-125', 'HFC-134a', 'HFC-143a', 'HFC-152a', 'HFC-227ea', 'HFC-23', 'HFC-236fa', 'HFC-245fa', 'HFC-32', 'HFC-365mfc',
    'HFC-4310mee', 'Aviation NOx']
species_to_include = emitted_species + [
    'co2',
    'aerosol-cloud interactions',
    'ozone',
    'contrails',
    'light absorbing particles on snow and ice',
    'h2o stratospheric',
    'land use',
    'solar',
    'volcanic'
]
scenarios_to_include = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585']

scenarios = []

# grab some emissions
# TODO: RCMIP to fair converter - something that is smart about dates, and can interpolate
# including to sub-annual timesteps
df_emis = pd.read_csv('data/rcmip/rcmip-emissions-annual-means-v5-1-0.csv')
df_forc = pd.read_csv('data/forcing/table_A3.3_historical_ERF_1750-2019_best_estimate.csv')

for iscen, scenario in enumerate(scenarios_to_include):
    list_of_species = []
    for ispec, species in enumerate(emitted_species):
        species_rcmip_name = species.replace("-", "")
        if species == 'Aviation NOx':
            species_rcmip_name = 'NOx|MAGICC Fossil and Industrial|Aircraft'
        elif species == 'CO2_FFI':
            species_rcmip_name = 'CO2|MAGICC Fossil and Industrial'
        elif species == 'CO2_AFOLU':
            species_rcmip_name = 'CO2|MAGICC AFOLU'
        emis_in = df_emis.loc[
            (df_emis['Scenario']==scenario) & (df_emis['Variable'].str.endswith("|"+species_rcmip_name)) & (df_emis['Region']=='World'), '1750':'2100'
        ].interpolate(axis=1).values.squeeze()

        # CO2 and N2O units need to behave: TODO, sort this out
        if species in ('CO2_FFI', 'CO2_AFOLU', 'N2O'):
            emis_in = emis_in / 1000
        list_of_species.append(Species(species_ids[species.lower()], emissions=emis_in))
    # declare that we want ACI and ozone
    list_of_species.append(Species(species_ids['co2']))
    list_of_species.append(Species(species_ids['aerosol-cloud interactions']))
    list_of_species.append(Species(species_ids['ozone']))
    list_of_species.append(Species(species_ids['contrails']))
    list_of_species.append(Species(species_ids['lapsi']))
    list_of_species.append(Species(species_ids['h2o stratospheric']))
    list_of_species.append(Species(species_ids['land use']))
    scenarios.append(Scenario(scenario, list_of_species))

    # this is something that really needs improving compared to fair1.6
    solar_forcing = np.zeros(351)
    solar_forcing[:270] = df_forc['solar'].values
    volcanic_forcing = np.zeros(351)
    volcanic_forcing[:270] = df_forc['volcanic'].values
    list_of_species.append(Species(species_ids['solar'], forcing=solar_forcing))
    list_of_species.append(Species(species_ids['volcanic'], forcing=volcanic_forcing))

df = pd.read_csv("data/calibration/4xCO2_cummins.csv")
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
        species_config = [species_config_from_default(species) for species in species_to_include]
        configs.append(Config(config_name, climate_response, species_config))

import time
start = time.time()
fair = FAIR(scenarios, configs, time=np.arange(1750.5, 2101))
fair.run()
end = time.time()
print (f"{len(scenarios) * len(configs)} ensemble members in {end - start}s.")

import matplotlib.pyplot as pl
pl.plot(np.arange(1750.5, 2101), fair.temperature[:, 0, :, 0, 0])
pl.show()
pl.plot(np.arange(1750.5, 2101), fair.forcing_array[:, 0, :, 57, 0])
pl.show()
pl.plot(np.arange(1750.5, 2101), fair.concentration_array[:, 2, :, 2, 0])
pl.show()
pl.plot(np.arange(1750.5, 2101), fair.concentration_array[:, 2, :, 49, 0])
pl.show()

print(fair.forcing_array[-1, 7, 0, :, 0])
