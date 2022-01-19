# this will eventally be split up and become the testing Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from fair21 import SpeciesID, Category, Config, Species, RunMode, Scenario, ClimateResponse, FAIR
from fair21.defaults import species_config_from_default

# top level
species_ids = {
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
    'solar': SpeciesID('Solar', Category.SOLAR),
    'volcanic': SpeciesID('Volcanic', Category.VOLCANIC)
}


# time for some verifiable scenarios
# this looks like it's case sensitive - yes it will be
emitted_species = [
    'CO2', 'CH4', 'N2O',
    'Sulfur', 'BC', 'OC', 'NH3', 'NOx', 'VOC', 'CO',
    'CFC-11', 'CFC-12', 'CFC-113', 'CFC-114', 'CFC-115',
    'CCl4', 'CHCl3', 'CH2Cl2', 'CH3Cl', 'CH3CCl3', 'CH3Br',
    'Halon-1211', 'Halon-1301', 'Halon-2402',
    'CF4', 'C2F6', 'C3F8', 'c-C4F8', 'C4F10', 'C5F12', 'C6F14', 'C7F16', 'C8F18',
    'NF3', 'SF6', 'SO2F2',
    'HFC-125', 'HFC-134a', 'HFC-143a', 'HFC-152a', 'HFC-227ea', 'HFC-23', 'HFC-236fa', 'HFC-245fa', 'HFC-32', 'HFC-365mfc',
    'HFC-4310mee']
species_to_include = emitted_species + ['aerosol-cloud interactions', 'ozone', 'contrails']
scenarios_to_include = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585']

scenarios = []

# grab some emissions
# TODO: RCMIP to fair converter
df = pd.read_csv('data/rcmip/rcmip-emissions-annual-means-v5-1-0.csv')
for iscen, scenario in enumerate(scenarios_to_include):
    list_of_species = []
    for ispec, species in enumerate(emitted_species):
        species_rcmip_name = species.replace("-", "")
        if species == 'Aviation NOx':
            species_rcmip_name = 'NOx|MAGICC Fossil and Industrial|Aircraft'
        emis_in = df.loc[
            (df['Scenario']==scenario) & (df['Variable'].str.endswith("|"+species_rcmip_name)) & (df['Region']=='World'), '1750':'2100'
        ].interpolate(axis=1).values.squeeze()

        # CO2 and N2O units need to behave: TODO, sort this out
        if species in ('CO2', 'N2O'):
            emis_in = emis_in / 1000
        list_of_species.append(Species(species_ids[species.lower()], emissions=emis_in))
    # declare that we want ACI and ozone
    list_of_species.append(Species(species_ids['aerosol-cloud interactions']))
    list_of_species.append(Species(species_ids['ozone']))
    list_of_species.append(Species(species_ids['contrails']))
    scenarios.append(Scenario(scenario, list_of_species))

df = pd.read_csv("data/calibration/4xCO2_cummins.csv")
models = df['model'].unique()

params = {}

configs = []
isample=0

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
# pl.plot(np.arange(1750.5, 2101), fair.temperature[:, 7, :, 0, 0])
# pl.show()
# pl.plot(np.arange(1750.5, 2101), fair.forcing_array[:, 7, :, -1, 0])
# pl.show()
pl.plot(np.arange(1750.5, 2101), fair.concentration_array[:, 2, :, 1, 0])
pl.show()


import sys
sys.exit()


# scen_dict = {}
# scen_dict['ssp119'] = [
#     Species(species_ids['co2'], emissions=np.arange(100)),
#     Species(species_ids['ch4'], emissions=np.ones(100)*300),
#     Species(species_ids['n2o'], emissions=np.ones(100)*10),
#     Species(species_ids['cfc-11'], emissions=np.ones(100)*100),
#     Species(species_ids['sulfur'], emissions=np.ones(100)*100),
#     Species(species_ids['bc'], emissions=np.ones(100)*8),
#     Species(species_ids['oc'], emissions=np.ones(100)*35),
#     Species(species_ids['aci']),
#     Species(species_ids['ozone']),
# ]
# scen_dict['ssp126'] = [
#     Species(species_ids['co2'], emissions=np.arange(100)*2),
#     Species(species_ids['ch4'], emissions=np.ones(100)*100),
#     Species(species_ids['n2o'], emissions=np.ones(100)*10),
#     Species(species_ids['cfc-11'], emissions=np.ones(100)*10),
#     Species(species_ids['sulfur'], emissions=np.arange(100)),
#     Species(species_ids['bc'], emissions=np.arange(100)*.08),
#     Species(species_ids['oc'], emissions=np.arange(100)*.35),
#     Species(species_ids['aci']),
#     Species(species_ids['ozone']),
# ]
# scen_dict['ssp245'] = [
#     Species(species_ids['co2'], emissions=np.arange(100)*0.2),
#     Species(species_ids['ch4'], emissions=np.zeros(100)),
#     Species(species_ids['n2o'], emissions=np.zeros(100)),
#     Species(species_ids['cfc-11'], emissions=np.zeros(100)),
#     Species(species_ids['sulfur'], emissions=np.arange(100)),
#     Species(species_ids['bc'], emissions=np.ones(100)*8),
#     Species(species_ids['oc'], emissions=np.ones(100)*35),
#     Species(species_ids['aci']),
#     Species(species_ids['ozone']),
# ]
# scen_dict['ssp434'] = [
#     Species(species_ids['co2'], emissions=np.arange(10)*0.2),
#     Species(species_ids['ch4'], emissions=np.zeros(100)),
#     Species(species_ids['n2o'], emissions=np.zeros(100)),
#     Species(species_ids['sulfur'], emissions=np.arange(100)),
#     Species(species_ids['aci'])
# ]
# scen_dict['ssp585'] = [
#     Species(species_ids['co2'], emissions=np.arange(100)*0.2),
#     Species(species_ids['ch4'], emissions=np.zeros(100)),
#     Species(species_ids['n2o'], emissions=np.zeros(100)),
#     Species(species_ids['aci']),
# ]
#
# try:
#     Scenario('only CO2', [Species(species_ids['co2'], emissions=0)])
#     print ("test failed")
# except IncompatibleConfigError:
#     print('test passed')
#
# try:
#     Scenario('duplicated CO2', [
#         Species(species_ids['co2'], emissions=np.arange(10)*0.2),
#         Species(species_ids['co2'], emissions=np.arange(10)*0.2),
#         Species(species_ids['ch4'], emissions=np.zeros(100)),
#         Species(species_ids['n2o'], emissions=np.zeros(100)),
#     ])
#     print('test failed')
# except DuplicationError:
#     print('test passed')


scenarios = [Scenario(scen, scen_dict[scen]) for scen in scen_dict]

config_dict = {}

#TODO: test with stochastic_run=True
config_dict['UKESM'] = {}
config_dict['UKESM']['climate'] = ClimateResponse(
    ocean_heat_capacity = (5, 20, 100), ocean_heat_transfer=(1,2,1), deep_ocean_efficacy=1.29
)
config_dict['UKESM']['species'] = [
    species_config_from_default('CO2', tropospheric_adjustment=0.04,  partition_fraction=[0.25, 0.25, 0.25, 0.25]),
    species_config_from_default('CH4'),
    species_config_from_default('N2O', tropospheric_adjustment=0),
    species_config_from_default('CFC-11'),
    species_config_from_default('sulfur'),
    species_config_from_default('bc'),
    species_config_from_default('oc'),
    species_config_from_default('aerosol-cloud interactions'),
    species_config_from_default('ozone', forcing_temperature_feedback=-0.037)
]

config_dict['MIROC6'] = {}
config_dict['MIROC6']['climate'] = ClimateResponse(ocean_heat_capacity = (2, 10, 80), ocean_heat_transfer=(1,2,3))

co2_gasprop2 = config_dict['UKESM']['species'][0]
co2_gasprop2.iirf_0 = 36
print(config_dict['UKESM']['species'][8])
config_dict['MIROC6']['species'] = [
    co2_gasprop2,
    species_config_from_default('ch4'),
    species_config_from_default('n2o'),
    species_config_from_default('CFC-11'),
    species_config_from_default('sulfur'),
    species_config_from_default('bc'),
    species_config_from_default('oc'),
    species_config_from_default('aerosol-cloud interactions', aci_params={"scale": 1.223, "Sulfur": 156.5, "BC+OC": 76.7}),
    species_config_from_default('ozone', forcing_temperature_feedback=-0.064)
]

config_dict['IPSL'] = {}
config_dict['IPSL']['climate'] = config_dict['UKESM']['climate']
config_dict['IPSL']['species'] = [
    co2_gasprop2,
    species_config_from_default('ch4'),
    species_config_from_default('n2o'),
    species_config_from_default('bc'),
]

config_dict['GISS'] = {}
config_dict['GISS']['climate'] = config_dict['MIROC6']['climate']
config_dict['GISS']['species'] = [
    co2_gasprop2,
    species_config_from_default('ch4'),
    species_config_from_default('n2o'),
    species_config_from_default('bc'),
]

configs = [Config(config, config_dict[config]['climate'], config_dict[config]['species']) for config in config_dict]


# try:
#     FAIR(scenarios=scenarios, configs=configs, time=np.arange(10))
#     print('test failed')
# except TimeMismatchError:
#     print('test passed')
#
# try:
#     fair = FAIR(scenarios=[scenarios[0],scenarios[1],scenarios[4]], configs=configs, time=np.arange(100))
#     print('test failed')
# except (SpeciesMismatchError):
#     print("test passed")
#
# try:
#     fair = FAIR(scenarios=[scenarios[0],scenarios[1],scenarios[2]], configs=configs, time=np.arange(10))
#     print("test failed")
# except (SpeciesMismatchError):
#     print("test passed")
#
# try:
#     fair = FAIR(
#         scenarios=[scenarios[0],scenarios[1],scenarios[2]],
#         configs=[configs[0],configs[1]],
#         time=np.arange(10),
#         run_config=RunConfig(aci_method = AciMethod.STEVENS2015)
#     )
#     fair.run()
#     print("test failed")
# except (TimeMismatchError):
#     print("test passed")
#
# try:
#     fair=FAIR()
#     fair.run()
#     print("test failed")
# except MissingInputError:
#     print("test passed")
#
#
# try:
#     fair=FAIR(scenarios=[scenarios[0],scenarios[1], 'scen3'], configs=[configs[0],configs[1]], time=np.arange(10))
#     print("test failed")
# except TypeError:
#     print ("test passed")
#
# try:
#     fair=FAIR(scenarios=[scenarios[0],scenarios[1],scenarios[2]], configs=[configs[0], 'config2'], time=np.arange(100))
#     print("test failed")
# except TypeError:
#     print ("test passed")
#
#
# # TODO: check temeprautre results are unchanged for the non-stochastic EBM with different gamma/xi.
# print()

fair = FAIR(scenarios=[scenarios[0],scenarios[1],scenarios[2]], configs=[configs[0],configs[1]], time=np.arange(100))
fair.remove_scenario(scenarios[2])
fair.add_scenario(scenarios[2])
fair.remove_config(configs[0])
fair.add_config(configs[0])
fair.run()

print()
for iscen in range(3):
    print(fair.scenarios[iscen].list_of_species[8].forcing)#.species_list[3].concentration)
    #print(fair.scenarios[iscen].temperature)
# this needs to be somehow processed out. And perhaps the best way is to use pyam or scmdata
#print(fair.scenarios[0].list_of_species[0]) #e.g the array of output is time x config
