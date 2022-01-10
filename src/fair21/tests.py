# this will eventally be split up and become the testing Module

from fair import *

# top level
species_ids = {
    'co2': SpeciesID('CO2', Category.CO2),
    'ch4': SpeciesID('CH4', Category.CH4),
    'n2o': SpeciesID('N2O', Category.N2O),
    'cfc_11': SpeciesID('CFC-11', Category.CFC_11),
    'hfc_134a': SpeciesID('HFC-134a', Category.F_GAS),
    'sulfur': SpeciesID('Sulfur', Category.SULFUR),
    'bc': SpeciesID('BC', Category.BC),
    'oc': SpeciesID('OC', Category.OC),
    'aci': SpeciesID('Aerosol-Cloud Interactions', Category.AEROSOL_CLOUD_INTERACTIONS),
    'solar': SpeciesID('Solar', Category.SOLAR),
    'volcanic': SpeciesID('Volcanic', Category.VOLCANIC)
}


scen_dict = {}
scen_dict['ssp119'] = [
    Species(species_ids['co2'], emissions=np.arange(100)),
    Species(species_ids['ch4'], emissions=np.ones(100)*300),
    Species(species_ids['n2o'], emissions=np.ones(100)*10),
    Species(species_ids['sulfur'], emissions=np.ones(100)*100),
    Species(species_ids['bc'], emissions=np.ones(100)*8),
    Species(species_ids['oc'], emissions=np.ones(100)*35),
    Species(species_ids['aci'])
]
scen_dict['ssp126'] = [
    Species(species_ids['co2'], emissions=np.arange(100)*2),
    Species(species_ids['ch4'], emissions=np.ones(100)*100),
    Species(species_ids['n2o'], emissions=np.ones(100)*10),
    Species(species_ids['sulfur'], emissions=np.arange(100)),
    Species(species_ids['bc'], emissions=np.arange(100)*.08),
    Species(species_ids['oc'], emissions=np.arange(100)*.35),
    Species(species_ids['aci'])
]
scen_dict['ssp245'] = [
    Species(species_ids['co2'], emissions=np.arange(100)*0.2),
    Species(species_ids['ch4'], emissions=np.zeros(100)),
    Species(species_ids['n2o'], emissions=np.zeros(100)),
    Species(species_ids['sulfur'], emissions=np.arange(100)),
    Species(species_ids['bc'], emissions=np.ones(100)*8),
    Species(species_ids['oc'], emissions=np.ones(100)*35),
    Species(species_ids['aci'])
]
scen_dict['ssp434'] = [
    Species(species_ids['co2'], emissions=np.arange(10)*0.2),
    Species(species_ids['ch4'], emissions=np.zeros(100)),
    Species(species_ids['n2o'], emissions=np.zeros(100)),
    Species(species_ids['sulfur'], emissions=np.arange(100)),
    Species(species_ids['aci'])
]
scen_dict['ssp585'] = [
    Species(species_ids['co2'], emissions=np.arange(100)*0.2),
    Species(species_ids['ch4'], emissions=np.zeros(100)),
    Species(species_ids['n2o'], emissions=np.zeros(100)),
    Species(species_ids['aci']),
]

try:
    Scenario('only CO2', [Species(species_ids['co2'], emissions=0)])
    print ("test failed")
except IncompatibleConfigError:
    print('test passed')

try:
    Scenario('duplicated CO2', [
        Species(species_ids['co2'], emissions=np.arange(10)*0.2),
        Species(species_ids['co2'], emissions=np.arange(10)*0.2),
        Species(species_ids['ch4'], emissions=np.zeros(100)),
        Species(species_ids['n2o'], emissions=np.zeros(100)),
    ])
    print('test failed')
except DuplicationError:
    print('test passed')


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
    species_config_from_default('sulfur'),
    species_config_from_default('bc'),
    species_config_from_default('oc'),
    species_config_from_default('aerosol-cloud interactions')
]

config_dict['MIROC6'] = {}
config_dict['MIROC6']['climate'] = ClimateResponse(ocean_heat_capacity = (2, 10, 80), ocean_heat_transfer=(1,2,3))

co2_gasprop2 = config_dict['UKESM']['species'][0]
co2_gasprop2.iirf_0 = 36

config_dict['MIROC6']['species'] = [
    co2_gasprop2,
    species_config_from_default('ch4'),
    species_config_from_default('n2o'),
    species_config_from_default('sulfur'),
    species_config_from_default('bc'),
    species_config_from_default('oc'),
    species_config_from_default('aerosol-cloud interactions', aci_params={"scale": 2.09841432, "Sulfur": 260.34644166, "BC+OC": 111.05064063})
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


try:
    FAIR(scenarios=scenarios, configs=configs, time=np.arange(10))
    print('test failed')
except TimeMismatchError:
    print('test passed')

try:
    fair = FAIR(scenarios=[scenarios[0],scenarios[1],scenarios[4]], configs=configs, time=np.arange(100))
    print('test failed')
except (SpeciesMismatchError):
    print("test passed")

try:
    fair = FAIR(scenarios=[scenarios[0],scenarios[1],scenarios[2]], configs=configs, time=np.arange(10))
    print("test failed")
except (SpeciesMismatchError):
    print("test passed")

try:
    fair = FAIR(
        scenarios=[scenarios[0],scenarios[1],scenarios[2]],
        configs=[configs[0],configs[1]],
        time=np.arange(10),
        run_config=RunConfig(aci_method = AciMethod.STEVENS2015)
    )
    fair.run()
    print("test failed")
except (TimeMismatchError):
    print("test passed")

try:
    fair=FAIR()
    fair.run()
    print("test failed")
except MissingInputError:
    print("test passed")


try:
    fair=FAIR(scenarios=[scenarios[0],scenarios[1], 'scen3'], configs=[configs[0],configs[1]], time=np.arange(10))
    print("test failed")
except TypeError:
    print ("test passed")

try:
    fair=FAIR(scenarios=[scenarios[0],scenarios[1],scenarios[2]], configs=[configs[0], 'config2'], time=np.arange(100))
    print("test failed")
except TypeError:
    print ("test passed")


# TODO: check temeprautre results are unchanged for the non-stochastic EBM with different gamma/xi.
print()

fair = FAIR(scenarios=[scenarios[0],scenarios[1],scenarios[2]], configs=[configs[0],configs[1]], time=np.arange(100))
fair.remove_scenario(scenarios[2])
fair.add_scenario(scenarios[2])
fair.remove_config(configs[0])
fair.add_config(configs[0])
fair.run()

print()
for iscen in range(3):
    print(fair.scenarios[iscen].temperature)
# this needs to be somehow processed out. And perhaps the best way is to use pyam or scmdata
#print(fair.scenarios[0].list_of_species[0]) #e.g the array of output is time x config
