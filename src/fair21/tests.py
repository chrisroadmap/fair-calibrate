# this will eventally be split up and become the testing Module

from fair import *

cr1 = ClimateResponse(ocean_heat_capacity = (5, 20, 100), ocean_heat_transfer=(1,2,1), deep_ocean_efficacy=1.29)
cr2 = ClimateResponse(ocean_heat_capacity = (2, 10, 80), ocean_heat_transfer=(1,2,3))

co2 = Species(name="CO2", category=Category.GREENHOUSE_GAS)
ch4 = Species("CH4", Category.GREENHOUSE_GAS)
so2 = Species("Sulfur", Category.AEROSOL)
bc = Species("BC", Category.AEROSOL)
aci = Species("Aerosol-Cloud Interactions", Category.AEROSOL_CLOUD_INTERACTIONS)

co2_cfg1 = SpeciesConfig(
    co2,
    lifetime=[1e9, 394, 36, 4.3],
    partition_fraction=[0.25, 0.25, 0.25, 0.25],
    radiative_efficiency=1.33e-5,
    molecular_weight=44.009,
    scale=1.04
)

co2_cfg2 = SpeciesConfig(
    co2,
    lifetime=[1e9, 394, 36, 4.3],
    partition_fraction=[0.25, 0.25, 0.25, 0.25],
    radiative_efficiency=1.33e-5,
    molecular_weight=44.009,
    iirf=IIRF(29, 0.000819, 0.00846, 4),
    scale=0.99
)
co2_cfg1.radiative_efficiency=1.6e-5

ch4_cfg1 = SpeciesConfig(
    ch4,
    lifetime=8.25,
    partition_fraction=1,
    molecular_weight=16.043,
    iirf=IIRF(8.25, 0.00032, 0, -0.3)
)
ch4_cfg2 = copy.copy(ch4_cfg1)
ch4_cfg2.scale=0.86

so2_cfg = SpeciesConfig(so2, erfari_emissions_to_forcing=-0.00362)
bc_cfg = SpeciesConfig(bc, erfari_emissions_to_forcing=0.0508)
aci_cfg = SpeciesConfig(aci, erfaci_beta=2.09841432, erfaci_shape_bcoc=76.7, erfaci_shape_sulfur=260.34644166)

config1 = Config('UKESM', cr1, [co2_cfg1, ch4_cfg1, so2_cfg])
config2 = Config('MIROC6', cr2, [co2_cfg2, ch4_cfg2, so2_cfg])
config3 = Config('IPSL', cr1, [co2_cfg1, ch4_cfg1, bc_cfg])
config4 = Config('GISS', cr2, [co2_cfg2, ch4_cfg2, bc_cfg])

config7 = Config('NorESM', cr2, [co2_cfg2, ch4_cfg2, aci_cfg])

# print(config1)
# print()
# print(config2)

# this should work
# fair=FAIR(scenarios=(scen1, scen2, scen3), configs=(config1, config2), time=np.arange(10))
# fair.run()
# print(fair)


co2_e1 = Emissions(co2, np.arange(50))
ch4_e1 = Emissions(ch4, np.ones(50)*200)
co2_e2 = Emissions(co2, np.arange(50)*2)
ch4_e2 = Emissions(ch4, np.ones(50)*100)
co2_e3 = Emissions(co2, np.ones(50)*10)
ch4_e3 = Emissions(ch4, np.zeros(50))
so2_e  = Emissions(so2, np.ones(50)*100)
co2_e5 = Emissions(co2, np.arange(10))
aci_p = Placeholder(aci)

scen1 = Scenario("ssp119", [co2_e1, ch4_e1, so2_e])
scen2 = Scenario("ssp126", [co2_e2, ch4_e2, so2_e])
scen3 = Scenario("ssp245", [co2_e3, ch4_e3, so2_e])
scen4 = Scenario("ssp370", [co2_e3])
scen5 = Scenario("ssp434", [co2_e5, ch4_e3, so2_e])
scen6 = Scenario("ssp460", [co2_e1, co2_e2, ch4_e1])
scen7 = Scenario("ssp585", [co2_e1, ch4_e1, aci_p])

print()

try:
    fair = FAIR(scenarios=[scen1, scen2, scen4], configs=[config1, config2], time=np.arange(10))
except (SpeciesMismatchError):
    print("test passed")

try:
    fair = FAIR(scenarios=[scen1, scen2, scen5], configs=[config1, config2], time=np.arange(10))
    print("test failed")
except (TimeMismatchError):
    print("test passed")

try:
    fair=FAIR()
    fair.run()
except MissingInputError:
    print("test passed")


try:
    fair=FAIR(scenarios=[scen1, scen2, 'scen3'], configs=[config1, config2], time=np.arange(10))
except TypeError:
    print ("test passed")

try:
    fair=FAIR(scenarios=[scen1, scen2, scen3], configs=[config1, 'config2'], time=np.arange(10))
except TypeError:
    print ("test passed")

try:
    fair=FAIR(scenarios=[scen1, scen2, scen3], configs=[config1, config2], time=np.arange(10))
    fair.run()
    print("test failed")
except TimeMismatchError:
    print("test passed")

try:
    fair=FAIR(scenarios=[scen1, scen2, scen3], configs=[config3, config4], time=np.arange(50))
    fair.run()
    print("test failed")
except SpeciesMismatchError:
    print("test passed")
# we want to make species and configs independent of sort order, currently this is not the case.
# config1 = Config(cr1, [ch4_cfg1, co2_cfg1, so2_cfg])
# config2 = Config(cr2, [co2_cfg2, ch4_cfg2, so2_cfg])
#
# fair=FAIR(scenarios=[scen1, scen2, scen3], configs=[config1, config2], time=np.arange(50))

try:
    fair = FAIR(scenarios=[scen6], configs=[config1, config2])
    print('test failed')
except DuplicationError:
    print('test passed')

fair = FAIR(scenarios=[scen7], configs=[config7])


fair = FAIR(scenarios=[scen1, scen2, scen3], configs=[config1, config2], time=np.arange(50))
fair.remove_scenario(scen3)
fair.add_scenario(scen3)
print(len(fair.scenarios))
fair.remove_config(config1)
fair.add_config(config1)
print(len(fair.configs))
fair.run()

print()
