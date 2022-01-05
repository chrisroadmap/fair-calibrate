# this will eventally be split up and become the testing Module

from fair import *

cr1 = ClimateResponse(ocean_heat_capacity = (5, 20, 100), ocean_heat_transfer=(1,2,1), deep_ocean_efficacy=1.29)
cr2 = ClimateResponse(ocean_heat_capacity = (2, 10, 80), ocean_heat_transfer=(1,2,3))

co2 = Species(name="CO2")
ch4 = Species("CH4")
so2 = Species("Sulfur")

co2_cfg1 = GasConfig(
    co2,
    lifetime=[1e9, 394, 36, 4.3],
    partition_fraction=[0.25, 0.25, 0.25, 0.25],
    radiative_efficiency=1.33e-5,
    molecular_weight=44.009,
    scale=1.04
)

co2_cfg2 = GasConfig(
    co2,
    lifetime=[1e9, 394, 36, 4.3],
    partition_fraction=[0.25, 0.25, 0.25, 0.25],
    radiative_efficiency=1.33e-5,
    molecular_weight=44.009,
    iirf=IIRF(29, 0.000819, 0.00846, 4),
    scale=0.99
)
co2_cfg1.radiative_efficiency=1.6e-5

ch4_cfg1 = GasConfig(
    ch4,
    lifetime=8.25,
    partition_fraction=1,
    molecular_weight=16.043,
    iirf=IIRF(8.25, 0.00032, 0, -0.3)
)
ch4_cfg2 = copy.copy(ch4_cfg1)
ch4_cfg2.scale=0.86

so2_cfg = AerosolConfig(so2, erfari_emissions_to_forcing=-0.00362)

config1 = Config(cr1, (co2_cfg1, ch4_cfg1, so2_cfg))
config2 = Config(cr2, (co2_cfg2, ch4_cfg2, so2_cfg))

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

scen1 = Scenario((co2_e1, ch4_e1, so2_e))
scen2 = Scenario((co2_e2, ch4_e2, so2_e))
scen3 = Scenario((co2_e3, ch4_e3, so2_e))
scen4 = Scenario((co2_e3))

##print(co2_e1.emissions)
#print(co2_e1)
print()

fair = FAIR(scenarios=[scen1, scen2, scen4], configs=[config1, config2], time=np.arange(10))
#print(fair.scenarios)
print()



# should give error
try:
    fair=FAIR()
    fair.run()
except MissingInputError:
    print("test passed")


try:
    fair=FAIR(scenarios=(scen1, scen2, 'scen3'), configs=(config1, config2), time=np.arange(10))
    fair.run()
except TypeError:
    print ("test passed")

try:
    fair=FAIR(scenarios=(scen1, scen2, scen3), configs=(config1, 'config2'), time=np.arange(10))
    fair.run()
except TypeError:
    print ("test passed")

print()
