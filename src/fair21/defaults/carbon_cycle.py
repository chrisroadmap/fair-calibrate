# CARBON CYCLE SPECIFIC
partition_fraction = {
    "CO2": np.array([0.2173, 0.2240, 0.2824, 0.2763])
}
# these provide a close representation to SSP2-4.5 concentrations provided for
# CMIP6. They are in fact probably too sensitive compared to the average CMIP6
# ESM.
iirf_0 = 29 # yr
iirf_cumulative = 0.00846 # yr/GtCO2
iirf_temperature = 4.0 # yr/K
iirf_airborne = 0.000819 # yr/GtCO2
iirf_horizon = 100 # yr