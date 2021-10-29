"""
Module for defaults
"""

# options in all anthropogenic categories are "emissions", "forcing" or None;
# greenhouse_gases can also be "concentration";
# solar and volcanic can only be "forcing" or None.

# TODO: standardise names across inputs and outputs
run_mode = {
    'CO2' : 'emissions',
    'CH4' : 'emissions',
    'N2O' : 'emissions',
    'Other greenhouse gases': 'emissions',
    'Aerosol' : 'emissions',
    'Ozone' : 'emissions',
    'Land use' : 'emissions',
    'Stratospheric water vapour' : 'emissions',
    'Contrails' : 'emissions',
    'Black carbon on snow' : 'emissions',
    'Solar': 'forcing',
    'Volcanic': 'forcing'
}

valid_run_modes = {
    'CO2' : ('emissions', 'concentration', 'forcing', None),
    'CH4' : ('emissions', 'concentration', 'forcing', None),
    'N2O' : ('emissions', 'concentration', 'forcing', None),
    'Other greenhouse gases' : ('emissions', 'concentration', 'forcing', None),
    'Aerosol' : ('emissions', 'forcing', None),
    'Ozone' : ('emissions', 'forcing', None),
    'Land use' : ('emissions', 'forcing', None),
    'Stratospheric water vapour' : ('emissions', 'forcing', None),
    'Contrails' : ('emissions', 'forcing', None),
    'Black carbon on snow' : ('emissions', 'forcing', None),
    'Solar' : ('forcing', None),
    'Volcanic' : ('forcing', None),
}

# GAS LIST
#
# All gases in the database, except air and gas-phase SLCFs.
# We use this list to create conversion units.
gas_list = [
    "C2F6",
    "C3F8",
    "C4F10",
    "C5F12",
    "C6F14",
    "C7F16",
    "C8F18",
    "cC4F8",  # not standard PubChem but used extensively in AR6
    "CCl4",
    "CF4",
    "CFC-113",
    "CFC-114",
    "CFC-115",
    "CFC-11",
    "CFC-12",
    "CH2Cl2",
    "CH3Br",
    "CH3CCl3",
    "CH3Cl",
    "CH4",
    "CHCl3",
    "CO2",
    "Halon-1211",
    "Halon-1301",
    "Halon-2402",
    "HCFC-141b",
    "HCFC-142b",
    "HCFC-22",
    "HFC-125",
    "HFC-134a",
    "HFC-143a",
    "HFC-152a",
    "HFC-227ea",
    "HFC-23",
    "HFC-236fa",
    "HFC-245fa",
    "HFC-32",
    "HFC-365mfc",
    "HFC-4310mee",
    "N2O",
    "NF3",
    "SF6",
    "SO2F2",
]

gas_list_excl_co2_ch4 = [gas for gas in gas_list if gas not in ('CO2', "CH4")]
minor_gas_list = [gas for gas in gas_list if gas not in ('CO2', 'CH4', 'N2O')]

# SHORT LIVED CLIMATE FORCER LISTS
#
# All short-lived forcers in the database; for the purposes of FaIR, CH4 is
# considered as a long-lived GHG and not included here.
#
# using RCMIP names
slcf_list = [
    "Sulfur",
    "BC",
    "OC",
    "NOx",
    "NH3",
    "VOC",
    "CO"
]

# UNITS
emissions_units = {}
for gas in gas_list:
    emissions_units[gas] = 'kt / yr'  # want exact pint/scmdata syntax here
emissions_units['CO2'] = 'Gt / yr'  # GtCO2
emissions_units['CH4'] = 'Mt / yr'
emissions_units['N2O'] = 'Mt / yr'
for slcf in slcf_list:
    emissions_units[slcf] = 'Mt / yr'

concentration_units = {}
for gas in gas_list:
    concentration_units[gas] = 'ppt'
concentration_units['CO2'] = 'ppm'
concentration_units['CH4'] = 'ppb'
concentration_units['N2O'] = 'ppb'
