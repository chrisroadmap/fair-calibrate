"""
Greenhouse gas properties
"""

import numpy as np

from .general import M_ATMOS

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

# MOLECULAR WEIGHTS
#
# Convention: AIR is top, then other gases by formula or common name in
# alphabetical order.
#
# Unless stated, source is PubChem (https://pubchem.ncbi.nlm.nih.gov/).
# Naming convention also follows PubChem in the sense that there is a dash
# between "CFC" and the gas number (e.g. CFC-11, not CFC11)
#
# Units are g/mol.
#
# TODO
# 1. allow CFCs, HFCs etc. to be specified equivalently without a dash.
# 2. all gases that were assessed in Chapter 2 of AR6 by Brad Hall.
# 3. potential extension to all Chapter 7 SM gases, for calculation of metrics.
molwt = {
    "AIR": 28.97,  # reference?
    "C2F6": 138.01,
    "C3F8": 188.02,
    "C4F10": 238.03,
    "C5F12": 288.03,
    "C6F14": 338.04,
    "C7F16": 388.05,
    "C8F18": 438.06,
    "cC4F8": 200.03,  # not standard PubChem but used extensively in AR6
    "CCl4": 153.8,
    "CF4": 88.004,
    "CFC-113": 187.37,
    "CFC-114": 170.92,
    "CFC-115": 154.46,
    "CFC-11": 137.36,
    "CFC-12": 120.91,
    "CH2Cl2": 84.93,
    "CH3Br": 94.94,
    "CH3CCl3": 133.4,
    "CH3Cl": 50.49,
    "CH4": 16.043,
    "CHCl3": 119.37,
    "CO2": 44.009,
    "Halon-1211": 165.36,
    "Halon-1301": 148.91,
    "Halon-2402": 259.82,
    "HCFC-141b": 116.95,
    "HCFC-142b": 100.49,
    "HCFC-22": 86.47,
    "HFC-125": 120.02,
    "HFC-134a": 102.03,
    "HFC-143a": 84.04,
    "HFC-152a": 66.05,
    "HFC-227ea": 170.03,
    "HFC-23": 70.014,
    "HFC-236fa": 152.04,
    "HFC-245fa": 134.05,
    "HFC-32": 52.023,
    "HFC-365mfc": 148.07,
    "HFC-4310mee": 252.05,
    "N2O": 44.013,
    "NF3": 71.002,
    "SF6": 146.06,
    "SO2F2": 102.06,
}

# ATMOSPHERIC LIFETIMES
#
# Convention: alphabetical order by formula or common name
# Unless stated, source is Smith et al. (2021), AR6 Chapter 7 Supplementary Material
# https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter_07_Supplementary_Material.pdf
lifetime = {
    "C2F6": 10000,
    "C3F8": 2600,
    "C4F10": 2600,
    "C5F12": 4100,
    "C6F14": 3100,
    "C7F16": 3000,
    "C8F18": 3000,
    "cC4F8": 3200,  # not standard PubChem name but used extensively in AR6
    "CCl4": 32,
    "CF4": 50000,
    "CFC-113": 93,
    "CFC-114": 189,
    "CFC-115": 540,
    "CFC-11": 52,
    "CFC-12": 102,
    "CH2Cl2": 0.493,
    "CH3Br": 0.8,
    "CH3CCl3": 5,
    "CH3Cl": 0.9,
    "CH4": 8.25,  # atmospheric burden lifetime in pre-industrial conditions. Source: Leach et al. (2021)
    "CHCl3": 0.501,
    "CO2": np.array([1e9, 394.4, 36.54, 4.304]),
    "Halon-1211": 16,
    "Halon-1301": 72,
    "Halon-2402": 28,
    "HCFC-141b": 9.4,
    "HCFC-142b": 18,
    "HCFC-22": 11.9,
    "HFC-125": 30,
    "HFC-134a": 14,
    "HFC-143a": 51,
    "HFC-152a": 1.6,
    "HFC-227ea": 11.4,
    "HFC-23": 228,
    "HFC-236fa": 213,
    "HFC-245fa": 7.9,
    "HFC-32": 5.4,
    "HFC-365mfc": 8.9,
    "HFC-4310mee": 17,
    "N2O": 109,
    "NF3": 569,
    "SF6": 3200,
    "SO2F2": 36,
}

# CONCENTRATION GROWTH UNITS
#
# How much the atmospheric burden grows for a given emission
burden_per_emission = {}
for gas in gas_list:
    burden_per_emission[gas] = (
        1 / (M_ATMOS / 1e18 * molwt[gas] / molwt["AIR"])
    )
# # CO2 and probably N2O need further adjustment
# burden_per_emission['CO2'] = (
#     burden_per_emission['CO2'] * molwt["CO2"] / molwt["C"]
# )
