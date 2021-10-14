"""
Default parameters relating to greenhouse gases
"""

import numpy as np

from ..constants.gases import gas_list, lifetime

# RADIATIVE EFFICIENCIES
#
# Rationale for putting this in "defaults" rather than "constants" is that
# radiative efficiencies get updated from time to time and are not constants
# anyway, being pressure and temperature dependent.

# Reference: Hodnebrog et al. (2020) https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019RG000691, as used in the AR6 repo
# Chris Smith, Piers Forster, Matt Palmer, Bill Collins, Nick Leach, Masa Watanabe, Sophie Berger, & Brad Hall. (2021). IPCC-WG1/Chapter-7: IPCC WGI AR6 Chapter 7 (v.1.0). Zenodo. https://doi.org/10.5281/zenodo.5211358
radiative_efficiency = {
    'HFC-125': 0.23378,
    'HFC-134a': 0.16714,
    'HFC-143a': 0.168,
    'HFC-152a': 0.10174,
    'HFC-227ea': 0.27325,
    'HFC-23': 0.19111,
    'HFC-236fa': 0.25069,
    'HFC-245fa': 0.24498,
    'HFC-32': 0.11144,
    'HFC-365mfc': 0.22813,
    'HFC-4310mee': 0.35731,
    'NF3': 0.20448,
    'C2F6': 0.26105,
    'C3F8': 0.26999,
    'C4F10': 0.36874,
    'C5F12': 0.4076,
    'C6F14': 0.44888,
    'C7F16': 0.50312,
    'C8F18': 0.55787,
    'CF4': 0.09859,
    'cC4F8': 0.31392,
    'SF6': 0.56657,
    'SO2F2': 0.21074,
    'CCl4': 0.16616,
    'CFC-11': 0.25941,
    'CFC-112': 0.28192,
    'CFC-112a': 0.24564,
    'CFC-113': 0.30142,
    'CFC-113a': 0.24094, 
    'CFC-114': 0.31433,
    'CFC-114a': 0.29747,
    'CFC-115': 0.24625,
    'CFC-12': 0.31998,
    'CFC-13': 0.27752,
    'CH2Cl2': 0.02882,
    'CH3Br': 0.00432,
    'CH3CCl3': 0.06454,
    'CH3Cl': 0.00466,
    'CHCl3': 0.07357,
    'HCFC-124': 0.20721,
    'HCFC-133a': 0.14995,
    'HCFC-141b': 0.16065,
    'HCFC-142b': 0.19329,
    'HCFC-22': 0.21385,
    'HCFC-31': 0.068,
    'Halon-1211': 0.30014,
    'Halon-1301': 0.29943,
    'Halon-2402': 0.31169,
}


# PRE-INDUSTRIAL GREENHOUSE GAS CONCENTRATIONS
#
# Data from: Bradley Hall, Histories of long-lived greenhouse gases (global
# annual mean at Earth's surface) derived from multiple sources. Year 1750 is
# used as a proxy for pre-industrial.
#
# Used in IPCC AR6 WG1 Chapter 2, Chapter 7 and Annex 3
#
# Reference:
# IPCC, 2021: Annex III: Tables of historical and projected well-mixed
# greenhouse gas mixing ratios and effective radiative forcing of all climate
# forcers [Dentener F. J., B. Hall, C. Smith (eds.)]. In: Climate Change 2021:
# The Physical Science Basis. Contribution of Working Group I to the Sixth
# Assessment Report of the Intergovernmental Panel on Climate Change
# [Masson-Delmotte, V., P. Zhai, A. Pirani, S. L. Connors, C. Péan, S. Berger,
# N. Caud, Y. Chen, L. Goldfarb, M. I. Gomis, M. Huang, K. Leitzell, E. Lonnoy,
# J.B.R. Matthews, T. K. Maycock, T. Waterfield, O. Yelekçi, R. Yu and B. Zhou
# (eds.)]. Cambridge University Press. In Press.
#
# All quantities are ppt volume mixing ratio in dry air unless specified
pre_industrial_concentration = {}
for gas in gas_list:
    pre_industrial_concentration[gas] = 0
pre_industrial_concentration.update(
    {
        "CO2" : 278.3,
        "CH4" : 729.2,
        "N2O" : 270.1,
        "CF4" : 34.05,
        "CCl4": 0.025,
        "CH3Cl": 457,
        "CH3Br" : 5.3,
        "CH2Cl2": 6.91,
        "CHCl3": 4.8,
        "Halon-1211" : 0.00445
    }
)

# TROPOSPHERIC ADJUSTMENTS
#
# Tropospheric "rapid" adjustments expressed as a multiplicative ratio of the
# stratospherically-adjusted radiative forcing. This calculates the effective
# radiative forcing given SARF.
#
# Reference:
# Forster, P., T. Storelvmo, K. Armour, W. Collins, J. L. Dufresne, D. Frame, 
# D. J. Lunt, T. Mauritsen, M. D. Palmer, M. Watanabe, M. Wild, H. Zhang,
# 2021, The Earth’s Energy Budget, Climate Feedbacks, and Climate Sensitivity.
# In: Climate Change 2021: The Physical Science Basis. Contribution of Working
# Group I to the Sixth Assessment Report of the Intergovernmental Panel on 
# Climate Change [Masson-Delmotte, V., P. Zhai, A. Pirani, S. L. Connors, C.
# Péan, S. Berger, N. Caud, Y. Chen, L. Goldfarb, M. I. Gomis, M. Huang, K.
# Leitzell, E. Lonnoy, J.B.R. Matthews, T. K. Maycock, T. Waterfield, O. 
# Yelekçi, R. Yu and B. Zhou (eds.)]. Cambridge University Press.
tropospheric_adjustment = {}
for gas in gas_list:
    tropospheric_adjustment[gas] = 1
tropospheric_adjustment["CO2"] = 1.05
tropospheric_adjustment["CH4"] = 0.86
tropospheric_adjustment["N2O"] = 1.07
tropospheric_adjustment["CFC-11"] = 1.13
tropospheric_adjustment["CFC-12"] = 1.12


# Carbon cycle partition fractions
partition_fraction = {}
for gas in gas_list:
    partition_fraction[gas] = 1
partition_fraction["CO2"] = np.array([0.2173, 0.2240, 0.2824, 0.2763])

# Variable lifetime gases: only relevant for CO2 and CH4, so we don't fill
# for all gases.
iirf_0 = {
    "CO2" : 29,
    "CH4" : lifetime["CH4"]
} # yr
iirf_cumulative = {
    "CO2" : 0.00846,
    "CH4" : 0
} # yr/GtCO2
iirf_temperature = {
    "CO2" : 4.0,
    "CH4" : -0.3,
} # yr/K
iirf_airborne = {
    "CO2" : 0.000819,
    "CH4" : 0.00032
} # yr/GtCO2

iirf_horizon = 100 # yr
