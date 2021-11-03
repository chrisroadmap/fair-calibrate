"""
Default parameters relating to ozone forcing
"""

import numpy as np

from . import gas_list

# OZONE FORCING COEFFICIENTS
#
# This defines the effective radiative forcing per emission or concentration
# of an ozone forming / ozone depleting substance.

# Reference: Smith, C., Z.R.J. Nicholls, K. Armour, W. Collins, P. Forster, M. Meinshausen, M. D. Palmer, M. Watanabe, 2021, The Earth’s Energy Budget, Climate Feedbacks, and Climate Sensitivity Supplementary Material. In: Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change [Masson-Delmotte, V., P. Zhai, A. Pirani, S. L. Connors, C. Péan, S. Berger, N. Caud, Y. Chen, L. Goldfarb, M. I. Gomis, M. Huang, K. Leitzell, E. Lonnoy, J.B.R. Matthews, T. K. Maycock, T. Waterfield, O. Yelekçi, R. Yu and B. Zhou (eds.)]. Available from https://ipcc.ch/static/ar6/wg1.
# from table 7.SM.3
radiative_efficiency = {
    "CH4": 1.75e-4, # W m-2 ppb-1
    "N2O": 7.10e-4, # W m-2 ppb-1
    "Montreal Gases": -1.25e-4, # W m-2 ppt-1  (maybe time to use bar notation after all)
    "CO": 1.55e-4, # W m-2 (MtCO yr)-1
    "VOC": 3.29e-4, # W m-2 (MtVOC yr)-1
    "NOx": 1.797e-3, # W m-2 (MtNO2 yr)-1
}

br_cl_ratio = 45

# Fractional release (for ozone depletion)
# References:
# Daniel, J. and Velders, G.: A focus on information and options for
# policymakers, in: Scientific Assessment of Ozone Depletion, WMO, 2011
# Newman et al., 2007: A new formulation of equivalent effective stratospheric
# chlorine (EESC)
fractional_release = {}
for gas in gas_list:
    fractional_release[gas] = 0
fractional_release.update(
    {
        'CCl4': 0.56,
        'CFC-11': 0.47,
        'CFC-113': 0.29,
        'CFC-114': 0.12,
        'CFC-115': 0.04,
        'CFC-12': 0.23,
        'CH2Cl2': 0, # TODO: try to update: no literature value available
        'CH3Br': 0.60,
        'CH3CCl3': 0.67,
        'CH3Cl': 0.44,
        'CHCl3': 0, # TODO: try to update: no literature value available
        'HCFC-141b': 0.34,
        'HCFC-142b': 0.17,
        'HCFC-22': 0.13,
        'Halon-1211': 0.62,
        'Halon-1301': 0.28,
        'Halon-2402': 0.65,
    }
)
