"""
Defaults for (effective) radiative forcing.
"""

from ..constants.gases import gas_list

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

tropospheric_adjustment["Ozone"] = 1
