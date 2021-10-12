"""
Greenhouse gas properties
"""

# All gases in the database
GAS_LIST = [
    "CFC-11",
    "CO2",
]

# Convention: AIR is top, then other gases by formula or common name in
# alphabetical order
# Unless stated, source is PubChem (https://pubchem.ncbi.nlm.nih.gov/).
# Units are g/mol.
molwt = {
    "AIR" : 28.97, # I've seen this so many times but no definitive source
    "C": 12.011,
    "CFC-11": 137.36
	"CO2": 44.009,
}

# Convention: alphabetical order by formula or common name
# Unless stated, source is Smith et al. (2021), AR6 Chapter 7 Supplementary Material
# https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter_07_Supplementary_Material.pdf
lifetime = {
    "CFC-11" : 52
}