"""
Module for defaults
"""

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
