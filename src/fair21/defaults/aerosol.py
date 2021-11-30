"""
Default parameters relating to aerosol forcing.

TODO: investigate overlaps between NH3 and SO2 - does not appear to be any
really convincing literature on this at the moment.
"""

from . import species_list

# aerosol-radiation: assumes linear scaling with emissions
radiative_efficiency = {
    "AR6": {specie: 0 for specie in species_list},
    "Smith2021": {specie: 0 for specie in species_list},
    "Stevens2015": {specie: 0 for specie in species_list},
}
radiative_efficiency["AR6"].update(
    {
        "Sulfur": -0.0036167830509091486,
        "BC": 0.0507748226795483,
        "OC": -0.006214374446217472,
        "NH3": -0.0020809236231100624,
    }

)
radiative_efficiency["Smith2021"].update(
    {
        "Sulfur": -0.0025,
        "BC": 0.0285,
        "OC": -0.0085,
    }
)
radiative_efficiency["Stevens2015"].update({"Sulfur": 0.001875})

# aerosol-cloud: everything's different here
# probably need to include a Leach 2021 too
beta = {
    "AR6": 2.09841432,
    "Smith2021": 1.223,
    "Stevens2015": 0.634
}

shape = {
    "AR6": {
        "Sulfur": 260.34644166,
        "BC+OC": 111.05064063
    },
    "Smith2021": {
        "Sulfur": 156.5,
        "BC+OC": 76.7
    },
    "Stevens2015": {"Sulfur": 60}  # has not been tested
}
