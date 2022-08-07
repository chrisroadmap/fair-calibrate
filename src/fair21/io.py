import os

import pandas as pd

HERE = os.path.dirname(os.path.realpath(__file__))
DEFAULT_PROPERTIES_FILE = os.path.join(HERE, "defaults", "data", "ar6", "species_configs_properties.csv")


# todo:
# read_fair1
# read_scen
# read_rcmip
# read_scmdata
def read_properties(filename=DEFAULT_PROPERTIES_FILE, species=None):
    """Gets a properties file.

    Inputs
    ------
    filename : str, optional
        path to a csv file. Default is an AR6 WG1-like config for FaIR
        covering all of the species considered in CMIP6.
    species : list of str or None, optional
        the species that are to be included in the FaIR run. All of these
        species should be present in the index (first column) of the csv. If
        None (default), return all of the species in the defaults.

    Returns
    -------
    properties : dict
        species properties that control the FaIR run
    """

    df = pd.read_csv(filename, index_col=0)

    if species is None:
        species = list(df.index)

    properties = {}
    for specie in species:
        properties[specie] = {
            'type': df.loc[specie].type,
            'emissions': bool(df.loc[specie].emissions),
            'concentration': bool(df.loc[specie].concentration),
            'forcing': bool(df.loc[specie].forcing),
            'input_mode': df.loc[specie].input_mode,
            'greenhouse_gas': bool(df.loc[specie].greenhouse_gas),
            'aerosol_radiation_precursor': bool(df.loc[specie].aerosol_radiation_precursor),
        }
    return properties



# write_configs
# write_output
