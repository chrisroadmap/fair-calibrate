from ..structure.config_level import SpeciesConfig
from ..structure.top_level import SpeciesID, Category, RunMode

import numpy as np

# TODO: radiative efficiency for the big three should be calculated internally
# TODO: many of these default values need to be revisited, and decided whether
#       we want to fix on SSP emissions or the final dataset used in AR6
default_species_config = {
    'co2_afolu' : SpeciesConfig(
        species_id = SpeciesID(name='CO2 AFOLU', category=Category.CO2_AFOLU, run_mode=RunMode.EMISSIONS),
        land_use_cumulative_emissions_to_forcing = -0.000287290278097,
    ),
    'co2_ffi': SpeciesConfig(
        species_id = SpeciesID(name='CO2 fossil fuel and industrial', category=Category.CO2_FFI, run_mode=RunMode.EMISSIONS),
    ),
    'co2' : SpeciesConfig(
        species_id = SpeciesID(name='CO2', category=Category.CO2, run_mode=RunMode.FROM_OTHER_SPECIES),
        molecular_weight = 44.009,
        lifetime = np.array([1e9, 394.4, 36.54, 4.304]),
        partition_fraction = np.array([0.2173, 0.2240, 0.2824, 0.2763]),
        radiative_efficiency = 1.3344985680386619e-05,
        iirf_0=29,
        iirf_airborne=0.000819,
        iirf_cumulative=0.00846,
        iirf_temperature=4.0,
        baseline_concentration = 278.3,
        tropospheric_adjustment = 0.05
    ),
    'ch4' : SpeciesConfig(
        species_id = SpeciesID('CH4', Category.CH4, run_mode=RunMode.EMISSIONS),
        molecular_weight = 16.043,
        lifetime = 8.25,  # Thornhill lifetime best fit: 10.8537568
        radiative_efficiency = 0.00038864402860869495,
        iirf_airborne = 0.00032,
        iirf_temperature = -0.3,
        baseline_concentration = 729.2,
        tropospheric_adjustment = -0.14,
        ozone_radiative_efficiency = 1.75e-4,
        h2o_stratospheric_factor = 0.091914639065882,
        # soil_lifetime = 185,  # remember to set this if running AerChemMIP lifetime
        ch4_lifetime_chemical_sensitivity = 0.18,
        ch4_lifetime_temperature_sensitivity = -0.00167,
        normalisation = 1023.2219696044921,
    ),
    'n2o': SpeciesConfig(
        species_id = SpeciesID('N2O', Category.N2O, run_mode=RunMode.EMISSIONS),
        molecular_weight = 44.013,
        lifetime = 109,
        radiative_efficiency = 0.00319550741640458,
        baseline_concentration = 270.1,
        tropospheric_adjustment = 0.07,
        ozone_radiative_efficiency = 7.1e-4,
        ch4_lifetime_chemical_sensitivity = -0.039,
        normalisation = 53.96694437662762,
    ),
    'cfc-11': SpeciesConfig(
        species_id = SpeciesID('CFC-11', Category.CFC_11, run_mode=RunMode.EMISSIONS),
        molecular_weight = 137.36,
        lifetime = 52,
        radiative_efficiency = 0.25941,
        ozone_radiative_efficiency = -1.25e-4,
        cl_atoms = 3,
        fractional_release = 0.47,
        tropospheric_adjustment = 0.13,
    ),
    'cfc-12': SpeciesConfig(
        species_id = SpeciesID('CFC-12', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 120.91,
        lifetime = 102,
        radiative_efficiency = 0.31998,
        ozone_radiative_efficiency = -1.25e-4,
        cl_atoms = 2,
        fractional_release = 0.23,
        tropospheric_adjustment = 0.12,
    ),
    'cfc-113': SpeciesConfig(
        species_id = SpeciesID('CFC-113', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 187.37,
        lifetime = 93,
        radiative_efficiency = 0.30142,
        ozone_radiative_efficiency = -1.25e-4,
        cl_atoms = 3,
        fractional_release = 0.29,
    ),
    'cfc-114': SpeciesConfig(
        species_id = SpeciesID('CFC-114', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 170.92,
        lifetime = 189,
        radiative_efficiency = 0.31433,
        ozone_radiative_efficiency = -1.25e-4,
        cl_atoms = 2,
        fractional_release = 0.12,
    ),
    'cfc-115': SpeciesConfig(
        species_id = SpeciesID('CFC-115', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 154.46,
        lifetime = 540,
        radiative_efficiency = 0.24625,
        ozone_radiative_efficiency = -1.25e-4,
        cl_atoms = 1,
        fractional_release = 0.04,
    ),
    'hcfc-22': SpeciesConfig(
        species_id = SpeciesID('HCFC-22', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 86.47,
        lifetime = 11.9,
        radiative_efficiency = 0.21385,
        ozone_radiative_efficiency = -1.25e-4,
        cl_atoms = 1,
        fractional_release = 0.13,
    ),
    'hcfc-141b': SpeciesConfig(
        species_id = SpeciesID('HCFC-141b', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 116.95,
        lifetime = 9.4,
        radiative_efficiency = 0.16065,
        ozone_radiative_efficiency = -1.25e-4,
        cl_atoms = 2,
        fractional_release = 0.34,
    ),
    'hcfc-142b': SpeciesConfig(
        species_id = SpeciesID('HCFC-142b', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 100.49,
        lifetime = 18,
        radiative_efficiency = 0.19329,
        ozone_radiative_efficiency = -1.25e-4,
        cl_atoms = 1,
        fractional_release = 0.17,
    ),
    'ccl4': SpeciesConfig(
        species_id = SpeciesID('CCl4', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 153.8,
        lifetime = 32,
        radiative_efficiency = 0.16616,
        natural_emissions_adjustment = 0.024856862,
        baseline_concentration = 0.025,
        ozone_radiative_efficiency = -1.25e-4,
        cl_atoms = 4,
        fractional_release = 0.56,
    ),
    'chcl3': SpeciesConfig(
        species_id = SpeciesID('CHCl3', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 119.37,
        lifetime = 0.501,
        radiative_efficiency = 0.07357,
        natural_emissions_adjustment = 300.92479,
        baseline_concentration = 4.8,
        ozone_radiative_efficiency = -1.25e-4,
        cl_atoms = 3,
        fractional_release = 0,  # no literature value available
    ),
    'ch2cl2': SpeciesConfig(
        species_id = SpeciesID('CH2Cl2', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 84.93,
        lifetime = 0.493,
        radiative_efficiency = 0.02882,
        natural_emissions_adjustment = 246.6579,
        baseline_concentration = 6.91,
        ozone_radiative_efficiency = -1.25e-4,
        cl_atoms = 2,
        fractional_release = 0,  # no literature value available
    ),
    'ch3cl': SpeciesConfig(
        species_id = SpeciesID('CH3Cl', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 50.49,
        lifetime = 0.9,
        radiative_efficiency = 0.00466,
        natural_emissions_adjustment = 4275.7449,
        baseline_concentration = 457,
        ozone_radiative_efficiency = -1.25e-4,
        cl_atoms = 1,
        fractional_release = 0.44,
    ),
    'ch3ccl3': SpeciesConfig(
        species_id = SpeciesID('CH3CCl3', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 133.4,
        lifetime = 5,
        radiative_efficiency = 0.06454,
        ozone_radiative_efficiency = -1.25e-4,
        cl_atoms = 3,
        fractional_release = 0.67,
    ),
    'ch3br': SpeciesConfig(
        species_id = SpeciesID('CH3Br', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 94.94,
        lifetime = 0.8,
        radiative_efficiency = 0.00432,
        natural_emissions_adjustment = 105.08773,
        baseline_concentration = 5.3,
        ozone_radiative_efficiency = -1.25e-4,
        br_atoms = 1,
        fractional_release = 0.6,
    ),
    'halon-1211': SpeciesConfig(
        species_id = SpeciesID('Halon-1211', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 165.36,
        lifetime = 16,
        radiative_efficiency = 0.30014,
        natural_emissions_adjustment = 0.0077232726,
        baseline_concentration = 0.00445,
        ozone_radiative_efficiency = -1.25e-4,
        cl_atoms = 1,
        br_atoms = 1,
        fractional_release = 0.62,
    ),
    'halon-1301': SpeciesConfig(
        species_id = SpeciesID('Halon-1301', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 148.91,
        lifetime = 72,
        radiative_efficiency = 0.29943,
        baseline_concentration = 0.,
        ozone_radiative_efficiency = -1.25e-4,
        br_atoms = 1,
        fractional_release = 0.28,
    ),
    'halon-2402': SpeciesConfig(
        species_id = SpeciesID('Halon-2402', Category.OTHER_HALOGEN, run_mode=RunMode.EMISSIONS),
        molecular_weight = 259.82,
        lifetime = 28,
        radiative_efficiency = 0.31169,
        ozone_radiative_efficiency = -1.25e-4,
        br_atoms = 2,
        fractional_release = 0.65,
    ),
    'cf4': SpeciesConfig(
        species_id = SpeciesID('CF4', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 88.004,
        lifetime = 50000,
        radiative_efficiency = 0.09859,
        baseline_concentration = 34.05,
        natural_emissions_adjustment = 0.010071225,
    ),
    'c2f6': SpeciesConfig(
        species_id = SpeciesID('C2F6', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 138.01,
        lifetime = 10000,
        radiative_efficiency = 0.26105,
    ),
    'c3f8': SpeciesConfig(
        species_id = SpeciesID('C3F8', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 188.02,
        lifetime = 2600,
        radiative_efficiency = 0.26999,
    ),
    'c-c4f8': SpeciesConfig(
        species_id = SpeciesID('C-C4F8', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 200.03,
        lifetime = 3200,
        radiative_efficiency = 0.31392,
    ),
    'c4f10': SpeciesConfig(
        species_id = SpeciesID('C4F10', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 238.03,
        lifetime = 2600,
        radiative_efficiency = 0.36874,
    ),
    'c5f12': SpeciesConfig(
        species_id = SpeciesID('C5F12', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 288.03,
        lifetime = 4100,
        radiative_efficiency = 0.4076,
    ),
    'c6f14': SpeciesConfig(
        species_id = SpeciesID('C6F14', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 338.04,
        lifetime = 3100,
        radiative_efficiency = 0.44888,
    ),
    'c7f16': SpeciesConfig(
        species_id = SpeciesID('C7F16', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 388.05,
        lifetime = 3000,
        radiative_efficiency = 0.50312,
    ),
    'c8f18': SpeciesConfig(
        species_id = SpeciesID('C8F18', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 438.06,
        lifetime = 3000,
        radiative_efficiency = 0.55787,
    ),
    'hfc-125': SpeciesConfig(
        species_id = SpeciesID('HFC-125', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 120.02,
        lifetime = 30,
        radiative_efficiency = 0.23378,
        ch4_lifetime_chemical_sensitivity = -0.0009642894772697214,
        normalisation = 15.355007727940878,
    ),
    'hfc-134a': SpeciesConfig(
        species_id = SpeciesID('HFC-134a', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 102.03,
        lifetime = 14,
        radiative_efficiency = 0.16714,
        ch4_lifetime_chemical_sensitivity = -0.003615023786272495,
        normalisation = 80.51572863260905,
    ),
    'hfc-143a': SpeciesConfig(
        species_id = SpeciesID('HFC-143a', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 84.04,
        lifetime = 51,
        radiative_efficiency = 0.168,
        ch4_lifetime_chemical_sensitivity = -0.0006883484644028191,
        normalisation = 15.25278091430664,
    ),
    'hfc-152a': SpeciesConfig(
        species_id = SpeciesID('HFC-152a', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 66.05,
        lifetime = 1.6,
        radiative_efficiency = 0.10174,
        ch4_lifetime_chemical_sensitivity = -0.00021133459396516935,
        normalisation = 7.732658425966899,
    ),
    'hfc-227ea': SpeciesConfig(
        species_id = SpeciesID('HFC-227ea', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 170.03,
        lifetime = 36,
        radiative_efficiency = 0.27325,
        ch4_lifetime_chemical_sensitivity = -7.381860158059581e-05,
        normalisation = 1.0056663114594357,
    ),
    'hfc-23': SpeciesConfig(
        species_id = SpeciesID('HFC-23', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 70.014,
        lifetime = 228,
        radiative_efficiency = 0.19111,
        ch4_lifetime_chemical_sensitivity = -0.001380483674751533,
        normalisation = 26.890435059865315,
    ),
    'hfc-236fa': SpeciesConfig(
        species_id = SpeciesID('HFC-236fa', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 152.04,
        lifetime = 213,
        radiative_efficiency = 0.25069,
        ch4_lifetime_chemical_sensitivity = -8.79292645911157e-06,
        normalisation = 0.130570899695158,
    ),
    'hfc-245fa': SpeciesConfig(
        species_id = SpeciesID('HFC-245fa', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 134.05,
        lifetime = 7.9,
        radiative_efficiency = 0.24498,
        ch4_lifetime_chemical_sensitivity = -0.00013473391955397875,
        normalisation = 2.047369738419851,
    ),
    'hfc-32': SpeciesConfig(
        species_id = SpeciesID('HFC-32', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 52.023,
        lifetime = 5.4,
        radiative_efficiency = 0.11144,
        ch4_lifetime_chemical_sensitivity = -0.00024957399034576547,
        normalisation = 8.33675058935114,
    ),
    'hfc-365mfc': SpeciesConfig(
        species_id = SpeciesID('HFC-365mfc', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 148.07,
        lifetime = 8.9,
        radiative_efficiency = 0.22813,
        ch4_lifetime_chemical_sensitivity = -4.688253387776103e-05,
        normalisation = 0.7650303095579147,
    ),
    'hfc-4310mee': SpeciesConfig(
        species_id = SpeciesID('HFC-4310mee', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 252.05,
        lifetime = 17,
        radiative_efficiency = 0.35731,
        ch4_lifetime_chemical_sensitivity = -2.3733082702031964e-05,
        normalisation = 0.2472628938655058,
    ),
    'nf3': SpeciesConfig(
        species_id = SpeciesID('NF3', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 71.002,
        lifetime = 569,
        radiative_efficiency = 0.20448,
    ),
    'sf6': SpeciesConfig(
        species_id = SpeciesID('SF6', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 146.06,
        lifetime = 3200,
        radiative_efficiency = 0.56657,
    ),
    'so2f2': SpeciesConfig(
        species_id = SpeciesID('SO2F2', Category.F_GAS, run_mode=RunMode.EMISSIONS),
        molecular_weight = 102.06,
        lifetime = 36,
        radiative_efficiency = 0.21074,
    ),
    'sulfur': SpeciesConfig(
        species_id = SpeciesID('Sulfur', Category.SULFUR, run_mode=RunMode.EMISSIONS),
        erfari_radiative_efficiency = -0.0036167830509091486,
        baseline_emissions = 2.44004843482201
    ),
    'bc': SpeciesConfig(
        species_id = SpeciesID('BC', Category.BC, run_mode=RunMode.EMISSIONS),
        erfari_radiative_efficiency = 0.0507748226795483,
        baseline_emissions = 2.09777075542297,
        lapsi_radiative_efficiency = 0.011585926
    ),
    'oc': SpeciesConfig(
        species_id = SpeciesID('OC', Category.OC, run_mode=RunMode.EMISSIONS),
        erfari_radiative_efficiency = -0.006214374446217472,
        baseline_emissions = 15.4476681469614
    ),
    'nh3': SpeciesConfig(
        species_id = SpeciesID('NH3', Category.OTHER_AEROSOL, run_mode=RunMode.EMISSIONS),
        erfari_radiative_efficiency = -0.0020809236231100624,
        baseline_emissions = 6.92769009144426
    ),
    'co': SpeciesConfig(
        species_id = SpeciesID('CO', Category.SLCF_OZONE_PRECURSOR, run_mode=RunMode.EMISSIONS),
        ozone_radiative_efficiency = 1.55e-4,
        baseline_emissions = 348.52735877736,
    ),
    'nox' : SpeciesConfig(
        species_id = SpeciesID('NOx', Category.SLCF_OZONE_PRECURSOR, run_mode=RunMode.EMISSIONS),
        ozone_radiative_efficiency = 1.797e-3,
        baseline_emissions = 12.7352119423177,
        ch4_lifetime_chemical_sensitivity = -0.291428531,
        normalisation = 142.18364862292066,
    ),
    'voc' : SpeciesConfig(
        species_id = SpeciesID('VOC', Category.SLCF_OZONE_PRECURSOR, run_mode=RunMode.EMISSIONS),
        ozone_radiative_efficiency = 3.29e-4,
        baseline_emissions = 60.0218262241548,
        ch4_lifetime_chemical_sensitivity = 0.241599754,
        normalisation = 166.74246925530488,
    ),
    'nox_aviation' : SpeciesConfig(
        species_id = SpeciesID('NOx Aviation', Category.NOX_AVIATION, run_mode=RunMode.EMISSIONS),
        contrails_radiative_efficiency = 0.014664524317963392,   # W/m2/(MtNO2-aviation)
    ),
    'ari' : SpeciesConfig(
        species_id = SpeciesID('Aerosol-Radiation Interactions', Category.AEROSOL_RADIATION_INTERACTIONS, run_mode=RunMode.FROM_OTHER_SPECIES)
    ),
    'aci': SpeciesConfig(
        species_id = SpeciesID('Aerosol-Cloud Interactions', Category.AEROSOL_CLOUD_INTERACTIONS, run_mode=RunMode.FROM_OTHER_SPECIES),
        aci_params={"scale": 2.09841432, "Sulfur": 260.34644166, "BC+OC": 111.05064063}
    ),
    'lapsi': SpeciesConfig(
        species_id = SpeciesID('Light absorbing particles on snow and ice', Category.LAPSI, run_mode=RunMode.FROM_OTHER_SPECIES)
    ),
    'h2o_stratospheric': SpeciesConfig(
        species_id = SpeciesID('H2O Stratospheric', Category.H2O_STRATOSPHERIC, run_mode=RunMode.FROM_OTHER_SPECIES)
    ),
    'ozone': SpeciesConfig(
        species_id = SpeciesID('Ozone', Category.OZONE, run_mode=RunMode.FROM_OTHER_SPECIES),
        forcing_temperature_feedback = -0.037
    ),
    'contrails': SpeciesConfig(
        species_id = SpeciesID('Contrails', Category.CONTRAILS, run_mode=RunMode.FROM_OTHER_SPECIES)
    ),
    'land_use': SpeciesConfig(
        species_id = SpeciesID('Land Use', Category.LAND_USE, run_mode=RunMode.FROM_OTHER_SPECIES)
    ),
    'solar': SpeciesConfig(
        species_id = SpeciesID('Solar', Category.SOLAR, run_mode=RunMode.FORCING)
    ),
    'volcanic': SpeciesConfig(
        species_id = SpeciesID('Volcanic', Category.VOLCANIC, run_mode=RunMode.FORCING)
    )
}
