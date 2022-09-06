"""Module for unit test of fair."""

import numpy as np
import pytest

from fair import FAIR
from fair.io import read_properties

f = FAIR()


def minimal_ghg_run():
    fair_obj = FAIR()
    species = ["CO2", "CH4", "N2O"]
    species, properties = read_properties(species=species)
    for specie in species:
        properties[specie]["input_mode"] = "concentration"
    fair_obj.define_species(species, properties)
    fair_obj.define_time(1750, 2020, 270)
    fair_obj.define_scenarios(["historical"])
    fair_obj.define_configs(["UKESM1-0-LL"])
    fair_obj.allocate()
    fair_obj.climate_configs["ocean_heat_capacity"][0, :] = np.array(
        [2.917300055, 11.28317472, 73.2487238]
    )
    fair_obj.climate_configs["ocean_heat_transfer"][0, :] = np.array(
        [0.65576633, 2.597877675, 0.612933889]
    )
    fair_obj.climate_configs["deep_ocean_efficacy"][0] = 1.133708775
    fair_obj.climate_configs["gamma_autocorrelation"][0] = 28.2398724
    fair_obj.climate_configs["sigma_xi"][0] = 0.439133746
    fair_obj.climate_configs["sigma_eta"][0] = 0.497438603
    fair_obj.climate_configs["forcing_4co2"][0] = 7.378788155
    fair_obj.climate_configs["seed"][0] = 23
    fair_obj.climate_configs["use_seed"][0] = True
    fair_obj.climate_configs["stochastic_run"][0] = True
    fair_obj.fill_species_configs()
    fair_obj.species_configs["baseline_concentration"][0, :] = [277, 731, 270]
    fair_obj.species_configs["forcing_reference_concentration"][0, :] = [277, 731, 270]
    fair_obj.concentration[0, 0, 0, :] = [277, 731, 270]
    fair_obj.concentration[1, 0, 0, :] = [410, 1900, 325]
    fair_obj.forcing[0, 0, 0, :] = 0
    fair_obj.temperature[0, 0, 0, :] = 0
    fair_obj.cumulative_emissions[0, 0, 0, :] = 0
    fair_obj.airborne_emissions[0, 0, 0, :] = 0
    return fair_obj


def test_aci_method():
    f.aci_method = "SMITH2021"
    assert f.aci_method == "smith2021"


def test_invalid_aci_method():
    with pytest.raises(ValueError):
        f.aci_method = "Quaas2022"


def test_ghg_method():
    f.ghg_method = "LEACH2021"
    assert f.ghg_method == "leach2021"


def test_invalid_ghg_method():
    with pytest.raises(ValueError):
        f.ghg_method = "Quaas2022"


def test_ch4_method():
    f.ch4_method = "LEACH2021"
    assert f.ch4_method == "leach2021"


def test_invalid_ch4_method():
    with pytest.raises(ValueError):
        f.ch4_method = "Quaas2022"


def test_specie_not_in_properties():
    species, properties = read_properties()
    with pytest.raises(ValueError):
        f.define_species(["Kryponite"], properties)


def test_species_invalid_type():
    species, properties = read_properties()
    properties["N2O"]["type"] = "laughing gas"
    with pytest.raises(ValueError):
        f.define_species(species, properties)


def test_species_invalid_input_mode():
    species, properties = read_properties()
    properties["N2O"]["input_mode"] = "inhaled"
    with pytest.raises(ValueError):
        f.define_species(species, properties)


def test_duplicate_species():
    species, properties = read_properties()
    properties["CO2 FFI"]["type"] = "co2"
    properties["CO2 AFOLU"]["type"] = "co2"
    with pytest.raises(ValueError):
        f.define_species(species, properties)


def test_allocate_before_definitions():
    with pytest.raises(AttributeError):
        f.allocate()


def test_ghg_routines():
    EXPECTED_RESULTS = {
        "myhre1998": np.array([2.2028445, 0.44916397, 0.19313647]),
        "etminan2016": np.array([2.21891894, 0.55302311, 0.18624564]),
        "meinshausen2020": np.array([2.1849852, 0.55574659, 0.18577101]),
        "leach2021": np.array([2.20722625, 0.54091863, 0.18102735]),
    }
    for method, results in EXPECTED_RESULTS.items():
        ftest = minimal_ghg_run()
        ftest.ghg_method = method
        ftest.run(progress=False)
        forcing_out = ftest.forcing.squeeze()
        np.testing.assert_allclose(forcing_out[1, ...], results)


def test_calculate_iirf0():
    ftest = minimal_ghg_run()
    ftest.calculate_iirf0()
    np.testing.assert_allclose(
        ftest.species_configs["iirf_0"],
        np.array([[52.35538747, 8.2499551, 65.44969575]]),
    )


def test_calculate_g():
    ftest = minimal_ghg_run()
    ftest.calculate_g()
    np.testing.assert_allclose(
        ftest.species_configs["g0"],
        np.array([[0.01017829, 0.36785517, 0.07675559]]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ftest.species_configs["g1"],
        np.array([[11.41262243, 8.24941081, 25.49528818]]),
        rtol=1e-6,
    )


def test_fill_from_rcmip_del_statement():
    ftest = minimal_ghg_run()
    ftest.fill_from_rcmip()


def test_fill_from_rcmip_species_not_recognised():
    species = ["Carbon dioxide"]
    scenarios = ["ssp119"]
    ftest = FAIR()
    for input_mode in ["emissions", "concentration", "forcing"]:
        properties = {
            "Carbon dioxide": {
                "input_mode": input_mode,
                "type": "co2",
            }
        }
        ftest.define_scenarios(scenarios)
        ftest.define_species(species, properties)
        with pytest.raises(ValueError):
            ftest.fill_from_rcmip()


def test_climate_configs_nan_raises_error():
    ftest = minimal_ghg_run()
    ftest.climate_configs["ocean_heat_capacity"][0, :] = np.array([np.nan] * 3)
    with pytest.raises(ValueError):
        ftest.run()


def test_stochastic_run_nan_raises_error():
    ftest = minimal_ghg_run()
    ftest.climate_configs["sigma_eta"][0] = np.nan
    with pytest.raises(ValueError):
        ftest.run()


def test_deterministic_run():
    ftest = minimal_ghg_run()
    ftest.climate_configs["stochastic_run"][0] = False
    ftest.run()


def test_raise_if_nan():
    ftest = minimal_ghg_run()
    ftest.concentration[0, 0, 0, :] = [np.nan] * 3
    with pytest.raises(ValueError):
        ftest.run()
    ftest = minimal_ghg_run()
    species = ["CO2", "CH4", "N2O"]
    species, properties = read_properties(species=species)
    for specie in species:
        properties[specie]["input_mode"] = "emissions"
    ftest.define_species(species, properties)
    ftest.emissions[0, 0, 0, :] = [np.nan] * 3
    with pytest.raises(ValueError):
        ftest.run()
    ftest = minimal_ghg_run()
    species = ["CO2", "CH4", "N2O"]
    species, properties = read_properties(species=species)
    for specie in species:
        properties[specie]["input_mode"] = "forcing"
    ftest.define_species(species, properties)
    ftest.emissions[0, 0, 0, :] = [np.nan] * 3
    with pytest.raises(ValueError):
        ftest.run()


def test_prescribed_temperature_raise_if_nan():
    ftest = minimal_ghg_run()
    ftest.temperature_prescribed = True
    with pytest.raises(ValueError):
        ftest.run()
