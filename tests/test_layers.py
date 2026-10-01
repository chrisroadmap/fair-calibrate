"""Tests for the layer-count helpers the sampling and constraining scripts share."""

import numpy as np
import pandas as pd
import pytest
from fair import FAIR
from fair.interface import fill, initialise

from fair_calibrate import layers, parameters


def test_committed_n_layers_is_supported():
    assert layers.get_n_layers() == parameters.N_LAYERS
    assert parameters.N_LAYERS in layers.SUPPORTED_LAYERS


@pytest.mark.parametrize("value", [2, 3])
def test_get_n_layers_reads_parameters(monkeypatch, value):
    monkeypatch.setattr(parameters, "N_LAYERS", value)
    assert layers.get_n_layers() == value


def test_environment_variable_does_not_override_parameters(monkeypatch):
    """A machine-local .env must not contradict the committed calibration."""
    monkeypatch.setattr(parameters, "N_LAYERS", 2)
    monkeypatch.setenv("N_LAYERS", "3")
    assert layers.get_n_layers() == 2


@pytest.mark.parametrize("value", [1, 4, 0])
def test_get_n_layers_rejects_unsupported(monkeypatch, value):
    monkeypatch.setattr(parameters, "N_LAYERS", value)
    with pytest.raises(SystemExit, match="parameters.py"):
        layers.get_n_layers()


def test_file_names_carry_the_layer_count():
    assert layers.calibration_file(2) == "4xCO2_cummins_ebm2_cmip6.csv"
    assert layers.calibration_file(3) == "4xCO2_cummins_ebm3_cmip6.csv"
    assert layers.climate_response_file(2) == "climate_response_ebm2.csv"
    assert layers.climate_response_file(3) == "climate_response_ebm3.csv"


def test_three_layer_columns_and_renames_are_unchanged():
    """The three-layer names must be exactly what the scripts hard-coded before."""
    assert layers.climate_response_columns(3) == [
        "gamma",
        "c1",
        "c2",
        "c3",
        "kappa1",
        "kappa2",
        "kappa3",
        "epsilon",
        "sigma_eta",
        "sigma_xi",
        "F_4xCO2",
    ]
    assert layers.climate_response_renames(3) == {
        "gamma": "gamma_autocorrelation",
        "c1": "ocean_heat_capacity[0]",
        "c2": "ocean_heat_capacity[1]",
        "c3": "ocean_heat_capacity[2]",
        "kappa1": "ocean_heat_transfer[0]",
        "kappa2": "ocean_heat_transfer[1]",
        "kappa3": "ocean_heat_transfer[2]",
        "epsilon": "deep_ocean_efficacy",
        "sigma_eta": "sigma_eta",
        "sigma_xi": "sigma_xi",
        "F_4xCO2": "forcing_4co2",
    }


def test_two_layer_columns_and_renames():
    assert layers.climate_response_columns(2) == [
        "gamma",
        "c1",
        "c2",
        "kappa1",
        "kappa2",
        "epsilon",
        "sigma_eta",
        "sigma_xi",
        "F_4xCO2",
    ]
    renames = layers.climate_response_renames(2)
    assert list(renames) == layers.climate_response_columns(2)
    assert "ocean_heat_capacity[2]" not in renames.values()
    assert renames["kappa2"] == "ocean_heat_transfer[1]"


def _prior(n_layers, n_rows=6):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.uniform(1, 2, (n_rows, 2 * n_layers + 5)),
        columns=layers.climate_response_columns(n_layers),
    )
    for i in range(n_layers):
        df[f"c{i + 1}"] = 5 * 4**i + df[f"c{i + 1}"]  # increasing with depth
    return df


@pytest.mark.parametrize("n_layers", [2, 3])
def test_layer_config_picks_the_right_rows_and_columns(n_layers):
    df = _prior(n_layers)
    cfg = layers.layer_config(df, slice(1, 3), n_layers)  # label slice, inclusive
    assert cfg["n_layers"] == n_layers
    layer_keys = set(cfg) - {"n_layers"}
    assert layer_keys == set(
        layers.heat_capacity_columns(n_layers) + layers.heat_transfer_columns(n_layers)
    )
    np.testing.assert_array_equal(cfg["c1"], df.loc[1:3, "c1"].values)
    assert all(len(cfg[key]) == 3 for key in layer_keys)


BASELINE = {"CO2": 284.3169988, "CH4": 808.2490285, "N2O": 273.021047}


def _small_fair(n_layers, cfg, n_configs):
    f = FAIR(n_layers=n_layers)
    f.define_time(1850, 1860, 1)
    f.define_scenarios(["test"])
    f.define_configs(list(range(n_configs)))
    # FaIR needs CO2, CH4 and N2O together to compute their forcing
    properties = {
        specie: {
            "type": specie.lower(),
            "input_mode": "concentration",
            "greenhouse_gas": True,
            "aerosol_chemistry_from_emissions": False,
            "aerosol_chemistry_from_concentration": False,
        }
        for specie in BASELINE
    }
    f.define_species(list(BASELINE), properties)
    f.allocate()
    return f


@pytest.mark.parametrize("n_layers", [2, 3])
def test_fill_layers_and_run_fair(n_layers):
    """FaIR runs with the layer arrays the helpers fill, for two and three layers."""
    df = _prior(n_layers, n_rows=4)
    cfg = layers.layer_config(df, slice(0, 3), n_layers)
    f = _small_fair(n_layers, cfg, n_configs=4)
    layers.fill_layers(f, cfg)
    fill(f.climate_configs["deep_ocean_efficacy"], 1.2)
    fill(f.climate_configs["gamma_autocorrelation"], 2.0)
    fill(f.climate_configs["stochastic_run"], False)
    fill(f.climate_configs["forcing_4co2"], 8.0)
    f.fill_species_configs()
    for specie, value in BASELINE.items():
        fill(f.species_configs["baseline_concentration"], value, specie=specie)
        fill(f.species_configs["forcing_reference_concentration"], value, specie=specie)
        fill(f.concentration, value, specie=specie)
    # CO2 rises 1% a year; the other gases stay at their baseline
    f.concentration.loc[dict(specie="CO2")] = (
        BASELINE["CO2"] * 1.01 ** np.arange(11)[:, None, None]
    )
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    f.run(progress=False)

    capacity = f.climate_configs["ocean_heat_capacity"].values
    assert capacity.shape == (4, n_layers)
    np.testing.assert_allclose(capacity[:, 0], df["c1"].values)
    np.testing.assert_allclose(capacity[:, -1], df[f"c{n_layers}"].values)
    assert f.temperature.shape[-1] == n_layers
    assert np.all(np.isfinite(f.temperature.values))
    assert f.temperature.values[-1, 0, :, 0].min() > 0  # warms under rising CO2


@pytest.mark.parametrize("n_layers", [2, 3])
def test_override_defaults_accepts_the_dumped_parameter_names(n_layers, tmp_path):
    """05 dumps columns named by climate_response_renames; 07 loads them in FaIR."""
    df = _prior(n_layers, n_rows=3).rename(
        columns=layers.climate_response_renames(n_layers)
    )
    path = tmp_path / "calibrated_constrained_parameters.csv"
    df.to_csv(path)
    f = _small_fair(n_layers, None, n_configs=3)
    f.fill_species_configs()
    f.override_defaults(str(path))
    got = f.climate_configs["ocean_heat_capacity"].values
    assert got.shape == (3, n_layers)
    np.testing.assert_allclose(got[:, 0], df["ocean_heat_capacity[0]"].values)
    np.testing.assert_allclose(
        got[:, -1], df[f"ocean_heat_capacity[{n_layers - 1}]"].values
    )
