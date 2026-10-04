import numpy as np
import pandas as pd
import pytest
from fair import FAIR

from fair_calibrate import carbon_cycle_override as cco


def _table(n=5):
    cols = cco.IIRF_COLUMNS + cco.PARTITION_COLUMNS + cco.LIFETIME_COLUMNS
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((n, len(cols))), columns=cols)
    df.insert(0, "member", np.arange(n))
    return df


def test_unset_is_a_noop(monkeypatch):
    monkeypatch.delenv("CARBON_CYCLE_FILE", raising=False)
    assert cco.load_override(5) is None


def test_load_and_config(monkeypatch, tmp_path):
    path = tmp_path / "cc.csv"
    _table().to_csv(path, index=False)
    monkeypatch.setenv("CARBON_CYCLE_FILE", str(path))
    df = cco.load_override(5)
    config = cco.carbon_cycle_config(df, slice(1, 3))
    assert config["partition_fraction"].shape == (3, 4)
    assert config["unperturbed_lifetime_co2"].shape == (3, 4)
    assert len(config["iirf_0"]) == 3
    assert list(cco.dump_columns(df).columns) == [f"{c}[CO2]" for c in cco.IIRF_COLUMNS]


def test_rejects_missing_column_and_short_file(monkeypatch, tmp_path):
    path = tmp_path / "cc.csv"
    _table().drop(columns=["iirf_uptake"]).to_csv(path, index=False)
    monkeypatch.setenv("CARBON_CYCLE_FILE", str(path))
    with pytest.raises(SystemExit, match="missing columns"):
        cco.load_override(5)
    _table().to_csv(path, index=False)
    with pytest.raises(SystemExit, match="rows"):
        cco.load_override(50)


def test_set_boxes_sets_each_box_per_config():
    f = FAIR()
    f.define_time(1750, 1760, 1)
    f.define_scenarios(["s"])
    f.define_configs(["a", "b", "c"])
    f.define_species(["CO2"], {"CO2": {"type": "co2", "input_mode": "emissions",
                                        "greenhouse_gas": True,
                                        "aerosol_chemistry_from_emissions": False,
                                        "aerosol_chemistry_from_concentration": False}})
    f.allocate()
    pf = np.arange(12, dtype=float).reshape(3, 4)
    ul = pf + 100
    cco.set_boxes(f, pf, ul)
    got = f.species_configs["partition_fraction"].loc[dict(specie="CO2")].values
    assert np.array_equal(got, pf)
    got = f.species_configs["unperturbed_lifetime"].loc[dict(specie="CO2")].values
    assert np.array_equal(got, ul)
