"""Tests for the Python port of EBM::FitKalman (three-layer model).

Reference values in ``data/cummins_reference.json`` come from Donald Cummins' R
package (EBM v1.1.0); regenerate with ``tests/generate_cummins_reference.R``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fair.energy_balance_model import EnergyBalanceModel

from fair_calibrate import cummins_ebm as ce

DATA = Path(__file__).parent / "data"
REF = json.loads((DATA / "cummins_reference.json").read_text())
SERIES = pd.read_csv(DATA / "cummins_synthetic.csv")
DATASET = np.vstack([SERIES["T"].values, SERIES["N"].values])
TRUTH = {
    k: (np.array(v) if isinstance(v, list) else v) for k, v in REF["truth"].items()
}
INITS = dict(
    gamma=2,
    C=np.array([4, 15, 80]),
    kappa=np.array([1, 2, 1]),
    epsilon=1.1,
    sigma_eta=0.5,
    sigma_xi=0.5,
    F_4xCO2=8,
)


@pytest.fixture(scope="module")
def matrices():
    return ce.build_matrices(
        TRUTH["gamma"],
        TRUTH["C"],
        TRUTH["kappa"],
        TRUTH["epsilon"],
        TRUTH["sigma_eta"],
        TRUTH["sigma_xi"],
    )


def test_transform_roundtrip():
    p = ce.back_transform(ce.transform(TRUTH))
    for key, value in TRUTH.items():
        np.testing.assert_allclose(p[key], value, rtol=1e-12)


@pytest.mark.parametrize("name", ["A", "Q", "Ad", "Qd", "Gamma0", "Cd"])
def test_matrices_match_r(matrices, name):
    np.testing.assert_allclose(
        matrices[name], np.array(REF["matrices"][name]), rtol=1e-10, atol=1e-14
    )


@pytest.mark.parametrize("name", ["B", "Bd"])
def test_vectors_match_r(matrices, name):
    np.testing.assert_allclose(
        matrices[name],
        np.array(REF["matrices"][name]).squeeze(),
        rtol=1e-10,
        atol=1e-14,
    )


def test_drift_matrix_matches_fair(matrices):
    """FaIR's forward energy balance model uses the same drift matrix."""
    ebm = EnergyBalanceModel(
        ocean_heat_capacity=TRUTH["C"],
        ocean_heat_transfer=TRUTH["kappa"],
        deep_ocean_efficacy=TRUTH["epsilon"],
        forcing_4co2=TRUTH["F_4xCO2"],
        stochastic_run=True,
        sigma_eta=TRUTH["sigma_eta"],
        sigma_xi=TRUTH["sigma_xi"],
        gamma_autocorrelation=TRUTH["gamma"],
    )
    np.testing.assert_allclose(ebm._eb_matrix(), matrices["A"], rtol=1e-12)
    np.testing.assert_allclose(ebm.eb_matrix_d, matrices["Ad"], rtol=1e-10)


def test_loglik_matches_r(matrices):
    loglik = ce.kalman_loglik(
        matrices["Ad"],
        matrices["Bd"],
        matrices["Qd"],
        matrices["Gamma0"],
        matrices["Cd"],
        TRUTH["F_4xCO2"],
        DATASET,
    )
    assert loglik == pytest.approx(REF["loglik"], rel=1e-10)


def test_neg_log_lik_matches_r():
    value = ce.neg_log_lik(np.array(REF["par"]), DATASET, REF["alpha"])
    assert value == pytest.approx(REF["neg_log_lik"], rel=1e-10)


def test_fit_matches_r():
    result = ce.fit_kalman(
        INITS, DATASET[0], DATASET[1], alpha=REF["alpha"], maxeval=20000
    )
    ref = REF["fit"]
    assert result["neg_log_lik"] == pytest.approx(ref["neg_log_lik"], abs=1e-6)
    for key in ["gamma", "epsilon", "sigma_eta", "sigma_xi", "F_4xCO2"]:
        assert result[key] == pytest.approx(ref["p"][key], rel=1e-3)
    for key in ["C", "kappa"]:
        np.testing.assert_allclose(result[key], ref["p"][key], rtol=1e-3)
    assert not result["gamma_at_bound"]


def test_fit_raises_when_iterations_exhausted():
    with pytest.raises(ce.ConvergenceError):
        ce.fit_kalman(INITS, DATASET[0], DATASET[1], alpha=REF["alpha"], maxeval=50)


def test_fit_respects_gamma_max():
    """A bound below the unconstrained optimum (gamma = 1.97) must bind."""
    result = ce.fit_kalman(
        {**INITS, "gamma": 0.5},
        DATASET[0],
        DATASET[1],
        alpha=REF["alpha"],
        maxeval=20000,
        gamma_max=1.0,
    )
    assert result["gamma"] <= 1.0 * (1 + 1e-9)
    assert result["gamma_at_bound"]
    assert result["neg_log_lik"] > REF["fit"]["neg_log_lik"]  # bound costs fit


def test_gamma_max_must_exceed_initial_gamma():
    with pytest.raises(ValueError, match="gamma_max"):
        ce.fit_kalman(INITS, DATASET[0], DATASET[1], gamma_max=1.0)


def test_c3_constraint_inactive_leaves_fit_unchanged():
    """The truth has C3/C2 = 4.4, so C3 >= C2 must not move the optimum."""
    free = ce.fit_kalman(
        INITS, DATASET[0], DATASET[1], alpha=REF["alpha"], maxeval=20000
    )
    constrained = ce.fit_kalman(
        INITS,
        DATASET[0],
        DATASET[1],
        alpha=REF["alpha"],
        maxeval=20000,
        deep_min_ratio=1.0,
    )
    assert not constrained["deep_at_bound"]
    assert constrained["neg_log_lik"] == pytest.approx(free["neg_log_lik"], abs=1e-4)
    for key in ["C", "kappa"]:
        np.testing.assert_allclose(constrained[key], free[key], rtol=1e-2)


def test_c3_constraint_binds():
    """Requiring C3/C2 >= 8 when the data prefer about 4 must bind and cost fit."""
    free = ce.fit_kalman(
        INITS, DATASET[0], DATASET[1], alpha=REF["alpha"], maxeval=20000
    )
    result = ce.fit_kalman(
        {**INITS, "C": np.array([4, 15, 200])},  # feasible start: C3/C2 = 13
        DATASET[0],
        DATASET[1],
        alpha=REF["alpha"],
        maxeval=20000,
        deep_min_ratio=8.0,
    )
    assert result["C"][2] / result["C"][1] >= 8.0 * (1 - 1e-9)
    assert result["deep_at_bound"]
    assert result["neg_log_lik"] > free["neg_log_lik"]


def test_deep_min_ratio_must_hold_for_initial_values():
    with pytest.raises(ValueError, match="deep_min_ratio"):
        ce.fit_kalman(INITS, DATASET[0], DATASET[1], deep_min_ratio=10.0)


def _fit(**overrides):
    base = {"epsilon": 1.2, "deep_at_bound": False}
    return {**base, **overrides}


def test_suspect_reasons_clean_fit():
    assert ce.suspect_reasons(_fit(), start_gap=0.01) == []


@pytest.mark.parametrize("epsilon", [0.49, 0.159, 2.51, 28379.7])
def test_suspect_reasons_epsilon_outside_range(epsilon):
    reasons = ce.suspect_reasons(_fit(epsilon=epsilon), start_gap=0.0)
    assert len(reasons) == 1 and reasons[0].startswith("epsilon")


@pytest.mark.parametrize("epsilon", [0.5, 1.0, 2.5])
def test_suspect_reasons_epsilon_range_is_inclusive(epsilon):
    assert ce.suspect_reasons(_fit(epsilon=epsilon), start_gap=0.0) == []


def test_suspect_reasons_other_flags():
    assert "deepest/above heat capacity ratio on its bound" in ce.suspect_reasons(
        _fit(deep_at_bound=True)
    )
    assert "only one start converged" in ce.suspect_reasons(_fit(), start_gap=np.nan)
    assert any(
        r.startswith("starts differ") for r in ce.suspect_reasons(_fit(), start_gap=0.5)
    )


# High gamma (30): scipy's Van Loan matrix exponential returns an indefinite Qd
# here (min eigenvalue about -1e8), which made the filter fail. The Lyapunov route
# must give a valid covariance. R's own Qd is only accurate to about 1e-5 at this
# gamma, so the comparison with R is loose.
STRESS = REF["stress"]


@pytest.fixture(scope="module")
def stress_matrices():
    p = ce.back_transform(np.array(STRESS["par"]))
    return p, ce.build_matrices(
        p["gamma"], p["C"], p["kappa"], p["epsilon"], p["sigma_eta"], p["sigma_xi"]
    )


@pytest.mark.parametrize("name", ["Qd", "Gamma0"])
def test_high_gamma_covariances_are_valid(stress_matrices, name):
    cov = stress_matrices[1][name]
    np.testing.assert_allclose(cov, cov.T, rtol=1e-9, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh((cov + cov.T) / 2)
    assert eigenvalues.min() > -1e-10 * eigenvalues.max()


@pytest.mark.parametrize("name", ["Ad", "Qd", "Gamma0", "Cd"])
def test_high_gamma_matrices_close_to_r(stress_matrices, name):
    ref = np.array(STRESS["matrices"][name])
    got = stress_matrices[1][name]
    assert np.abs(got - ref).max() <= 1e-4 * np.abs(ref).max()


def test_high_gamma_loglik_close_to_r(stress_matrices):
    p, m = stress_matrices
    loglik = ce.kalman_loglik(
        m["Ad"], m["Bd"], m["Qd"], m["Gamma0"], m["Cd"], p["F_4xCO2"], DATASET
    )
    assert np.isfinite(loglik)
    assert loglik == pytest.approx(STRESS["loglik"], abs=1e-2)


# Two-layer model: reference values from EBM's own k = 2 code.
TWO = REF["two_layer"]
TWO_INITS = {
    key: (np.array(value) if isinstance(value, list) else value)
    for key, value in TWO["inits"].items()
}


@pytest.fixture(scope="module")
def two_layer_matrices():
    p = ce.back_transform(np.array(TWO["par"]))
    assert len(p["C"]) == 2
    return p, ce.build_matrices(
        p["gamma"], p["C"], p["kappa"], p["epsilon"], p["sigma_eta"], p["sigma_xi"]
    )


@pytest.mark.parametrize("name", ["A", "Ad", "Qd", "Gamma0", "Cd"])
def test_two_layer_matrices_match_r(two_layer_matrices, name):
    np.testing.assert_allclose(
        two_layer_matrices[1][name],
        np.array(TWO["matrices"][name]),
        rtol=1e-10,
        atol=1e-14,
    )


def test_two_layer_bd_matches_r(two_layer_matrices):
    np.testing.assert_allclose(
        two_layer_matrices[1]["Bd"],
        np.array(TWO["matrices"]["Bd"]).squeeze(),
        rtol=1e-10,
        atol=1e-14,
    )


def test_two_layer_drift_matrix_matches_fair(two_layer_matrices):
    p, m = two_layer_matrices
    ebm = EnergyBalanceModel(
        ocean_heat_capacity=p["C"],
        ocean_heat_transfer=p["kappa"],
        deep_ocean_efficacy=p["epsilon"],
        forcing_4co2=p["F_4xCO2"],
        stochastic_run=True,
        sigma_eta=p["sigma_eta"],
        sigma_xi=p["sigma_xi"],
        gamma_autocorrelation=p["gamma"],
    )
    np.testing.assert_allclose(ebm._eb_matrix(), m["A"], rtol=1e-12)


def test_two_layer_loglik_matches_r(two_layer_matrices):
    p, m = two_layer_matrices
    loglik = ce.kalman_loglik(
        m["Ad"], m["Bd"], m["Qd"], m["Gamma0"], m["Cd"], p["F_4xCO2"], DATASET
    )
    assert loglik == pytest.approx(TWO["loglik"], rel=1e-10)
    nll = ce.neg_log_lik(np.array(TWO["par"]), DATASET, REF["alpha"])
    assert nll == pytest.approx(TWO["neg_log_lik"], rel=1e-10)


def test_two_layer_fit_matches_r():
    result = ce.fit_kalman(
        TWO_INITS, DATASET[0], DATASET[1], alpha=REF["alpha"], maxeval=20000
    )
    ref = TWO["fit"]
    assert len(result["C"]) == 2 and len(result["kappa"]) == 2
    assert result["neg_log_lik"] == pytest.approx(ref["neg_log_lik"], abs=1e-4)
    for key in ["gamma", "epsilon", "sigma_eta", "sigma_xi", "F_4xCO2"]:
        assert result[key] == pytest.approx(ref["p"][key], rel=1e-2)
    for key in ["C", "kappa"]:
        np.testing.assert_allclose(result[key], ref["p"][key], rtol=1e-2)


def test_two_layer_deep_min_ratio_applies_to_c2_over_c1():
    """Demand twice the C2/C1 that R's unconstrained fit found: it must bind."""
    free_ratio = TWO["fit"]["p"]["C"][1] / TWO["fit"]["p"]["C"][0]
    bound = 2 * free_ratio
    result = ce.fit_kalman(
        {**TWO_INITS, "C": np.array([7.5, 7.5 * 4 * free_ratio])},  # feasible start
        DATASET[0],
        DATASET[1],
        alpha=REF["alpha"],
        maxeval=20000,
        deep_min_ratio=bound,
    )
    assert result["C"][1] / result["C"][0] >= bound * (1 - 1e-9)
    assert result["deep_at_bound"]


def test_back_transform_rejects_other_layer_counts():
    with pytest.raises(ValueError, match="parameters"):
        ce.back_transform(np.zeros(10))
