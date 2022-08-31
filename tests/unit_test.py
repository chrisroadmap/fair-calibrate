"""Module for unit tests."""

import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fair.energy_balance_model import EnergyBalanceModel
from fair.forcing.aerosol.erfaci import leach2021aci, smith2021, stevens2015
from fair.forcing.ghg import etminan2016, leach2021ghg, meinshausen2020, myhre1998
from fair.interface import fill

HERE = os.path.dirname(os.path.realpath(__file__))

# this model has the potential to cause problems for scipy's linear algebra
EBM_CAMS_STOCHASTIC = EnergyBalanceModel(
    ocean_heat_capacity=[2.632438882, 9.262194928, 52.92769715],
    ocean_heat_transfer=[1.876253552, 5.15359085, 0.643546006],
    deep_ocean_efficacy=1.285458434,
    gamma_autocorrelation=28.2398724,
    sigma_xi=0.439493317,
    sigma_eta=2.690512385,
    forcing_4co2=8.870602356,
    seed=23,
    stochastic_run=True,
    n_timesteps=5,
)

EBM_CAMS_DETERMINISTIC = EnergyBalanceModel(
    ocean_heat_capacity=[2.632438882, 9.262194928, 52.92769715],
    ocean_heat_transfer=[1.876253552, 5.15359085, 0.643546006],
    deep_ocean_efficacy=1.285458434,
    gamma_autocorrelation=28.2398724,
    forcing_4co2=8.870602356,
    seed=None,
    stochastic_run=False,
    n_timesteps=5,
)


def test_ebm_init_array_mismatch_error():
    with pytest.raises(ValueError):
        EnergyBalanceModel(
            ocean_heat_capacity=[2, 10, 75], ocean_heat_transfer=[1.0, 3.0]
        )


def test_ebm_init_ocean_layers_less_than_two_error():
    with pytest.raises(ValueError):
        EnergyBalanceModel(ocean_heat_capacity=[2], ocean_heat_transfer=[1.0])


def test_ebm_stochastic_d():
    np.testing.assert_array_equal(EBM_CAMS_DETERMINISTIC.stochastic_d, 0)


def test_ebm_emergent_parameters():
    EBM_CAMS_STOCHASTIC.impulse_response()
    EBM_CAMS_STOCHASTIC.emergent_parameters()
    # Generates the test data. Uncomment next lines if you want it.
    # np.savetxt(os.path.join(
    #     HERE, "test_data", "ebm3_cams-csm1-0_timescales.txt"),
    #     EBM_CAMS_STOCHASTIC.timescales
    # )
    # np.savetxt(os.path.join(
    #     HERE, "test_data", "ebm3_cams-csm1-0_response_coefficients.txt"),
    #     EBM_CAMS_STOCHASTIC.response_coefficients
    # )
    # np.savetxt(os.path.join(
    #     HERE, "test_data", "ebm3_cams-csm1-0_ecs_tcr.txt"),
    #     np.array([EBM_CAMS_STOCHASTIC.ecs, EBM_CAMS_STOCHASTIC.tcr])
    # )
    timescales = np.loadtxt(
        os.path.join(HERE, "test_data", "ebm3_cams-csm1-0_timescales.txt")
    )
    response_coefficients = np.loadtxt(
        os.path.join(HERE, "test_data", "ebm3_cams-csm1-0_response_coefficients.txt")
    )
    ecs_tcr = np.loadtxt(
        os.path.join(HERE, "test_data", "ebm3_cams-csm1-0_ecs_tcr.txt")
    )
    np.testing.assert_allclose(EBM_CAMS_STOCHASTIC.timescales, timescales)
    np.testing.assert_allclose(
        EBM_CAMS_STOCHASTIC.response_coefficients, response_coefficients
    )
    np.testing.assert_allclose(
        np.array([EBM_CAMS_STOCHASTIC.ecs, EBM_CAMS_STOCHASTIC.tcr]), ecs_tcr
    )


@pytest.mark.filterwarnings("ignore:covariance is not positive-semidefinite")
def test_ebm_run():
    EBM_CAMS_STOCHASTIC.add_forcing(np.zeros(5), timestep=1)
    EBM_CAMS_STOCHASTIC.run()
    # Generates the test data. Uncomment next lines if you want it.
    # np.savetxt(os.path.join(
    #     HERE, "test_data", "ebm3_cams-csm1-0_temperature.txt"),
    #     EBM_CAMS_STOCHASTIC.temperature
    # )
    temperature = np.loadtxt(
        os.path.join(HERE, "test_data", "ebm3_cams-csm1-0_temperature.txt")
    )
    # implement a fairly generous absolute tolerance on the temperature differences
    # because the scipy linalg routines seem to change with each version, and if they
    # are out by less than one microkelvin I am sure we can accept this.
    np.testing.assert_allclose(EBM_CAMS_STOCHASTIC.temperature, temperature, atol=1e-6)


def test_aerosol_erfaci_leach2021aci():
    erfaci = leach2021aci(
        emissions=np.array([[[[100, 10, 35]]]]),
        baseline_emissions=np.array([[[[2, 2, 10]]]]),
        forcing_scaling=np.ones((1, 1, 1, 3)),
        scale_sulfur=0.74,
        shape_sulfur=38.9,
        scale_bcoc=0.000265,
        sulfur_index=0,
        bc_index=1,
        oc_index=2,
    )
    np.testing.assert_almost_equal(erfaci[0, 0, 0, 0], -0.895996898166541)


def test_aerosol_erfaci_smith2021():
    erfaci = smith2021(
        emissions=np.array([[[[100, 10, 35]]]]),
        baseline_emissions=np.array([[[[2, 2, 10]]]]),
        forcing_scaling=np.ones((1, 1, 1, 3)),
        scale=0.741,
        shape_sulfur=39.5,
        shape_bcoc=228.1,
        sulfur_index=0,
        bc_index=1,
        oc_index=2,
    )
    np.testing.assert_almost_equal(erfaci[0, 0, 0, 0], -0.9024402325593872)


def test_aerosol_erfaci_stevens2015():
    erfaci = stevens2015(
        emissions=np.array([[[[100]]]]),
        baseline_emissions=np.array([[[[2]]]]),
        forcing_scaling=np.ones((1, 1, 1, 1)),
        scale=0.741,
        shape_sulfur=39.5,
        sulfur_index=0,
    )
    np.testing.assert_almost_equal(erfaci[0, 0, 0, 0], -0.8983670399523528)


def test_ghg_etminan2016():
    erfghg = etminan2016(
        np.array([[[[410, 1900, 325]]]]) * np.ones((1, 1, 1, 3)),
        np.array([[[[277, 731, 270]]]]) * np.ones((1, 1, 1, 3)),
        np.array([[[[1.05, 0.86, 1.07]]]]) * np.ones((1, 1, 1, 3)),
        np.ones((1, 1, 1, 3)),
        0,
        1,
        2,
        [False, False, False],
    ).squeeze()
    np.testing.assert_allclose(erfghg, np.array([2.21891894, 0.55302311, 0.18624564]))


def test_ghg_meinshausen2020():
    erfghg = meinshausen2020(
        np.array([[[[410, 1900, 325]]]]) * np.ones((1, 1, 1, 3)),
        np.array([[[[277, 731, 270]]]]) * np.ones((1, 1, 1, 3)),
        np.array([[[[1.05, 0.86, 1.07]]]]) * np.ones((1, 1, 1, 3)),
        np.ones((1, 1, 1, 3)),
        0,
        1,
        2,
        [False, False, False],
    ).squeeze()
    np.testing.assert_allclose(erfghg, np.array([2.1849852, 0.55574659, 0.18577101]))


def test_ghg_leach2021ghg():
    erfghg = leach2021ghg(
        np.array([[[[410, 1900, 325]]]]) * np.ones((1, 1, 1, 3)),
        np.array([[[[277, 731, 270]]]]) * np.ones((1, 1, 1, 3)),
        np.array([[[[1.05, 0.86, 1.07]]]]) * np.ones((1, 1, 1, 3)),
        np.ones((1, 1, 1, 3)),
        0,
        1,
        2,
        [False, False, False],
    ).squeeze()
    np.testing.assert_allclose(erfghg, np.array([2.20722625, 0.54091863, 0.18102735]))


def test_ghg_myhre1998():
    erfghg = myhre1998(
        np.array([[[[410, 1900, 325]]]]) * np.ones((1, 1, 1, 3)),
        np.array([[[[277, 731, 270]]]]) * np.ones((1, 1, 1, 3)),
        np.array([[[[1.05, 0.86, 1.07]]]]) * np.ones((1, 1, 1, 3)),
        np.ones((1, 1, 1, 3)),
        0,
        1,
        2,
        [False, False, False],
    ).squeeze()
    np.testing.assert_allclose(erfghg, np.array([2.2028445, 0.44916397, 0.19313647]))


def test_fill_raises_error():
    np.random.seed(0)
    temperature = 15 + 8 * np.random.randn(2, 2, 3)
    lon = [[-99.83, -99.32], [-99.79, -99.23]]
    lat = [[42.25, 42.21], [42.63, 42.59]]
    time = pd.date_range("2014-09-06", periods=3)
    reference_time = pd.Timestamp("2014-09-05")
    da = xr.DataArray(
        data=temperature,
        dims=["x", "y", "time"],
        coords=dict(
            lon=(["x", "y"], lon),
            lat=(["x", "y"], lat),
            time=time,
            reference_time=reference_time,
        ),
        attrs=dict(
            description="Ambient temperature.",
            units="degC",
        ),
    )
    with pytest.raises(ValueError):
        fill(da, 0, specie="CH4")
