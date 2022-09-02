"""Module for ERFaci tests."""

import numpy as np

from fair.forcing.aerosol.erfaci import leach2021aci, smith2021, stevens2015


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
