"""Shared pytest fixtures."""
import pytest
import numpy as np


@pytest.fixture
def default_spx_strikes():
    """A typical SPX strike grid in log-moneyness, k ∈ [-0.4, 0.2], 21 points."""
    return np.linspace(-0.4, 0.2, 21)


@pytest.fixture
def default_maturities():
    """Five SPX-typical maturities in years."""
    return np.array([0.083, 0.25, 0.5, 1.0, 2.0])


@pytest.fixture
def default_lh_params():
    """Default lifted Heston test parameters."""
    return dict(H=0.10, n=20, r_n=2.5, nu=0.4, rho=-0.7)


@pytest.fixture
def default_ab_params():
    """Default aBergomi test parameters."""
    return dict(H=0.10, n=20, eta=2.0, rho=-0.7, kernel="l2")
