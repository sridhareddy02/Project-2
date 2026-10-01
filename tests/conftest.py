import pytest

from lifecycle_lab.features import labelled_snapshot
from lifecycle_lab.rfm import rfm_scores
from lifecycle_lab.simulate import simulate


@pytest.fixture(scope="session")
def world():
    return simulate(seed=5, new_base=150)


@pytest.fixture(scope="session")
def train(world):
    return labelled_snapshot(world.customers, world.orders, "2024-06-30")


@pytest.fixture(scope="session")
def test_snap(world):
    return labelled_snapshot(world.customers, world.orders, "2025-06-30")


@pytest.fixture(scope="session")
def scored(test_snap):
    return rfm_scores(test_snap)
