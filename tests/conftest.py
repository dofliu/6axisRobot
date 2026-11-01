import numpy as np
import pytest

from sixaxis import SixAxisRobot
from sixaxis.pbsim import PBRobot


@pytest.fixture(scope="session")
def robot():
    return SixAxisRobot()


@pytest.fixture(scope="session")
def pbr(robot):
    pr = PBRobot(robot)
    yield pr
    pr.close()


@pytest.fixture
def rng():
    return np.random.default_rng(20261101)
