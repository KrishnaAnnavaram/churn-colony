import os

# Few native threads: fast and stable on CI runners and on machines with many cores.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import pytest  # noqa: E402

from churn_colony.schema import split_xy, validate  # noqa: E402
from churn_colony.synthetic import generate  # noqa: E402


@pytest.fixture(scope="session")
def raw_small():
    return generate(800, seed=3)


@pytest.fixture(scope="session")
def clean_small(raw_small):
    df, _ = validate(raw_small)
    return df


@pytest.fixture(scope="session")
def xy_small(clean_small):
    return split_xy(clean_small)
