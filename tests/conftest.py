import pytest

from pleat.half import IdObject


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset global ID counters before each test for deterministic behavior."""
    IdObject.reset_ids()
