import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Override global test data download for RMT unit tests."""
    return None
