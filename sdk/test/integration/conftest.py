import os

import pytest


@pytest.fixture
def test_output_dir():
    return os.path.join(".", "sdk", "test", "integration", "outputs")
