import os

import pytest


@pytest.fixture
def test_output_dir():
    return os.path.join(".", "sdk", "test", "k8s", "outputs")


@pytest.fixture
def test_namespace():
    return "argo"
