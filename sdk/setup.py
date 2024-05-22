import os
from typing import List

from setuptools import find_packages, setup


def get_requirements(requirements_file: str) -> List[str]:
    """Read requirements from requirements.in."""

    file_path = os.path.join(os.path.dirname(__file__), requirements_file)
    with open(file_path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if not line.startswith("#") and line]
    return lines


setup(
    name="bettmensch_ai",
    version="0.1.0",
    author="Sebastian Scherer @ Github:SebastianScherer88",
    author_email="scherersebastian@yahoo.de",
    packages=find_packages(),
    license="LICENSE.txt",
    description="A python SDK for creating and managing bettmensch.ai Pipelines & Flows.",
    long_description=open("README.md").read(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.8.0,<3.13.0",
    include_package_data=True,
)
