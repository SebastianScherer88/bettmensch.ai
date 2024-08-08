from enum import Enum
from typing import Dict, List

from setuptools import find_packages, setup


class SDKExtras(Enum):
    dashboard: str = "dashboard"
    pipelines: str = "pipelines"
    torch_pipelines: str = "torch-pipelines"
    serving: str = "serving"
    test: str = "test"


def get_extra_requirements() -> Dict[str, List[str]]:
    """Produces the requirements dictionary for the sdk's extras."""

    extra_requirements = {
        SDKExtras.dashboard.value: [
            "streamlit==1.37.1",
            "streamlit-option-menu==0.3.13",
            "st-pages==0.5.0",
            "streamlit-extras==0.4.6",
            "streamlit-flow-component<1.0.0",
            "opencv-python",
            "argo-workflows==6.5.6",
        ],
        SDKExtras.pipelines.value: [
            "hera==5.15.1",
            "argo-workflows==6.5.6",
        ],
        SDKExtras.serving.value: ["fastapi==0.112.0"],
        SDKExtras.test.value: [
            "pytest==8.2.2",
            "pytest-order==1.2.1",
        ],
    }

    extra_requirements[SDKExtras.torch_pipelines.value] = extra_requirements[
        SDKExtras.pipelines.value
    ] + [
        "torch==2.2.2",
        "lightning==2.4.0",
        "numpy==1.24.1",
    ]

    return extra_requirements


setup(
    name="bettmensch_ai",
    version="0.1.0",
    author="Sebastian Scherer @ Github:SebastianScherer88",
    author_email="scherersebastian@yahoo.de",
    packages=find_packages(),
    license="LICENSE.txt",
    description="A python SDK for creating and managing bettmensch.ai Pipelines & Flows.",  # noqa: E501
    long_description=open("README.md").read(),
    install_requires=[
        "pydantic==2.6.4",
        "pydantic-settings==2.2.1",
        "PyYAML==6.0.1",
    ],
    extras_require=get_extra_requirements(),
    python_requires=">=3.8.0,<3.13.0",
    include_package_data=True,
)
