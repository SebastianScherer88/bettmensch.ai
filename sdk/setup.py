from enum import Enum
from typing import Dict, List

from setuptools import find_packages, setup


class SDKExtras(Enum):
    dashboard: str = "dashboard"
    pipelines: str = "pipelines"
    pipelines_adapter: str = "pipelines-adapter"
    annotated_transformer: str = "annotated-transformer"
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
            "hera==5.15.1",
        ],
        SDKExtras.pipelines.value: [
            "hera==5.15.1",
            "GPUtil==1.4.0",
        ],
        SDKExtras.pipelines_adapter.value: [
            "boto3==1.35.59",
        ],
        SDKExtras.serving.value: ["fastapi==0.112.0"],
    }

    extra_requirements[SDKExtras.test.value] = (
        extra_requirements[SDKExtras.pipelines.value]
        + [
            "torch==2.3.1",
            "lightning==2.4.0",
            "numpy==1.24.1",
            "scipy==1.14.1",
            "pytest==8.2.2",
            "pytest-order==1.2.1",
        ],
    )

    extra_requirements[SDKExtras.annotated_transformer.value] = [
        "torchtext==0.18.0",
        "torchdata==0.9.0",
        "portalocker==2.10.1",
        "spacy==3.8.2",
    ]

    return extra_requirements


setup(
    name="bettmensch_ai",
    version="0.2.0",
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
