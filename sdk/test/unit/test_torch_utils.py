import os

import pytest
from bettmensch_ai.scripts.example_components import (
    torch_ddp as distributed_test_ddp_function,
)
from bettmensch_ai.torch_utils import torch_distribute


def distributed_test_function_1():
    """Function #1 to be distributed using the bettmensch_ai.torch_distribute
    decorator. Needs to be defined in the main module scope to be pickling
    compatible."""
    print("Executing test_function_1")


def distributed_test_function_2(a: int, b: str = "test"):
    """Function #2 to be distributed using the bettmensch_ai.torch_distribute
    decorator. Needs to be defined in the main module scope to be pickling
    compatible."""
    print("Executing test_function_2")
    print(f"Value of a: {a}")
    print(f"Value of b: {b}")


@pytest.mark.parametrize(
    "test_function,test_function_args",
    [
        (distributed_test_function_1, []),
        (distributed_test_function_2, [1, "test_value"]),
        (distributed_test_ddp_function, [5, 2]),
    ],
)
def test_torch_distribute_decorator_2(
    test_output_dir, test_function, test_function_args
):
    """Test the torch_distribute decorator with 3 test functions."""

    torch_distribute_decorator = torch_distribute(
        log_dir=os.path.join(test_output_dir, "logs")
    )
    torch_distributed_function = torch_distribute_decorator(test_function)
    torch_distributed_function(*test_function_args)
