import os

import pytest
from bettmensch_ai.scripts.torch_ddp_test import distributed_test_ddp_function
from bettmensch_ai.torch_utils import torch_distribute


def distributed_test_function_1():
    print("Executing test_function_1")


def distributed_test_function_2(a: int, b: str = "test"):
    print("Executing test_function_2")
    print(f"Value of a: {a}")
    print(f"Value of b: {b}")


@pytest.mark.parametrize(
    "test_function,test_function_args",
    [
        (distributed_test_function_1, []),
        (distributed_test_function_2, [1, "test_value"]),
        (distributed_test_ddp_function, [10, 5]),
    ],
)
def test_torch_distribute_decorator(test_function, test_function_args):
    """Test the torch_distribute decorator with 3 test functions."""

    test_log_dir = os.path.join(".", "sdk", "test", "unit", "logs")

    torch_distribute_decorator = torch_distribute(log_dir=test_log_dir)
    torch_distributed_function = torch_distribute_decorator(test_function)
    torch_distributed_function(*test_function_args)
