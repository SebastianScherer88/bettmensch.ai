import os

import pytest
from bettmensch_ai.pipelines.component import as_torch_ddp
from bettmensch_ai.pipelines.component.examples import (
    lightning_train,
    tensor_reduce,
)


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
        (tensor_reduce, [5, 2]),
        (lightning_train, []),
    ],
)
def test_torch_distribute_decorator(
    test_output_dir, test_function, test_function_args
):
    """Test the torch_distribute decorator with 3 test functions."""

    torch_distribute_decorator = as_torch_ddp(
        log_dir=os.path.join(test_output_dir, "logs"), max_restarts=0
    )
    torch_distributed_function = torch_distribute_decorator(test_function)
    torch_distributed_function(*test_function_args)
