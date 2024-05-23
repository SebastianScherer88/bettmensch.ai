import inspect
from typing import Any, Callable, Dict, List

COMPONENT_BASE_IMAGE = "bettmensch88/bettmensch.ai:3.11-2769afb"


def get_func_args(
    func: Callable,
    attribute_name: str = "",
    attribute_value_range: List[Any] = [],
) -> Dict[str, inspect.Parameter]:
    """Utility method to retrieve function arguments with optional filtering on
    specified argument attribute.

    Args:
        func (Callable): The function whose arguments will be retrieved.
        attribute_name (str, optional): The name of the argument attribute to
            filter on. Defaults to ''.
        attribute_value_range (List[Any], optional): The value range of the
            argument attribute to filter on. Defaults to [].
    """

    func_args: Dict[str, inspect.Parameter] = inspect.signature(func).parameters

    if attribute_name and attribute_value_range:
        func_args = dict(
            [
                (k, param)
                for k, param in func_args.items()
                if (getattr(param, attribute_name) in attribute_value_range)
            ]
        )

    return func_args


def validate_func_args(func: Callable, argument_types: List[type]):
    """Validates the function's input type annotations.

    Args:
        func (Callable): The function the we want to wrap in a Component.

    Raises:
        TypeError: Raised if any of the func arguments input argument type
            annotations are something other than
            - ComponentInput
            - ComponentOutput
    """

    func_args = get_func_args(func)
    invalid_func_args: Dict[str, inspect.Parameter] = dict(
        [
            (k, param)
            for k, param in func_args.items()
            if (param.annotation not in argument_types)
        ]
    )

    if invalid_func_args:
        raise TypeError(
            f"Invalid function argument type annotation(s): "
            f"{invalid_func_args}. All function arguments need to be annotated"
            f"with one of {argument_types} types."
        )
