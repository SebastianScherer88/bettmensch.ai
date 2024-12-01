import inspect
from typing import Any, Callable, Dict, List, Union

from .io import InputArtifact, InputParameter, OutputArtifact, OutputParameter


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

    func_args: Dict[str, inspect.Parameter] = inspect.signature(
        func
    ).parameters  # noqa: E501

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
            annotations are something other than the types listed in the
            argument_types argument
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


def build_container_ios(
    container: Union["Component", "Pipeline"],  # noqa: F821
    func: Callable,
    annotation_types: List[
        Union[InputParameter, InputArtifact, OutputParameter, OutputArtifact]
    ],
) -> Dict[str, Union[InputArtifact, OutputParameter, OutputArtifact]]:
    """Builds the container's template's inputs/outputs based on the
    underlying function's arguments annotated with the
    - InputParameter for the template inputs or
    - OutputsParameter or the
    - OutputArtifact
    for the template outputs. To be used in the `build_task_factory`
    method.

    Note that InputParameter type arguments dont need to be passed
    explicitly to hera's  @script decorator since they are inferred from
    the decorated function's argument spec automatically.

    Args:
        func (Callable): For a Component type contaienr, the function the
            we want to wrap. For a Pipeline type container, the function
            that defines the pipeline's DAG.
        annotation_types:
            List[Union[InputArtifact,OutputParameter,OutputArtifact]]: The
            annotation types to extract.
    Returns:
        Dict[str,Union[
                InputParameter,
                InputArtifact,
                OutputParameter,
                OutputArtifact
                ]
            ]: For a Copmonent type container, its template's
            inputs/outputs. For a Pipeline type container, its DAG's
            inputs/outputs.
    """

    func_ios = get_func_args(func, "annotation", annotation_types)

    template_ios = {}

    for io_name, io_param in func_ios.items():
        template_io = io_param.annotation(name=io_name)
        template_io.set_owner(container)

        template_ios[io_name] = template_io

    return template_ios
