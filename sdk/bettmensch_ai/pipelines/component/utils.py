from functools import wraps
from typing import Callable, Type, Union

from hera.workflows import Script, Step, Task
from hera.workflows._context import _context
from hera.workflows.script import FuncIns, FuncR, _add_type_hints


@_add_type_hints(Script, Step, Task)  # type: ignore
def bettmensch_ai_script(
    script_class: Type[Script], **script_kwargs
) -> Callable:
    """A decorator that wraps a function into a Script object.

    Using this decorator users can define a function that will be executed as a
    script in a container. Once the `Script` is returned users can use it as
    they generally use a `Script` e.g. as a callable inside a DAG or Steps.
    Note that invoking the function will result in the template associated with
    the script to be added to the workflow context, so users do not have to
    worry about that.

    Parameters
    ----------
    **script_kwargs
        Keyword arguments to be passed to the Script object.

    Returns:
    -------
    Callable
        Function that wraps a given function into a `Script`.
    """

    def bettmensch_ai_script_wrapper(
        func: Callable[FuncIns, FuncR]
    ) -> Callable:
        """Wraps the given callable so it can be invoked as a Step or Task.

        Parameters
        ----------
        func: Callable
            Function to wrap.

        Returns:
        -------
        Callable
            Callable that represents the `Script` object `__call__` method when
            in a Steps or DAG context, otherwise returns the callable function
            unchanged.
        """
        # instance methods are wrapped in `staticmethod`. Hera can capture that
        # type and extract the underlying function for remote submission since
        # it does not depend on any class or instance attributes, so it is
        # submittable
        if isinstance(func, staticmethod):
            source: Callable = func.__func__
        else:
            source = func

        if "name" in script_kwargs:
            # take the client-provided `name` if it is submitted, pop the name
            # for otherwise there will be two kwargs called `name`
            name = script_kwargs.pop("name")
        else:
            # otherwise populate the `name` from the function name
            name = source.__name__.replace("_", "-")

        s = script_class(name=name, source=source, **script_kwargs)

        @wraps(func)
        def task_wrapper(*args, **kwargs) -> Union[FuncR, Step, Task, None]:
            """Invokes a `Script` object's `__call__` method using the given
            SubNode (Step or Task) args/kwargs."""
            if _context.active:
                return s.__call__(*args, **kwargs)
            return func(*args, **kwargs)

        # Set the wrapped function to the original function so that we can use
        # it later
        task_wrapper.wrapped_function = func  # type: ignore
        return task_wrapper

    return bettmensch_ai_script_wrapper
