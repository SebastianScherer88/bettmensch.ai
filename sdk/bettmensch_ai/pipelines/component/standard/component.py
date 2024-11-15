from typing import Any, Callable, Dict, List, Optional, Union

from bettmensch_ai.pipelines.component.base.component import BaseComponent
from bettmensch_ai.pipelines.component.utils import bettmensch_ai_script
from bettmensch_ai.pipelines.constants import (
    COMPONENT_IMAGE,
    COMPONENT_IMPLEMENTATION,
)
from hera.workflows import Task

from .script import BettmenschAIStandardScript


class Component(BaseComponent):

    implementation: str = COMPONENT_IMPLEMENTATION.standard.value
    default_image: str = COMPONENT_IMAGE.standard.value
    cpu: Optional[Union[float, int, str]] = "100m"
    memory: Optional[str] = "100Mi"

    def build_hera_task_factory(self) -> Callable:
        """Generates the task factory task_wrapper callable from the
        hera.workflows.script decorator definition. Needs to be called outide
        of an active hera context.

        Returns:
            Callable: A callable which, if called inside an active hera DAG
                context, generate the hera Task.
        """

        script_decorator_kwargs = super().build_script_decorator_kwargs()

        # this will invoke our custom ComponentInlineScriptRunner under the
        # hood
        script_wrapper = bettmensch_ai_script(
            script=BettmenschAIStandardScript, **script_decorator_kwargs
        )

        task_factory = script_wrapper(func=self.func)

        return task_factory

    def to_hera(self) -> Task:
        """Generates a hera.workflow.Task instance. Needs to be called from
            within an active hera context, specifically:
            - an outer layer hera.WorkflowTemplate context
            - an inner layer hera.DAG context.
        Otherwise the `task_factory` invocation won't return the
        hera.workflows.Task instance, and it wont be added to either
        hera.WorkflowTemplate or the hera.DAG.

        Returns:
            Task: A task that implements this Component instance in the hera
                library.
        """
        task_inputs = self.task_inputs.values()

        task = self.task_factory(
            arguments=[task_input.to_hera() for task_input in task_inputs],
            name=self.name,
            depends=self.depends,
        )

        return task


def as_component(
    func: Callable,
) -> Callable[[Callable, str, Dict, List[Any]], Component]:
    """Takes a calleable and generates a configured Component factory that will
    generate a Component version of the callable if invoked inside an active
    PipelineContext.

    Usage:

    ```python
    @bettmensch_ai.component #-> component factory
    def add(
        a: InputParameter = 1,
        b: InputParameter = 2,
        sum: OutputParameter = None
    ):
        sum.assign(a + b)
    ```

    Decorating the above `add` method returns a component factory that
    generates a Component class instance when called from within an active
    PipelineContext.
    """

    def component_factory(
        name: str = "",
        hera_template_kwargs: Dict = {},
        **component_inputs_kwargs,
    ) -> Component:

        return Component(
            func=func,
            name=name,
            hera_template_kwargs=hera_template_kwargs,
            **component_inputs_kwargs,
        )

    return component_factory
