from typing import Callable, Dict

from bettmensch_ai.base_component import BaseComponent
from bettmensch_ai.base_inline_script_runner import (
    BaseComponentInlineScriptRunner,
)
from bettmensch_ai.utils import (
    COMPONENT_BASE_IMAGE,
    BettmenschAIScript,
    bettmensch_ai_script,
)
from hera.shared import global_config
from hera.workflows import Task
from hera.workflows.models import ImagePullPolicy


class ComponentInlineScriptRunner(BaseComponentInlineScriptRunner):

    """
    A customised version of the InlineScriptConstructor that implements a
    modified `_get_param_script_portion` and `generate_source` methods to
    ensure proper handling of the SDK's I/O objects at runtime.
    """

    pass


global_config.set_class_defaults(
    BettmenschAIScript, constructor=ComponentInlineScriptRunner()
)


class Component(BaseComponent):
    def build_hera_task_factory(self) -> Callable:
        """Generates the task factory task_wrapper callable from the
        hera.workflows.script decorator definition. Needs to be called outide
        of an active hera context.

        Returns:
            Callable: A callable which, if called inside an active hera DAG
                context, generate the hera Task.
        """

        script_decorator_kwargs = self.hera_template_kwargs.copy()
        script_decorator_kwargs.update(
            {
                "inputs": [
                    template_input.to_hera(template=True)
                    for template_input in self.template_inputs.values()
                ],
                "outputs": [
                    template_output.to_hera()
                    for template_output in self.template_outputs.values()
                ],
                "name": self.base_name,
            }
        )

        if "image" not in script_decorator_kwargs:
            script_decorator_kwargs["image"] = COMPONENT_BASE_IMAGE

        if "image_pull_policy" not in script_decorator_kwargs:
            script_decorator_kwargs[
                "image_pull_policy"
            ] = ImagePullPolicy.always

        if "resources" not in script_decorator_kwargs:
            script_decorator_kwargs["resources"] = self.build_resources()

        if "tolerations" not in script_decorator_kwargs:
            script_decorator_kwargs["tolerations"] = self.build_tolerations()

        # this will invoke our custom ComponentInlineScriptRunner under the
        # hood
        script_wrapper = bettmensch_ai_script(**script_decorator_kwargs)

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


def component(func: Callable) -> Callable[..., Component]:
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
