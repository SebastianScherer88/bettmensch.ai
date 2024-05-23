import inspect
import os
from typing import Callable, Dict, List, Union

from argo_workflows.api import workflow_template_service_api
from bettmensch_ai.arguments import (
    ComponentInput,
    ComponentOutput,
    Parameter,
    PipelineInput,
)
from bettmensch_ai.client import client
from bettmensch_ai.component import (
    PipelineContext,
    _pipeline_context,
    component,
)
from bettmensch_ai.server import RegisteredFlow, RegisteredPipeline
from bettmensch_ai.utils import get_func_args, validate_func_args
from hera.auth import ArgoCLITokenGenerator
from hera.shared import global_config
from hera.workflows import DAG, Workflow, WorkflowTemplate
from hera.workflows.models import WorkflowTemplateRef


class Pipeline(object):
    """Manages the PipelineContext meta data storage utility."""

    _config = global_config
    type: str = "workflow"
    name: str = None
    _namespace: str = None
    _built: bool = False
    _registered: bool = False
    clear_context: bool = None
    func: Callable = None
    inputs: Dict[str, PipelineInput] = None
    user_built_workflow_template: WorkflowTemplate = None
    registered_workflow_template: WorkflowTemplate = None

    def __init__(
        self,
        name: str = None,
        namespace: str = None,
        func: Callable = None,
        clear_context: bool = True,
        registered_pipeline: WorkflowTemplate = None,
    ):
        """_summary_

        Args:
            name (str): Name of the local pipeline
            namespace (str): K8s namespace of the pipeline
            func (Callable): The decorated function defining the pipeline logic
            clear_context (bool, optional): Whether the global pipeline context
                should be cleared when building. Defaults to True.
        """
        self.configure()
        self.clear_context = clear_context

        if registered_pipeline is None:
            assert all([arg is not None for arg in (name, namespace, func)])
            self.build_from_user(name, namespace, func)
        else:
            self.build_from_registry(registered_pipeline)

    def configure(self):
        """Applies unsecure authentication configuration for an ArgoWorkflow
        server available on port 2746."""
        self._config.host = "https://localhost:2746"
        self._config.token = ArgoCLITokenGenerator
        self._config.verify_ssl = False

    def build_from_user(self, name: str, namespace: str, func: Callable):
        """Builds the pipeline definition and stores resulting
        hera.workflows.WorkflowTemplate instance in the workflow_template
        attribute

        Args:
            name (str): Name of the pipeline
            namespace (str): K8s namespace of the pipeline
            func (Callable): The decorated function defining the pipeline logic
        """

        self.name = name
        self._namespace = namespace
        self.func = func
        self.inputs = self.generate_inputs_from_func(func)
        self.user_built_workflow_template = self.build_workflow_template()
        self._built = True

    def build_from_registry(
        self, registered_workflow_template: WorkflowTemplate
    ):
        """Stores the argument to the register register_pipeline attribute and
        updates _registered

        Args:
            registered_pipeline (WorkflowTemplate): The WorkflowTemplate
                instance.
        """

        self.registered_workflow_template = registered_workflow_template
        self._registered = True

    @property
    def parameter_owner_name(self) -> str:
        return f"{self.type}"

    @property
    def context(self) -> PipelineContext:
        """Utility handle for global pipeline context instance.

        Returns:
            PipelineContext: The global pipeline context.
        """

        return _pipeline_context

    @property
    def built(self):
        return self._built

    @property
    def registered(self):
        return self._registered

    @property
    def registered_id(self):
        if not self.registered:
            return None

        return self.registered_workflow_template.metadata.uid

    @property
    def registered_name(self):
        if not self.registered:
            return None

        return self.registered_workflow_template.metadata.name

    @property
    def registered_namespace(self):
        if not self.registered:
            return self._name

        return self.registered_workflow_template.metadata.namespace

    def generate_inputs_from_func(
        self, func: Callable
    ) -> Dict[str, PipelineInput]:
        """Generates pipeline inputs from the underlying function. Also
        - checks for correct PipelineInput type annotations in the decorated
            original function
        - ensures all original function inputs without default values are being
            specified

        Args:
            func (Callable): The function the we want to wrap in a Component.

        Raises:
            Exception: Raised if the pipeline is not given an input for at
                least one of the underlying function's arguments without
                default value.

        Returns:
            Dict[str,PipelineInput]: The pipeline's inputs.
        """

        validate_func_args(func, argument_types=[PipelineInput])
        func_args = get_func_args(func)
        func_inputs = get_func_args(func, "annotation", [PipelineInput])
        non_default_args = get_func_args(func, "default", [inspect._empty])
        required_func_inputs = dict(
            [(k, v) for k, v in func_inputs.items() if k in non_default_args]
        )

        result = {}

        for name in func_inputs:

            # assemble component input
            default = (
                func_args[name].default
                if func_args[name].default != inspect._empty
                else None
            )
            pipeline_input = PipelineInput(name=name, value=default)

            pipeline_input.set_owner(self)

            # remove declared input from required inputs (if relevant)
            if name in required_func_inputs:
                del required_func_inputs[name]

            result[name] = pipeline_input

        # ensure no required inputs are left unspecified
        if required_func_inputs:
            raise Exception(
                f"Unspecified required input(s) left: {required_func_inputs}"
            )

        return result

    def build_workflow_template(self) -> WorkflowTemplate:
        # add components to the global pipeline context
        with self:
            self.func(**self.inputs)

        # invoke all components' hera task generators from within a nested
        # WorkflowTemplate & DAG context
        with WorkflowTemplate(
            generate_name=f"pipeline-{self.name}-",
            entrypoint="bettmensch-ai-dag",
            namespace=self._namespace,
            arguments=[
                input.to_hera_parameter() for input in self.inputs.values()
            ],
        ) as wft:

            with DAG(name="bettmensch-ai-dag"):
                for component in self.context.components:
                    component.to_hera_task()

        return wft

    def __enter__(self):
        _pipeline_context.activate()

        # clear the global pipeline context when entering the pipeline
        # instance's context, if specified
        if self.clear_context:
            _pipeline_context.clear()

        return self

    def __exit__(self, *args, **kwargs):
        _pipeline_context.deactivate()

    def export(self, dir: str = "."):
        """Writes workflow_template attribute as f"{self.name}.yaml"
        WorkflowTemplate CR manifest to specified file path.

        Args:
            dir (str): The directory to write to.
        """

        if self.built:
            self.user_built_workflow_template.to_file(dir)
        if self.registered:
            self.registered_workflow_template.to_file(dir)

    def register(self):
        """Register the Pipeline instance on the bettmensch.ai server.

        Raises:
            ValueError: Raised if the Pipeline instance has already been
                registered yet.
        """

        if self.registered:
            raise ValueError(
                f"Pipeline has already been registered with id {self.id}"
            )

        registered_workflow_template = (
            self.user_built_workflow_template.create()
        )

        self.build_from_registry(registered_workflow_template)

    def run(self, **pipeline_input_kwargs) -> Workflow:
        """Run a Flow using the registered Pipeline instance and user specified
        inputs.

        Raises:
            ValueError: Raised if the Pipeline instance hasnt been registered
                yet.

        Returns:
            Workflow: The return of the Workflow.create class method call.
        """
        if not self.registered:
            raise ValueError(
                f"Pipeline needs to be registered first. Are you sure you have"
                "ran `register`?"
            )

        pipeline_ref = WorkflowTemplateRef(name=self.registered_name)

        pipeline_inputs = [
            Parameter(name=k, value=v) for k, v in pipeline_input_kwargs.items()
        ]

        workflow = Workflow(
            generate_name=f"{self.registered_name}-flow-",
            workflow_template_ref=pipeline_ref,
            namespace=self.registered_namespace,
            arguments=pipeline_inputs,
        )

        registered_workflow = workflow.create()

        return registered_workflow

    @classmethod
    def from_registered_pipeline(
        cls, registered_pipeline: WorkflowTemplate
    ) -> "Pipeline":
        """Class method to initialize a Pipeline instance from a
        WorkflowTemplate instance.

        Args:
            registered_pipeline (WorkflowTemplate): The WorkflowTemplate
                pipeline instance retrieved from the bettmensch.ai server,
                .e.g using pipeline.get()

        Returns:
            Pipeline: The (registered) Pipeline instance.
        """

        return cls(registered_pipeline=registered_pipeline)

    # @classmethod
    # def from_registry(
    #     cls, registered_name: str, registered_namespace: str = "argo"
    # ) -> "Pipeline":
    #     """Class method to initialize a Pipeline instance from a
    #     registered_name and registered_namespace spec directly from the server.

    #     Args:
    #         registered_name (str): The name of the registered Pipeline to use.
    #         registered_namespace (str, optional): The namespace of the
    #             registered Pipeline to use.. Defaults to 'argo'.

    #     Returns:
    #         Pipeline: The (registered) Pipeline instance.
    #     """

    #     return get(
    #         registered_name=registered_name,
    #         registered_namespace=registered_namespace,
    #         as_workflow_template=False,
    #     )


# def get(
#     registered_name: str,
#     registered_namespace: str = "argo",
#     as_workflow_template: bool = False,
#     **kwargs,
# ) -> Union[WorkflowTemplate, RegisteredPipeline]:
#     """Pipeline query utility. Wrapper around hera's WorkflowTemplate `get`
#     query function that optionally converts to RegisteredPipeline.

#     Args:
#         registered_name (str): The `registered_name` of the Pipeline
#             (equivalent to the `name` of its underlying WorkflowTemplate).
#         registered_namespace (str): The `registered_namespace` of the Pipeline
#             (equivalent to the `namespace` of its underlying WorkflowTemplate).
#     Returns:
#         Union[None,WorkflowTemplate,RegisteredPipeline]: Returns None if no
#             pipeline with the specified uid could be found on the configured
#             server. Returns the matching WorkflowTemplate if possible and
#             as_workflow_template=False, otherwise converts the matching
#             WorkflowTemplate to a RegisteredPipeline before returning.
#     """

#     api_instance = workflow_template_service_api.WorkflowTemplateServiceApi(
#         client
#     )
#     wt: WorkflowTemplate = api_instance.get_workflow_template(
#         name=registered_name, namespace=registered_namespace, **kwargs
#     )

#     if not as_workflow_template:
#         rp = RegisteredPipeline.from_argo_workflow_cr(wt)
#         p = Pipeline.from_registered_pipeline(rp)

#         return p

#     return wt


# def list(
#     registered_namespace: str = "argo",
#     as_workflow_template: bool = False,
#     **kwargs,
# ) -> List[Union[WorkflowTemplate, RegisteredPipeline]]:
#     """Pipeline query utility. Wrapper around hera's WorkflowTemplate `list`
#     query function that optionally converts to RegisteredPipeline elements.

#     Returns:
#         Union[None,WorkflowTemplate,RegisteredPipeline]: Returns None if no
#             pipeline with the specified uid could be found on the configured
#             server. Returns the matching WorkflowTemplate if possible and
#             as_workflow_template=False, otherwise converts the matching
#             WorkflowTemplate to a RegisteredPipeline before returning.
#     """

#     api_instance = workflow_template_service_api.WorkflowTemplateServiceApi(
#         client
#     )
#     wt_list: List[WorkflowTemplate] = api_instance.list_workflow_templates(
#         namespace=registered_namespace, **kwargs
#     )

#     if not as_workflow_template:
#         rp_list = [
#             RegisteredPipeline.from_argo_workflow_cr(wt) for wt in wt_list
#         ]
#         p_list = [Pipeline.from_registered_pipeline(rp) for rp in rp_list]

#         return p_list

#     return wt_list


def pipeline(
    name: str, namespace: str, clear_context: bool
) -> Callable[[], Pipeline]:
    """Takes a calleable and returns a Pipeline instance with populated
    workflow_template attribute holding an hera.workflows.WorkflowTemplate
    instance that implements the pipeline defined in the decorated callable.

    Usage:
    @component
    def add(a: ComponentInput, b: ComponentInput, sum: ComponentOutput = None) -> None:

        sum.assign(a + b)

    @pipeline('test_pipeline','argo',True)
    def a_plus_b_plus_c(a: PipelineInput=1,
                        b: PipelineInput=2,
                        c: PipelineInput=3):

        a_plus_b = add(a = a, b = b)
        a_plus_b_plus_c = add(a = a_plus_b.outputs['sum'], b = c)
    """

    def pipeline_factory(func: Callable) -> Pipeline:

        return Pipeline(
            name=name,
            namespace=namespace,
            clear_context=clear_context,
            func=func,
        )

    return pipeline_factory


def test_pipeline():
    @component
    def add(
        a: ComponentInput, b: ComponentInput, sum: ComponentOutput = None
    ) -> None:

        sum.assign(a + b)

    @pipeline("test-pipeline", "argo", True)
    def a_plus_b_plus_c_times_2(
        a: PipelineInput = 1, b: PipelineInput = 2, c: PipelineInput = 3
    ):

        first_sum = add(
            hera_template_kwargs={
                "image": "bettmensch88/bettmensch.ai:3.11-50b2887"
            },
            a=a,
            b=b,
        )

        second_sum = add(
            hera_template_kwargs={
                "image": "bettmensch88/bettmensch.ai:3.11-50b2887"
            },
            a=first_sum.outputs["sum"],
            b=c,
        )

        last_sum = add(
            hera_template_kwargs={
                "image": "bettmensch88/bettmensch.ai:3.11-50b2887"
            },
            a=second_sum.outputs["sum"],
            b=second_sum.outputs["sum"],
        )

    print(f"Pipeline type: {type(a_plus_b_plus_c_times_2)}")

    a_plus_b_plus_c_times_2.export()
    a_plus_b_plus_c_times_2.register()
    a_plus_b_plus_c_times_2.run(a=11, b=22, c=33)


if __name__ == "__main__":
    test_pipeline()
