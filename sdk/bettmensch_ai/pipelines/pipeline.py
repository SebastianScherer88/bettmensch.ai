import inspect
from typing import Any, Callable, Dict, List, Optional

from bettmensch_ai.client import client as pipeline_client
from bettmensch_ai.constants import COMPONENT_IMPLEMENTATION, PIPELINE_TYPE
from bettmensch_ai.io import InputParameter, Parameter
from bettmensch_ai.pipelines.pipeline_context import (
    PipelineContext,
    _pipeline_context,
)
from bettmensch_ai.utils import get_func_args, validate_func_args
from hera.auth import ArgoCLITokenGenerator
from hera.shared import global_config
from hera.workflows import DAG, Workflow, WorkflowsService, WorkflowTemplate
from hera.workflows.models import Workflow as WorkflowModel
from hera.workflows.models import (
    WorkflowTemplateDeleteResponse as WorkflowTemplateDeleteResponseModel,
)
from hera.workflows.models import (
    WorkflowTemplateRef as WorkflowTemplateRefModel,
)


class Pipeline(object):
    """Manages the PipelineContext meta data storage utility."""

    _config = global_config
    type: str = PIPELINE_TYPE
    name: str = None
    _namespace: str = None
    _built: bool = False
    _registered: bool = False
    clear_context: bool = None
    func: Callable = None
    inputs: Dict[str, InputParameter] = None
    user_built_workflow_template: WorkflowTemplate = None
    registered_workflow_template: WorkflowTemplate = None
    _client: WorkflowsService = pipeline_client

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
        self.inputs = self.build_pipeline_inputs_from_func(func)
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
        self.inputs = self.build_pipeline_inputs_from_wft()

        self._registered = True
        self.name = "-".join(self.registered_name.split("-")[1:-1])
        self._namespace = self.registered_namespace

    @property
    def io_owner_name(self) -> str:
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

        return self.registered_workflow_template.uid

    @property
    def registered_name(self):
        if not self.registered:
            return None

        return self.registered_workflow_template.name

    @property
    def registered_namespace(self):
        if not self.registered:
            return None

        return self.registered_workflow_template.namespace

    def build_pipeline_inputs_from_func(
        self, func: Callable
    ) -> Dict[str, InputParameter]:
        """Generates pipeline inputs from the underlying function. Also
        - checks for correct InputParameter type annotations in the decorated
            original function
        - ensures all original function inputs without default values are being
            specified

        Args:
            func (Callable): The function the we want to wrap in a Pipeline.

        Raises:
            Exception: Raised if the pipeline is not given an input for at
                least one of the underlying function's arguments without
                default value.

        Returns:
            Dict[str,InputParameter]: The pipeline's inputs.
        """

        validate_func_args(func, argument_types=[InputParameter])
        func_args = get_func_args(func)
        func_inputs = get_func_args(func, "annotation", [InputParameter])
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
            pipeline_input = InputParameter(name=name, value=default)

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

    def build_pipeline_inputs_from_wft(self) -> Dict[str, InputParameter]:
        """Generates pipeline inputs from the `registered_workflow_template`
        attribute instance. Used to populate the `inputs` attribute for
        pipelines that are built from the registry rather than the user
        provided function.

        Returns:
            Dict[str,InputParameter]: The pipeline's inputs.
        """

        result = {}

        wft_template_inputs: List[
            Parameter
        ] = self.registered_workflow_template.arguments.parameters

        for wft_input in wft_template_inputs:
            result[wft_input.name] = InputParameter(
                name=wft_input.name, value=wft_input.value
            )

        return result

    def build_workflow_template(self) -> WorkflowTemplate:
        """Builds the fully functional Argo WorkflowTemplate that implements
        the user specified Pipeline object.

        Returns:
            WorkflowTemplate: The (unsubmitted) Argo WorkflowTemplate object.
        """
        # add components to the global pipeline context
        with self.context:
            if self.clear_context:
                self.context.clear()

            self.func(**self.inputs)

        # build task factories
        for component in self.context.components:
            component.task_factory = component.build_hera_task_factory()

        # invoke all components' hera task generators from within a nested
        # WorkflowTemplate & DAG context
        with WorkflowTemplate(
            generate_name=f"pipeline-{self.name}-",
            entrypoint="bettmensch-ai-dag",
            namespace=self._namespace,
            arguments=[input.to_hera() for input in self.inputs.values()],
        ) as wft:

            # add non-script template
            for component in self.context.components:
                if component.implementation in (
                    COMPONENT_IMPLEMENTATION.torch.value,
                    COMPONENT_IMPLEMENTATION.lightning.value,
                ):
                    component.service_templates = (
                        component.build_service_templates()
                    )

            with DAG(name="bettmensch-ai-dag"):
                for component in self.context.components:
                    component.to_hera()

        return wft

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

        workflow_template = WorkflowTemplate.from_dict(
            registered_workflow_template.dict()
        )

        self.build_from_registry(workflow_template)

    def run(
        self,
        inputs: Dict[str, Any],
        wait: bool = False,
        poll_interval: int = 5,
    ) -> WorkflowModel:
        """Run a Flow using the registered Pipeline instance and user specified
        inputs.

        Raises:
            ValueError: Raised if the Pipeline instance hasnt been registered
                yet.

        Returns:
            WorkflowModel: The return of the Workflow.create class method call.
                Note that this is not the same as the hera.workflows.Workflow
                class.
        """

        # validate registration status of pipeline
        if not self.registered:
            raise ValueError(
                "Pipeline needs to be registered first. Are you sure you have"
                "ran `register`?"
            )

        # validate inputs
        non_default_args = dict([(p.name, p) for p in self.inputs.values()])
        missing_inputs = [k for k in non_default_args if k not in inputs]
        if missing_inputs:
            raise Exception(
                f"""The following non default inputs are missing: {
                    missing_inputs
                }. Pipeline inputs: {self.inputs}"""
            )
        unknown_inputs = [k for k in inputs if k not in non_default_args]
        if unknown_inputs:
            raise Exception(
                f"""The following inputs are not known for this pipeline: {
                    unknown_inputs
                }. Pipeline inputs: {self.inputs}"""
            )

        pipeline_ref = WorkflowTemplateRefModel(name=self.registered_name)

        workflow_inputs = [
            Parameter(name=k, value=v) for k, v in inputs.items()
        ]

        workflow: Workflow = Workflow(
            generate_name=f"{self.registered_name}-flow-",
            workflow_template_ref=pipeline_ref,
            namespace=self.registered_namespace,
            arguments=workflow_inputs,
        )

        registered_workflow: WorkflowModel = workflow.create(
            wait=wait, poll_interval=poll_interval
        )

        return registered_workflow

    @classmethod
    def from_workflow_template(
        cls, workflow_template: WorkflowTemplate
    ) -> "Pipeline":
        """Class method to initialize a Pipeline instance from a
        WorkflowTemplate instance.

        Args:
            workflow_template (WorkflowTemplate): The WorkflowTemplate
                pipeline instance retrieved from the bettmensch.ai server,
                .e.g using pipeline.get()

        Returns:
            Pipeline: The (registered) Pipeline instance.
        """

        return cls(registered_pipeline=workflow_template)

    @classmethod
    def from_registry(
        cls, registered_name: str, registered_namespace: str = "argo", **kwargs
    ) -> "Pipeline":
        """Class method to initialize a Pipeline instance from a
        registered_name and registered_namespace spec directly from the server.

        Args:
            registered_name (str): The name of the registered Pipeline to use.
            registered_namespace (str, optional): The namespace of the
                registered Pipeline to use.. Defaults to 'argo'.

        Returns:
            Pipeline: The (registered) Pipeline instance.
        """

        return get_registered_pipeline(
            registered_name=registered_name,
            registered_namespace=registered_namespace,
            **kwargs,
        )


def get_registered_pipeline(
    registered_name: str,
    registered_namespace: Optional[str] = None,
    **kwargs,
) -> Pipeline:
    """Returns the specified registered Pipeline.

    Args:
        registered_name (str): The `registered_name` of the Pipeline
            (equivalent to the `name` of its underlying WorkflowTemplate).
        registered_namespace (str): The `registered_namespace` of the Pipeline
            (equivalent to the `namespace` of its underlying WorkflowTemplate).
    Returns:
        Pipeline: The registered Pipeline instance.
    """

    registered_workflow_template = pipeline_client.get_workflow_template(
        namespace=registered_namespace, name=registered_name
    )

    workflow_template = WorkflowTemplate.from_dict(
        registered_workflow_template.dict()
    )

    pipeline = Pipeline.from_workflow_template(workflow_template)

    return pipeline


def list_registered_pipelines(
    registered_namespace: Optional[str] = None,
    registered_name_pattern: Optional[str] = None,
    label_selector: Optional[str] = None,
    field_selector: Optional[str] = None,
    **kwargs,
) -> List[Pipeline]:
    """Lists all registered pipelines.

    Returns:
        List[Pipeline]: A list of all registered Pipelines that meet the query
            scope.
    """

    response = pipeline_client.list_workflow_templates(
        namespace=registered_namespace,
        name_pattern=registered_name_pattern,
        label_selector=label_selector,
        field_selector=field_selector,
        **kwargs,
    )

    if response.items is not None:
        workflow_templates = [
            WorkflowTemplate.from_dict(registered_workflow_template.dict())
            for registered_workflow_template in response.items
        ]

        pipelines = [
            Pipeline.from_workflow_template(workflow_template)
            for workflow_template in workflow_templates
        ]
    else:
        pipelines = []

    return pipelines


def delete_registered_pipeline(
    registered_name: str, registered_namespace: Optional[str] = None, **kwargs
) -> WorkflowTemplateDeleteResponseModel:
    """Deletes the specified registered Pipeline from the server.

    Args:
        registered_name (str): The name of the registered Pipeline to delete.
        registered_namespace (Optional[str], optional): The namespace of the
            registered Pipeline to delete. Defaults to None.
    """

    delete_response = pipeline_client.delete_workflow_template(
        name=registered_name, namespace=registered_namespace, **kwargs
    )

    return delete_response


def pipeline(
    name: str, namespace: str, clear_context: bool
) -> Callable[[], Pipeline]:
    """Module's main decorator that takes a Calleable and returns a Pipeline
    instance with populated `user_built_workflow_template` attribute holding an
    WorkflowTemplate instance that implements the pipeline defined in the
    decorated Callable.

    Usage:

    ```python
    @component
    def add(
        a: InputParameter,
        b: InputParameter,
        sum: OutputParameter = None
    ) -> None:

        sum.assign(a + b)

    @pipeline('test_pipeline','argo',True)
    def a_plus_b_plus_c(a: InputParameter=1,
                        b: InputParameter=2,
                        c: InputParameter=3):

        a_plus_b = add(a = a, b = b)
        a_plus_b_plus_c = add(a = a_plus_b.outputs['sum'], b = c)
    ```
    """

    def pipeline_factory(func: Callable) -> Pipeline:

        return Pipeline(
            name=name,
            namespace=namespace,
            clear_context=clear_context,
            func=func,
        )

    return pipeline_factory
