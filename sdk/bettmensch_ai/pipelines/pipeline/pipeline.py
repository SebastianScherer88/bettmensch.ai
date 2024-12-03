import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from bettmensch_ai.pipelines.constants import (
    ARGO_NAMESPACE,
    COMPONENT_IMPLEMENTATION,
    FLOW_LABEL,
    PipelineDagTemplate,
    ResourceType,
)
from bettmensch_ai.pipelines.io import (
    InputParameter,
    OutputArtifact,
    OutputParameter,
    Parameter,
)
from bettmensch_ai.pipelines.pipeline_context import (
    PipelineContext,
    _pipeline_context,
)
from bettmensch_ai.pipelines.utils import (
    build_container_ios,
    get_func_args,
    validate_func_args,
)
from hera.auth import ArgoCLITokenGenerator
from hera.shared import global_config
from hera.workflows import DAG, Workflow, WorkflowsService, WorkflowTemplate
from hera.workflows.models import Workflow as WorkflowModel
from hera.workflows.models import WorkflowTemplate as WorkflowTemplateModel
from hera.workflows.models import (
    WorkflowTemplateDeleteResponse as WorkflowTemplateDeleteResponseModel,
)
from hera.workflows.models import (
    WorkflowTemplateRef as WorkflowTemplateRefModel,
)

from .client import ArgoWorkflowsBackendConfiguration, hera_client
from .flow import Flow, list_flows


class Pipeline(object):
    """Manages the PipelineContext meta data storage utility."""

    _config = global_config
    type: str = ResourceType.pipeline.value
    name: str = None
    _namespace: str = None
    _built: bool = False
    _registered: bool = False
    clear_context: bool = None
    func: Callable = None
    inputs: Dict[str, InputParameter] = None
    required_inputs: Dict[str, InputParameter] = None
    outputs: Dict[str, Union[OutputArtifact, OutputParameter]] = None
    task_inputs: Dict[str, InputParameter] = None
    user_built_workflow_template: WorkflowTemplate = None
    registered_workflow_template: WorkflowTemplate = None
    _client: WorkflowsService = hera_client

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

        configuration = ArgoWorkflowsBackendConfiguration()

        self._config.host = configuration.host
        self._config.token = ArgoCLITokenGenerator
        self._config.verify_ssl = configuration.verify_ssl

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
        (
            self.inputs,
            self.required_inputs,
        ) = self.build_inputs_from_func(func)
        self.outputs = self.build_outputs_from_func(func)
        self.task_inputs = self.build_task_inputs()
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
        (
            self.inputs,
            self.required_inputs,
        ) = self.build_inputs_from_wft()
        self.outputs = self.build_outputs_from_wft()
        self.task_inputs = self.build_task_inputs()

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
    def built(self) -> bool:
        return self._built

    @property
    def registered(self) -> bool:
        return self._registered

    @property
    def registered_id(self) -> Union[None, str]:
        if not self.registered:
            return None

        return self.registered_workflow_template.uid

    @property
    def registered_name(self) -> Union[None, str]:
        if not self.registered:
            return None

        return self.registered_workflow_template.name

    @property
    def registered_namespace(self) -> Union[None, str]:
        if not self.registered:
            return None

        return self.registered_workflow_template.namespace

    def build_inputs_from_func(
        self, func: Callable
    ) -> Tuple[Dict[str, InputParameter], Dict[str, InputParameter]]:
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
            Dict[str,InputParameter]: The pipeline's inputs and the pipeline's
                required inputs.
        """

        validate_func_args(
            func,
            argument_types=(InputParameter, OutputArtifact, OutputParameter),
        )
        func_args = get_func_args(func, "annotation", (InputParameter,))
        required_func_inputs = get_func_args(
            func, "default", (inspect._empty,)
        )

        inner_dag_template_inputs = {}
        required_inner_dag_template_inputs = {}

        for func_arg_name in func_args:

            if func_arg_name in required_func_inputs:
                workflow_template_input = InputParameter(
                    name=func_arg_name
                ).set_owner(self)
                required_inner_dag_template_inputs[
                    func_arg_name
                ] = workflow_template_input
            else:
                workflow_template_input = InputParameter(
                    name=func_arg_name, value=func_args[func_arg_name].default
                ).set_owner(self)

            inner_dag_template_inputs[func_arg_name] = workflow_template_input

        return inner_dag_template_inputs, required_inner_dag_template_inputs

    def build_outputs_from_func(
        self, func: Callable
    ) -> Dict[str, Union[OutputParameter, OutputArtifact]]:
        """Generate a dictionary with all the pipeline's outputs. To be used
        when defining the template of the inner DAG.

        Args:
            func (Callable): _description_

        Returns:
            Dict[str,Union[OutputParameter,OutputArtifact]]: _description_
        """

        return build_container_ios(
            self, func, annotation_types=(OutputArtifact, OutputParameter)
        )

    def build_task_inputs(self) -> Dict[str, InputParameter]:

        dag_inputs = {}
        for inner_dag_template_input in self.inputs.values():
            dag_task_input = (
                InputParameter(name=inner_dag_template_input.name)
                .set_source(inner_dag_template_input)
                .set_owner(self)
            )
            dag_inputs[dag_task_input.name] = dag_task_input

        return dag_inputs

    def build_inputs_from_wft(
        self,
    ) -> Tuple[Dict[str, InputParameter], Dict[str, InputParameter]]:
        """Generates pipeline inputs from the `registered_workflow_template`
        attribute instance. Used to populate the `inputs` attribute for
        pipelines that are built from the registry rather than the user
        provided function.

        Returns:
            Dict[str,InputParameter]: The pipeline's inputs.
        """

        inner_dag_template_inputs = {}
        required_inner_dag_template_inputs = {}

        inner_dag_template = [
            template
            for template in self.registered_workflow_template.templates
            if template.name == PipelineDagTemplate.inner.value
        ][0]
        wft_template_inputs = inner_dag_template.inputs.parameters

        for wft_input in wft_template_inputs:

            if wft_input.value is None:
                inner_dag_template_input = InputParameter(
                    name=wft_input.name
                ).set_owner(self)
                required_inner_dag_template_inputs[
                    wft_input.name
                ] = inner_dag_template_input
            else:
                try:
                    input_value = float(wft_input.value)
                except (ValueError, TypeError):
                    input_value = str(wft_input.value)
                inner_dag_template_input = InputParameter(
                    name=wft_input.name, value=input_value
                ).set_owner(self)

            inner_dag_template_inputs[
                wft_input.name
            ] = inner_dag_template_input

        return inner_dag_template_inputs, required_inner_dag_template_inputs

    def build_outputs_from_wft(
        self,
    ) -> Dict[str, Union[OutputParameter, OutputArtifact]]:

        inner_dag_template_outputs = {}

        wft_template_output_parameters = (
            self.registered_workflow_template.templates[0].outputs.parameters
        )

        wft_template_output_artifacts = (
            self.registered_workflow_template.templates[0].outputs.artifacts
        )

        if wft_template_output_parameters is not None:
            for wft_output_parameter in wft_template_output_parameters:
                inner_dag_template_outputs[
                    wft_output_parameter.name
                ] = OutputParameter(name=wft_output_parameter.name,).set_owner(
                    self
                )

        if wft_template_output_artifacts is not None:
            for wft_output_artifact in wft_template_output_artifacts:
                inner_dag_template_outputs[
                    wft_output_artifact.name
                ] = OutputArtifact(name=wft_output_artifact.name,).set_owner(
                    self
                )

        return inner_dag_template_outputs

    def build_inner_dag(self) -> DAG:

        # add non-script template
        for component in self.context.components:
            if (
                component.implementation
                == COMPONENT_IMPLEMENTATION.torch_ddp.value
            ):
                component.service_templates = (
                    component.build_service_templates()
                )

        with DAG(
            name=PipelineDagTemplate.inner.value,
            inputs=[dag_input.to_hera() for dag_input in self.inputs.values()],
            outputs=[
                dag_output.to_hera() for dag_output in self.context.outputs
            ],
        ) as inner_dag:
            for component in self.context.components:
                component.to_hera()

        return inner_dag

    def build_workflow_template(self) -> WorkflowTemplate:
        """Builds the fully functional Argo WorkflowTemplate that implements
        the user specified Pipeline object.

        Returns:
            WorkflowTemplate: The (unsubmitted) Argo WorkflowTemplate object.
        """
        # add components to the global pipeline context
        with self.context:
            print(f"Pipeline pipeline context: {self.context}")
            if self.clear_context:
                self.context.clear()

            self.func(
                **{
                    **self.inputs,
                    **self.outputs,
                }
            )

        # build task factories
        for component in self.context.components:
            component.task_factory = component.build_hera_task_factory()

        # invoke all components' hera task generators from within a nested
        # WorkflowTemplate & DAG context
        with WorkflowTemplate(
            generate_name=f"pipeline-{self.name}-",
            entrypoint=PipelineDagTemplate.outer.value,
            namespace=self._namespace,
            arguments=[
                workflow_template_input.to_hera()
                for workflow_template_input in self.inputs.values()  # noqa: E501
            ],
        ) as wft:

            inner_dag = self.build_inner_dag()

            with DAG(name=PipelineDagTemplate.outer.value):
                inner_dag(
                    arguments=[
                        dag_input.to_hera()
                        for dag_input in self.task_inputs.values()
                    ]
                )

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
    ) -> Flow:
        """Run a Flow using the registered Pipeline instance and user specified
        inputs.

        Raises:
            ValueError: Raised if the Pipeline instance hasnt been registered
                yet.

        Returns:
            Flow: The Flow obtained from the argo Workflow that was created by
                running the pipeline.
        """

        # validate registration status of pipeline
        if not self.registered:
            raise ValueError(
                "Pipeline needs to be registered first. Are you sure you have"
                "ran `register`?"
            )

        # validate inputs
        missing_inputs = [
            v for k, v in self.required_inputs.items() if k not in inputs
        ]
        if missing_inputs:
            raise Exception(
                f"The following required inputs are missing: {missing_inputs}"
                f" Pipeline inputs: {self.inputs}. Required"
                f" pipeline inputs: {self.required_inputs}."
            )
        unknown_inputs = [k for k in inputs if k not in self.inputs]
        if unknown_inputs:
            raise Exception(
                "The following inputs are not known for this pipeline:"
                f" {unknown_inputs}. Known pipeline inputs:"
                f" {self.inputs}"
            )

        # create workflow from workflow template
        pipeline_ref = WorkflowTemplateRefModel(name=self.registered_name)

        workflow_inputs = [
            Parameter(name=k, value=v) for k, v in inputs.items()
        ]

        workflow: Workflow = Workflow(
            generate_name=f"{self.registered_name}-flow-",
            workflow_template_ref=pipeline_ref,
            namespace=self.registered_namespace,
            arguments=workflow_inputs,
            labels={
                FLOW_LABEL.pipeline_name.value: self.registered_name,
                FLOW_LABEL.pipeline_id.value: self.registered_id,
            },
        )

        # run pipeline by submitting workflow to argo
        registered_workflow: WorkflowModel = workflow.create(
            wait=wait, poll_interval=poll_interval
        )

        return Flow.from_workflow_model(registered_workflow)

    @classmethod
    def from_workflow_template(
        cls, workflow_template: WorkflowTemplate
    ) -> "Pipeline":
        """Class method to initialize a Pipeline instance from a
        WorkflowTemplate instance.

        Args:
            workflow_template (WorkflowTemplate): An instance of hera's
                WorkflowTemplate class

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

    def list_flows(
        self,
        phase: Optional[str] = None,
        additional_labels: Dict = {},
        **kwargs,
    ) -> List[Flow]:
        """Lists all Flows that originate from this Pipeline.

        Args:
            phase (Optional[str], optional): Optional filter to only consider
                Flows that are in the specified phase. Defaults to None, i.e.
                no phase-based filtering. This will be added to the labels.
            additional_labels (Dict, optional): Optional filter to only
                consider Flows whose underlying argo Workflow resource contains
                all of the specified labels. Defaults to {}, i.e. no
                label-based filtering, however the pipeline's name and id will
                always be added automatically.

        Returns:
            List[Flow]: A list of Flows that meet the filtering specifications.
        """

        # validate registration status of pipeline
        if not self.registered:
            raise ValueError(
                "Pipeline needs to be registered first. Are you sure you have"
                "ran `register`?"
            )

        return list_flows(
            registered_namespace=self.registered_namespace,
            registered_pipeline_name=self.registered_name,
            phase=phase,
            labels=additional_labels,
            **kwargs,
        )


def get_registered_pipeline(
    registered_name: str,
    registered_namespace: Optional[str] = ARGO_NAMESPACE,
    **kwargs,
) -> Pipeline:
    """Get the registered pipeline.

    Args:
        registered_name (str): The `registered_name` of the Pipeline
            (i.e. the name its underlying WorkflowTemplate).
        registered_namespace (Optional[str], optional): The
            `registered_namespace` of the Pipeline (i.e. the namespace of its
            underlying WorkflowTemplate). Defaults to ARGO_NAMESPACE.

    Returns:
        Pipeline: A Pipeline object.
    """

    workflow_template_model: WorkflowTemplateModel = (
        hera_client.get_workflow_template(
            namespace=registered_namespace, name=registered_name, **kwargs
        )
    )

    workflow_template: WorkflowTemplate = WorkflowTemplate.from_dict(
        workflow_template_model.dict()
    )

    pipeline: Pipeline = Pipeline.from_workflow_template(workflow_template)

    return pipeline


def list_registered_pipelines(
    registered_namespace: str = ARGO_NAMESPACE,
    registered_name_pattern: Optional[str] = None,
    labels: Dict = {},
    **kwargs,
) -> List[Pipeline]:
    """Get all registered pipelines that meet the query specification.

    Args:
        registered_namespace (Optional[str], optional): The namespace in which
            the underlying argo WorkflowTemplate lives. Defaults to
            ARGO_NAMESPACE.
        registered_name_pattern (Optional[str], optional): The pattern to
            filter the argo WorkflowTemplates' names against. Defaults to None,
            i.e. no name-based filtering.
        labels (Dict, optional): Optional filter to only consider Pipelines
            whose underlying argo WorkflowTemplate resource contains all of the
            specified labels. Defaults to {}, i.e. no label-based filtering.

    Returns:
        List[Pipeline]: A list of all registered Pipelines that meet the query
            scope.
    """

    # build label selector
    if not labels:
        label_selector = None
    else:
        kv_label_list = list(labels.items())  # [('a',1),('b',2)]
        label_selector = ",".join(
            [f"{k}={v}" for k, v in kv_label_list]
        )  # "a=1,b=2"

    response = hera_client.list_workflow_templates(
        namespace=registered_namespace,
        name_pattern=registered_name_pattern,
        label_selector=label_selector,
        **kwargs,
    )

    if response.items is not None:
        workflow_templates: List[WorkflowTemplate] = [
            WorkflowTemplate.from_dict(workflow_template_model.dict())
            for workflow_template_model in response.items
        ]

        pipelines: List[Pipeline] = [
            Pipeline.from_workflow_template(workflow_template)
            for workflow_template in workflow_templates
        ]
    else:
        pipelines = []

    return pipelines


def delete_registered_pipeline(
    registered_name: str,
    registered_namespace: Optional[str] = ARGO_NAMESPACE,
    **kwargs,
) -> WorkflowTemplateDeleteResponseModel:
    """Deletes the specified registered Pipeline from the server.

    Args:
        registered_name (str): The name of the registered Pipeline to delete.
        registered_namespace (Optional[str], optional): The namespace of the
            registered Pipeline to delete. Defaults to None.
    """

    delete_response = hera_client.delete_workflow_template(
        name=registered_name, namespace=registered_namespace, **kwargs
    )

    return delete_response


def as_pipeline(
    name: str, namespace: str, clear_context: bool
) -> Callable[[Callable], Pipeline]:
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
