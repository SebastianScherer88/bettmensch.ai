"""`CompiledPipeline`: a Step Functions state machine definition compiled
from an `AssembledPipeline`, with the register()/deregister() CRUD that
creates and tears down the real AWS resources it needs.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from typing import Any, Dict

import boto3

from ...assembler.recording import serialize_assembled_pipeline
from ...compute.aws_batch_compute_backend import AwsBatchComputeBackend
from ...compute.aws_lambda_compute_backend import AwsLambdaComputeBackend
from ...metadata_store import BaseMetadataStore
from ...pipeline.assembled_pipeline import AssembledPipeline
from ...task.assembled_task import AssembledTask
from .asl import build_state_machine_definition

# Tags applied to every AWS resource `CompiledPipeline.register()` creates,
# so they're identifiable in the AWS console without cross-referencing
# anything back to this project's own metadata store.
_TAG_PIPELINE = "bettmensch-ai-pipeline"
_TAG_TASK = "bettmensch-ai-task"


def _batch_resource_requirements(resource_requirements, backend_config) -> list:
    """Converts a `ResourceRequirements` into AWS Batch's own
    `resourceRequirements` list shape (`[{"type", "value"}, ...]`).

    Fargate (this project's only supported Batch compute environment type -
    see `infrastructure/aws`) rejects a job definition with no `VCPU`/`MEMORY`
    requirement at all, unlike EC2-backed Batch where they're optional - so
    a task with no `@resource(...)` of its own falls back to
    `backend_config.default_vcpu`/`.default_memory` rather than omitting
    them. `gpu` is included only if declared (Fargate doesn't support GPU
    jobs at all; a GPU-requiring task registering onto a Fargate-only
    compute environment fails at Batch's own validation, not silently).

    Args:
        resource_requirements: The task's own `ResourceRequirements`.
        backend_config: Its `AwsBatchConfig`, for the Fargate-valid
            `default_vcpu`/`default_memory` fallback.
    """

    requirements = [
        {"type": "VCPU", "value": str(resource_requirements.cpu or backend_config.default_vcpu)},
        {
            "type": "MEMORY",
            "value": str(resource_requirements.memory or backend_config.default_memory),
        },
    ]
    if resource_requirements.gpu is not None:
        requirements.append({"type": "GPU", "value": str(resource_requirements.gpu)})

    return requirements


@dataclass
class CompiledPipeline:
    """A Step Functions state machine definition compiled from an
    `AssembledPipeline`, plus the pipeline it was compiled from (needed by
    `register()` to create per-task AWS resources from each task's own
    `AwsBatchConfig`/`AwsLambdaConfig`).

    `definition` as built by `StepFunctionsCompiler.compile()` may reference
    task-level resources (a Batch job definition, a Lambda function) that
    don't exist yet, if a task's config only carries registration-time
    build fields (`image`, `job_role_arn`, ...) rather than an existing
    `job_definition`/`function_name` - `register()` creates those first,
    then rebuilds `definition` from the now-complete configs before
    creating the state machine itself, so the two never actually go out of
    sync.
    """

    assembled_pipeline: AssembledPipeline
    definition: Dict[str, Any]

    def register(
        self, metadata_store: BaseMetadataStore, *, state_machine_role_arn: str
    ) -> uuid.UUID:
        """Creates every AWS resource this pipeline needs, then registers it.

        For each Batch-placed task with no existing `job_definition`,
        registers one under the deterministic name
        `f"{pipeline_name}-{task_name}"` from that task's `AwsBatchConfig`
        build fields, tagged with the pipeline and task name - Batch's own
        revisioning means re-registering the same pipeline just adds a new
        revision, not a conflict. Same pattern for a Lambda-placed task
        with no existing `function_name`. Then creates the state machine
        itself (tagged the same way) and records every created resource's
        ARN into `backend_metadata`.

        Args:
            metadata_store: The store to record this registration into.
            state_machine_role_arn: The IAM role Step Functions itself
                assumes to run this state machine (to call
                `batch:SubmitJob`/`lambda:InvokeFunction` on its own
                initiative, not this project's credentials) - a real,
                required field on AWS's own `create_state_machine` API,
                keyword-only and required here so it can never be silently
                omitted the way it was before this was caught.

        Returns:
            The new registration's id.
        """

        pipeline_name = self.assembled_pipeline.name
        batch_job_definitions: Dict[str, str] = {}
        lambda_functions: Dict[str, str] = {}

        for task in self.assembled_pipeline.tasks:
            backend = task.compute_backend
            if isinstance(backend, AwsBatchComputeBackend) and not backend.config.job_definition:
                batch_job_definitions[task.name] = self._register_batch_job_definition(
                    pipeline_name, task
                )
            elif isinstance(backend, AwsLambdaComputeBackend) and not backend.config.function_name:
                lambda_functions[task.name] = self._create_lambda_function(pipeline_name, task)

        # Rebuild now that every task's compute backend config is complete
        # (a task relying on registration to create its resource had an
        # incomplete config - and so a stale `definition` - until now).
        self.definition = build_state_machine_definition(self.assembled_pipeline)

        state_machine_arn = self._create_state_machine(pipeline_name, state_machine_role_arn)

        dag_structure, pipeline_inputs = serialize_assembled_pipeline(self.assembled_pipeline)
        backend_metadata = {
            "state_machine_arn": state_machine_arn,
            "batch_job_definitions": batch_job_definitions,
            "lambda_functions": lambda_functions,
        }

        return metadata_store.register_pipeline(
            pipeline_name,
            backend="aws_stepfunctions",
            dag_structure=dag_structure,
            pipeline_inputs=pipeline_inputs,
            backend_metadata=backend_metadata,
        )

    def deregister(
        self, metadata_store: BaseMetadataStore, pipeline_registration_id: uuid.UUID
    ) -> None:
        """Deletes every AWS resource `register()` created for this
        registration, then marks it retired.

        Deletes the state machine, every Batch job definition, and every
        Lambda function recorded in the registration's own
        `backend_metadata` - full cleanup, since the registration is what
        created (and so is what should tear down) all of them, not just
        the state machine.

        Args:
            metadata_store: The store the registration was recorded in.
            pipeline_registration_id: The registration to tear down.
        """

        registration = metadata_store.get_pipeline_registration(pipeline_registration_id)
        backend_metadata = registration.backend_metadata

        state_machine_arn = backend_metadata.get("state_machine_arn")
        if state_machine_arn:
            boto3.client("stepfunctions").delete_state_machine(
                stateMachineArn=state_machine_arn
            )

        batch_client = boto3.client("batch")
        for job_definition_arn in backend_metadata.get("batch_job_definitions", {}).values():
            batch_client.deregister_job_definition(jobDefinition=job_definition_arn)

        lambda_client = boto3.client("lambda")
        for function_arn in backend_metadata.get("lambda_functions", {}).values():
            lambda_client.delete_function(FunctionName=function_arn)

        metadata_store.deregister_pipeline(pipeline_registration_id)

    def _register_batch_job_definition(self, pipeline_name: str, task: AssembledTask) -> str:
        """Registers a Batch job definition for `task`, updating its
        `AwsBatchConfig` in place with the result.

        Always registers as a Fargate job definition
        (`platformCapabilities=["FARGATE"]`, a `networkConfiguration` with
        `assignPublicIp` enabled) - the only Batch compute environment type
        `infrastructure/aws` provisions, and Fargate requires both of these plus
        non-empty resource requirements (handled by
        `_batch_resource_requirements`) that EC2-backed Batch would treat as
        optional.

        Returns:
            The new job definition's ARN.
        """

        backend = task.compute_backend
        name = f"{pipeline_name}-{task.name}"

        response = boto3.client("batch").register_job_definition(
            jobDefinitionName=name,
            type="container",
            platformCapabilities=["FARGATE"],
            containerProperties={
                "image": backend.config.image,
                "jobRoleArn": backend.config.job_role_arn,
                "executionRoleArn": backend.config.execution_role_arn,
                "environment": [
                    {"name": key, "value": value}
                    for key, value in backend.config.environment.items()
                ],
                "resourceRequirements": _batch_resource_requirements(
                    task.resource_requirements, backend.config
                ),
                "networkConfiguration": {"assignPublicIp": "ENABLED"},
            },
            tags={_TAG_PIPELINE: pipeline_name, _TAG_TASK: task.name},
        )

        backend.config = replace(
            backend.config, job_definition=f"{name}:{response['revision']}"
        )

        return response["jobDefinitionArn"]

    def _create_lambda_function(self, pipeline_name: str, task: AssembledTask) -> str:
        """Creates a Lambda function for `task`, updating its
        `AwsLambdaConfig` in place with the result.

        Returns:
            The new function's ARN.
        """

        backend = task.compute_backend
        name = f"{pipeline_name}-{task.name}"

        response = boto3.client("lambda").create_function(
            FunctionName=name,
            Role=backend.config.role_arn,
            Code={"ImageUri": backend.config.image_uri},
            PackageType="Image",
            Timeout=backend.config.timeout_seconds,
            Environment={"Variables": dict(backend.config.environment)},
            Tags={_TAG_PIPELINE: pipeline_name, _TAG_TASK: task.name},
        )

        backend.config = replace(backend.config, function_name=name)

        return response["FunctionArn"]

    def _create_state_machine(self, pipeline_name: str, role_arn: str) -> str:
        """Creates the state machine itself from `self.definition`.

        Args:
            pipeline_name: The pipeline's name.
            role_arn: The IAM role Step Functions assumes to run it.

        Returns:
            The new state machine's ARN.
        """

        response = boto3.client("stepfunctions").create_state_machine(
            name=pipeline_name,
            definition=json.dumps(self.definition),
            roleArn=role_arn,
            tags=[{"key": _TAG_PIPELINE, "value": pipeline_name}],
        )

        return response["stateMachineArn"]
