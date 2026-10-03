"""`AwsLambdaComputeBackend`: runs a task as an AWS Lambda invocation."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

import boto3

from ..runner.exceptions import RemoteComputeConfigurationError, RemoteTaskExecutionError
from .base_compute_backend import BaseComputeBackend

if TYPE_CHECKING:
    from ..artifact_store import BaseArtifactStore
    from ..materializers.base_materializer import BaseMaterializer
    from ..task.assembled_task import AssembledTask


@dataclass(frozen=True)
class AwsLambdaConfig:
    """Configuration for placing a task on AWS Lambda.

    Attributes:
        function_name: The Lambda function to invoke (name or ARN). Must
            already exist - `AwsLambdaComputeBackend.run()` never
            provisions one, the same way `S3ArtifactStore` never creates
            its own bucket. Required for ad-hoc `LocalRunner`-driven
            execution; left unset only when this config is used purely as
            a *registration-time* spec (see `image_uri`/`role_arn` below),
            where `CompiledPipeline.register()` creates the function and
            fills this in.
        image_uri: The container image `CompiledPipeline.register()`
            should create the function from, if `function_name` isn't
            already given. Unused by ad-hoc execution.
        role_arn: The IAM role the function executes as, for registration.
            Unused by ad-hoc execution.
        environment: Environment variables baked into the function at
            creation time (typically the target `S3ArtifactStoreConfig`'s
            own `BETTMENSCH_AI_S3_ARTIFACT_STORE_*` fields) - a Lambda
            function's environment is static, configured on the function
            itself, not something an individual invocation payload can
            override (see the class docstring below). Unused by ad-hoc
            execution (the function it invokes already has its own
            environment, set when it was created).
        timeout_seconds: The invocation's own timeout, passed straight to
            `boto3`'s `invoke` call.
    """

    function_name: Optional[str] = None
    image_uri: Optional[str] = None
    role_arn: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 900


class AwsLambdaComputeBackend(BaseComputeBackend):
    """Runs a task as a synchronous AWS Lambda invocation.

    Never provisions the function it invokes - ad-hoc, `LocalRunner`-driven
    execution only ever runs against infrastructure that already exists
    (mirroring `S3ArtifactStore`'s own "never creates its own bucket"
    rule). Provisioning a function from `config`'s registration-only
    fields is `CompiledPipeline.register()`'s job, not this class's. The
    function's own deployment is expected to already carry whatever
    `BETTMENSCH_AI_S3_ARTIFACT_STORE_*` environment it needs - Lambda's own
    environment variables are static, configured on the function itself,
    not something an individual invocation can override; only per-
    invocation, dynamic data (which task, which keys) travels in the
    invoke payload.
    """

    name = "aws_lambda"
    requires_code_bundle = True

    def __init__(self, config: AwsLambdaConfig):
        self.config = config

    def to_dict(self) -> Dict[str, Any]:
        return {"function_name": self.config.function_name}

    def run(
        self,
        assembled_task: "AssembledTask",
        artifact_store: "BaseArtifactStore",
        code_bundle_key: Optional[str],
        input_keys: Dict[str, str],
        output_keys: Dict[str, str],
        pipeline_name: str,
        pipeline_run_id: uuid.UUID,
        default_materializer_cls: Type["BaseMaterializer"],
    ) -> Optional[str]:
        if not self.config.function_name:
            raise RemoteComputeConfigurationError(
                f"AwsLambdaConfig for task {assembled_task.name!r} has no "
                "function_name - ad-hoc execution needs a function that "
                "already exists; registering this pipeline is what "
                "creates one from image_uri/role_arn."
            )

        func = assembled_task.func
        payload = {
            "task_module": func.__module__,
            "task_qualname": func.__qualname__,
            "default_materializer_module": default_materializer_cls.__module__,
            "default_materializer_qualname": default_materializer_cls.__qualname__,
            "code_bundle_key": code_bundle_key,
            "static_inputs": assembled_task.static_inputs,
            "input_keys": input_keys,
            "output_keys": output_keys,
        }

        response = boto3.client("lambda").invoke(
            FunctionName=self.config.function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload).encode("utf-8"),
        )

        result = json.loads(response["Payload"].read())

        if response.get("FunctionError") or result.get("status") == "failed":
            raise RemoteTaskExecutionError(
                f"AWS Lambda invocation of {self.config.function_name!r} "
                f"for task {assembled_task.name!r} failed: "
                f"{result.get('traceback') or result.get('logs') or result}"
            )

        return result.get("logs")
