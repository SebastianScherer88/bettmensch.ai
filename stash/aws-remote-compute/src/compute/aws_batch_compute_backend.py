"""`AwsBatchComputeBackend`: runs a task as an AWS Batch job."""

from __future__ import annotations

import json
import time
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

_TERMINAL_STATUSES = {"SUCCEEDED", "FAILED"}


@dataclass(frozen=True)
class AwsBatchConfig:
    """Configuration for placing a task on AWS Batch.

    Attributes:
        job_queue: The Batch job queue to submit to. Must already exist -
            `AwsBatchComputeBackend.run()` never provisions one, the same
            way `S3ArtifactStore` never creates its own bucket.
        job_definition: The job definition to run against (name, name:
            revision, or ARN). Required for ad-hoc `LocalRunner`-driven
            execution - left unset only when this config is used purely as
            a *registration-time* spec (see `image`/`job_role_arn`/
            `execution_role_arn` below), where `CompiledPipeline.register()`
            creates one and fills this in.
        image: The container image `CompiledPipeline.register()` should
            build a job definition from, if `job_definition` isn't already
            given. Unused by ad-hoc execution.
        job_role_arn: The IAM role the job definition's container runs as,
            for registration. Unused by ad-hoc execution.
        execution_role_arn: The IAM role Batch itself uses to pull the
            image/write logs, for registration. Unused by ad-hoc execution.
        environment: Environment variables baked into the job definition's
            container at registration time (typically the target
            `S3ArtifactStoreConfig`'s own `BETTMENSCH_AI_S3_ARTIFACT_STORE_*`
            fields) - a submitted job has no other way to know which
            artifact store to talk to, since `containerOverrides` only ever
            carries the per-invocation command, not environment variables.
            Unused by ad-hoc execution (the job definition it runs against
            already has its own environment, set when it was registered).
        default_vcpu / default_memory: Fargate-valid `VCPU`/`MEMORY`
            resource requirement values (Batch's own string format, e.g.
            `"0.25"`/`"512"`) used at registration time only for a task with
            no `@resource(...)` of its own - AWS Batch on Fargate rejects a
            job definition with no resource requirements at all, unlike
            EC2-backed Batch where they're optional. Unused by ad-hoc
            execution.
        poll_interval_seconds: How often `run()` polls the submitted job's
            status.
        timeout_seconds: How long `run()` waits for the job to finish
            before giving up, or `None` to wait indefinitely.
    """

    job_queue: str
    job_definition: Optional[str] = None
    image: Optional[str] = None
    job_role_arn: Optional[str] = None
    execution_role_arn: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    default_vcpu: str = "0.25"
    default_memory: str = "512"
    poll_interval_seconds: float = 5.0
    timeout_seconds: Optional[float] = None


class AwsBatchComputeBackend(BaseComputeBackend):
    """Runs a task as an AWS Batch job, blocking until it finishes.

    Never provisions the job queue or job definition it submits against -
    ad-hoc, `LocalRunner`-driven execution only ever runs against
    infrastructure that already exists (mirroring `S3ArtifactStore`'s own
    "never creates its own bucket" rule). Provisioning a job definition
    from `config`'s registration-only fields is `CompiledPipeline.
    register()`'s job, not this class's.
    """

    name = "aws_batch"
    requires_code_bundle = True

    def __init__(self, config: AwsBatchConfig):
        self.config = config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_queue": self.config.job_queue,
            "job_definition": self.config.job_definition,
        }

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
        if not self.config.job_definition:
            raise RemoteComputeConfigurationError(
                f"AwsBatchConfig for task {assembled_task.name!r} has no "
                "job_definition - ad-hoc execution needs a job definition "
                "that already exists; registering this pipeline is what "
                "creates one from image/job_role_arn/execution_role_arn."
            )

        func = assembled_task.func
        command = [
            "python",
            "-m",
            "bettmensch_ai.pipelines.runner.remote_entrypoint",
            "--task-module",
            func.__module__,
            "--task-qualname",
            func.__qualname__,
            "--default-materializer-module",
            default_materializer_cls.__module__,
            "--default-materializer-qualname",
            default_materializer_cls.__qualname__,
            "--code-bundle-key",
            code_bundle_key or "",
            "--static-inputs",
            json.dumps(assembled_task.static_inputs),
            "--input-keys",
            json.dumps(input_keys),
            "--output-keys",
            json.dumps(output_keys),
        ]

        client = boto3.client("batch")
        submitted = client.submit_job(
            jobName=f"{pipeline_name}-{assembled_task.name}-{pipeline_run_id}",
            jobQueue=self.config.job_queue,
            jobDefinition=self.config.job_definition,
            containerOverrides={"command": command},
        )
        job_id = submitted["jobId"]

        job = self._wait_for_completion(client, job_id)

        if job["status"] == "FAILED":
            logs = self._fetch_logs(job)
            raise RemoteTaskExecutionError(
                f"AWS Batch job {job_id!r} for task {assembled_task.name!r} "
                f"failed: {job.get('statusReason', 'no reason given')}."
                + (f"\n\n{logs}" if logs else "")
            )

        return self._fetch_logs(job)

    def _wait_for_completion(self, client: Any, job_id: str) -> Dict[str, Any]:
        """Polls `describe_jobs` until `job_id` reaches a terminal status.

        Args:
            client: The `boto3` Batch client.
            job_id: The submitted job's id.

        Returns:
            That job's final description.

        Raises:
            RemoteTaskExecutionError: If `timeout_seconds` elapses first.
        """

        waited = 0.0
        while True:
            job = client.describe_jobs(jobs=[job_id])["jobs"][0]
            if job["status"] in _TERMINAL_STATUSES:
                return job

            if (
                self.config.timeout_seconds is not None
                and waited >= self.config.timeout_seconds
            ):
                raise RemoteTaskExecutionError(
                    f"AWS Batch job {job_id!r} did not finish within "
                    f"{self.config.timeout_seconds}s."
                )

            time.sleep(self.config.poll_interval_seconds)
            waited += self.config.poll_interval_seconds

    def _fetch_logs(self, job: Dict[str, Any]) -> Optional[str]:
        """Best-effort fetch of a finished job's CloudWatch logs.

        Never raises: a logs-fetch failure shouldn't itself fail (or hide
        the real failure of) a task run - mirrors this codebase's other
        best-effort, never-fails-the-caller patterns (e.g. the frontend's
        own artifact-metadata reads).

        Args:
            job: A `describe_jobs` job description.

        Returns:
            The fetched log text, or `None` if there's no log stream to
            read or fetching it failed.
        """

        try:
            log_stream_name = job["container"]["logStreamName"]
            events = boto3.client("logs").get_log_events(
                logGroupName="/aws/batch/job", logStreamName=log_stream_name
            )["events"]
            return "\n".join(event["message"] for event in events) or None
        except Exception:  # noqa: BLE001 - logs are best-effort, never fatal
            return None
