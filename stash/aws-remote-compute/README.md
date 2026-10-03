# Stash: AWS Batch / Lambda / Step Functions remote compute

Everything here was working, tested code and infrastructure - moved out of
the active tree on explicit request, not deleted, so it doesn't have to be
kept compatible while the artifact/metadata `Client`/`Store` abstractions
get redesigned. Bring it back once that design has landed.

## Why this was stashed

The redesign introduces a `Client` facade (and, for metadata, a whole new
always-on metadata service) sitting in front of `BaseArtifactStore`/
`BaseMetadataStore`. Every piece below constructs or depends on those
stores directly - `remote_entrypoint.py` builds its own `S3ArtifactStore`,
`CompiledPipeline.register()` calls `metadata_store.register_pipeline(...)`
directly, etc. Keeping all of that compatible with each step of the
Client/Store redesign would have meant updating three layers (Batch,
Lambda, Step Functions) in lockstep with an abstraction that wasn't settled
yet. Stashing it removes that drag; getting the basic design right matters
more right now than keeping remote compute running.

## What moved, and from where

```
src/
  compute/
    aws_batch_compute_backend.py    <- src/bettmensch_ai/pipelines/compute/
    aws_lambda_compute_backend.py   <- src/bettmensch_ai/pipelines/compute/
    call_site.py                    <- src/bettmensch_ai/pipelines/compute/   (aws_batch()/aws_lambda())
  compilers/                        <- src/bettmensch_ai/pipelines/compilers/ (whole package: StepFunctionsCompiler, CompiledPipeline, asl.py)
  runner/
    remote_entrypoint.py            <- src/bettmensch_ai/pipelines/runner/
    remote_runner.py                <- src/bettmensch_ai/pipelines/runner/   (RemoteRunner)
    registered_pipeline.py          <- src/bettmensch_ai/pipelines/runner/   (RegisteredPipeline)
    exceptions_additions.py           three ExecutionError subclasses trimmed from runner/exceptions.py:
                                       RemoteComputeConfigurationError, RemoteTaskExecutionError,
                                       UnsupportedBackendError

tests/                              <- tests/{unit,integration,functional}/pipelines/
  test_compute_backends.py
  test_call_site.py
  test_stepfunctions_compiler.py
  test_stepfunctions_compiler_integration.py
  test_remote_entrypoint.py
  test_registered_pipeline.py
  test_remote_runner.py
  test_aws_remote_compute_functional.py
  test_aws_stepfunctions_functional.py
  conftest_aws_fixture.py           extracted from tests/conftest.py: AwsStackConfig,
                                     _AWS_TEST_ENV_VARS, the aws_stack_config fixture

docker/
  Dockerfile.batch                  <- docker/task-runtime/
  Dockerfile.lambda                 <- docker/task-runtime/

infra/
  batch.py                          <- infrastructure/aws/  (Batch Fargate compute environment + job queue)
  test_fixtures.py                  <- infrastructure/aws/  (pre-created Batch job def + Lambda function for ad-hoc testing)
  iam_remote_compute.py             trimmed from infrastructure/aws/iam.py: the Batch job/execution,
                                     Lambda, and Step Functions IAM roles (self-contained - carries its
                                     own copies of the trust policies and _s3_read_write_policy)
  registry_remote_compute.py        trimmed from infrastructure/aws/registry.py: the two task-runtime
                                     ECR repos (Batch, Lambda)
  aws_makefile_snippet.mk           trimmed from infrastructure/aws/aws.makefile: task-runtime.push.batch,
                                     task-runtime.push.lambda, pipelines.test.aws
```

`src/bettmensch_ai/pipelines/compute/{base_compute_backend.py,
local_compute_backend.py}` and `runner/task_execution.py` stayed active -
they're generic (every task, local or remote, runs through
`execute_task`/`BaseComputeBackend`), not AWS-specific.

## Bringing it back

1. Move each file back to the path shown above (reverse the arrows).
2. Re-add the trimmed pieces to the files that shed them:
   - `src/bettmensch_ai/pipelines/compute/__init__.py`, the top-level
     `pipelines/__init__.py`, and `runner/__init__.py` - re-add the
     `Aws*`/`aws_batch`/`aws_lambda`/`RegisteredPipeline`/`RemoteRunner`
     imports and `__all__` entries (see this session's git history before
     the stash commit for the exact diff).
   - `runner/exceptions.py` - re-add the three classes from
     `exceptions_additions.py`.
   - `tests/conftest.py` - re-add `conftest_aws_fixture.py`'s contents.
   - `infrastructure/aws/iam.py` - re-add `iam_remote_compute.py`'s
     resources and merge its `Iam` dataclass fields back in.
   - `infrastructure/aws/registry.py` - re-add
     `registry_remote_compute.py`'s two repos and `Registry` fields.
   - `infrastructure/aws/__main__.py` - re-add `create_batch`/
     `create_test_fixtures` calls and their `pulumi.export(...)` outputs.
   - `infrastructure/aws/aws.makefile` - re-add `aws_makefile_snippet.mk`'s
     targets.
   - `pytest.ini` - re-add the `aws` marker.
3. Before trusting any of it again: this was real-AWS-verified once (an
   actual `pulumi up`, actual Batch/Lambda/Step Functions execution - see
   `docs/design-decisions.md`'s entries on the state_machine_role_arn/
   environment/Fargate-compatibility fixes, all discovered that way) - but
   the AWS account used was reset at least once during development
   (apparently a shared, auto-resetting sandbox). Re-verify end to end
   rather than assuming it still works untouched.
4. Rewire whatever constructed `S3ArtifactStore`/`PostgresMetadataStore`
   directly in here (`remote_entrypoint.py`, `CompiledPipeline`,
   `RemoteRunner`) to go through the new `Client` abstractions instead.
