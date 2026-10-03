"""Layer 1: local, backend-agnostic pipeline assembly.

See docs/architecture.md for the full picture. In short: decorate plain
functions with `@task`/`@pipeline` to assemble a DAG (`Task`/`Pipeline` ->
`AssembledTask`/`AssembledPipeline`, via the `Assembler`); `LocalRunner`
executes an `AssembledPipeline` in-process, materializing every value
through a `BaseArtifactStore` (`LocalArtifactStore` by default) via a
`BaseMaterializer`, and recording run bookkeeping through a
`BaseMetadataStore` (`LocalMetadataStore` by default); `CodeBundler`
packages this project's own code so a remote task's runtime can import it.
"""

from .artifact_store import (
    BaseArtifactStore,
    LocalArtifactStore,
    LocalArtifactStoreConfig,
    S3ArtifactStore,
    S3ArtifactStoreConfig,
)
from .assembler import (
    Assembler,
    record_assembly,
    record_assembly_from_dicts,
    serialize_assembled_pipeline,
)
from .client import ArtifactClient, ArtifactClientConfig, Client, MetadataClient, MetadataClientConfig
from .code_bundler import CodeBundler
from .compute import BaseComputeBackend, LocalComputeBackend
from .exceptions import (
    AssemblyError,
    CyclicGraphError,
    IOBindingError,
    MaterializerResolutionError,
    MissingRequiredInputError,
)
from .io_binding import NO_DEFAULT, IOBinding, TaskInput
from .materializers import (
    BaseMaterializer,
    DefaultMaterializer,
    JsonMaterializer,
    PolarsParquetMaterializer,
    PydanticJsonMaterializer,
    resolve_materializer_for_type,
    resolve_materializer_for_value,
    resolve_materializer_from_artifact,
)
from .metadata_store import (
    BaseMetadataStore,
    LocalMetadataStore,
    LocalMetadataStoreConfig,
    PipelineAssemblyRecord,
    PipelineRegistrationRecord,
    PipelineRunRecord,
    PostgresMetadataStore,
    PostgresMetadataStoreConfig,
    RemoteMetadataStore,
    RemoteMetadataStoreConfig,
    RunStatus,
    TaskOutputRecord,
    TaskRunRecord,
    TriggerRecord,
)
from .pipeline import AssembledPipeline, Pipeline, PipelineInput, PipelineOutput, pipeline
from .runner import (
    ExecutionError,
    LocalRunner,
    MaterializerMismatchWarning,
    MissingPipelineInputError,
    UnknownPipelineInputError,
    run_locally,
)
from .task import (
    AssembledTask,
    ResourceRequirements,
    Task,
    TaskOutput,
    UvRequirements,
    resource,
    task,
    uv,
)

__all__ = [
    "BaseArtifactStore",
    "LocalArtifactStore",
    "LocalArtifactStoreConfig",
    "S3ArtifactStore",
    "S3ArtifactStoreConfig",
    "Assembler",
    "record_assembly",
    "record_assembly_from_dicts",
    "serialize_assembled_pipeline",
    "AssemblyError",
    "CyclicGraphError",
    "IOBindingError",
    "MaterializerResolutionError",
    "MissingRequiredInputError",
    "ArtifactClient",
    "ArtifactClientConfig",
    "Client",
    "MetadataClient",
    "MetadataClientConfig",
    "CodeBundler",
    "BaseComputeBackend",
    "LocalComputeBackend",
    "IOBinding",
    "TaskInput",
    "NO_DEFAULT",
    "BaseMaterializer",
    "DefaultMaterializer",
    "JsonMaterializer",
    "PolarsParquetMaterializer",
    "PydanticJsonMaterializer",
    "resolve_materializer_for_type",
    "resolve_materializer_for_value",
    "resolve_materializer_from_artifact",
    "BaseMetadataStore",
    "LocalMetadataStore",
    "LocalMetadataStoreConfig",
    "PipelineAssemblyRecord",
    "PipelineRegistrationRecord",
    "PipelineRunRecord",
    "PostgresMetadataStore",
    "PostgresMetadataStoreConfig",
    "RemoteMetadataStore",
    "RemoteMetadataStoreConfig",
    "RunStatus",
    "TaskOutputRecord",
    "TaskRunRecord",
    "TriggerRecord",
    "AssembledPipeline",
    "Pipeline",
    "PipelineInput",
    "PipelineOutput",
    "pipeline",
    "ExecutionError",
    "LocalRunner",
    "MaterializerMismatchWarning",
    "MissingPipelineInputError",
    "UnknownPipelineInputError",
    "run_locally",
    "AssembledTask",
    "ResourceRequirements",
    "Task",
    "TaskOutput",
    "UvRequirements",
    "resource",
    "task",
    "uv",
]
