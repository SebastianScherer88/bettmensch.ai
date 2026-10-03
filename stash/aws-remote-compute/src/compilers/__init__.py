"""Layer 2 compilers: turn a backend-agnostic `AssembledPipeline` into a
backend-specific `CompiledPipeline` - `Assembler`/`AssembledPipeline` stay
backend-agnostic; `compile`/`Compiler`/`CompiledPipeline` are reserved for
this, backend-specific step (see design-decisions.md's naming convention).
Sits alongside `assembler/` for the same reason `S3ArtifactStore` sits
alongside `LocalArtifactStore` inside `artifact_store/`: a Layer 2,
backend-flavoured piece of the same package, not a separate one.
"""
