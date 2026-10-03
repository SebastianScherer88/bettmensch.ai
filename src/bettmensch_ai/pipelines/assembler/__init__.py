"""Assembly: `Assembler`, the `AssemblyError` hierarchy it raises
(re-exported here from the shared top-level `exceptions.py` for
convenience), and `serialize_assembled_pipeline`/`record_assembly`/
`record_assembly_from_dicts` for persisting a pipeline's structure into a
`BaseMetadataStore`.
"""

from ..exceptions import (
    AssemblyError,
    CyclicGraphError,
    IOBindingError,
    MaterializerResolutionError,
    MissingRequiredInputError,
)
from .assembler import Assembler
from .recording import (
    record_assembly,
    record_assembly_from_dicts,
    serialize_assembled_pipeline,
)

__all__ = [
    "Assembler",
    "AssemblyError",
    "CyclicGraphError",
    "IOBindingError",
    "MaterializerResolutionError",
    "MissingRequiredInputError",
    "record_assembly",
    "record_assembly_from_dicts",
    "serialize_assembled_pipeline",
]
