"""`CodeBundler`: packages and ships a project's own code to every task in
a run, mirroring Metaflow's "code package".
"""

import fnmatch
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, List, Optional, Union

from .artifact_store import BaseArtifactStore

if TYPE_CHECKING:
    from .client import ArtifactClient

_DEFAULT_IGNORE_PATTERNS = (
    ".git",
    "__pycache__",
    "*.pyc",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
    "build",
    "dist",
)

_IGNORE_FILE_NAME = ".bettmensch_aiignore"

# The code bundle is shared by every task in a run - it isn't scoped to any
# one of them - so it uses fixed, reserved names rather than a real task's/
# artifact's, giving it a stable, idempotent key. Both can never collide
# with a real task/artifact name, since those are always dasherized
# (underscores are stripped) when a `Task`/`AssembledTask` is assembled.
CODE_BUNDLE_TASK_NAME = "__code_bundle__"
CODE_BUNDLE_ARTIFACT_NAME = "__code_bundle__"


class CodeBundler:
    """Packages a project root directory into a single archive and uploads
    it to a `BaseArtifactStore`, once per pipeline run - mirroring
    Metaflow's "code package": every task's remote runtime unpacks the
    exact same code, so whatever local module a pickled (or otherwise
    locally-defined) object references is importable at the same path
    everywhere, regardless of which machine/container actually executes a
    given task.

    Deliberately does not try to determine which files a specific task
    needs, the way static import analysis would - it bundles everything
    under `project_root` not excluded by the ignore patterns, the same way
    for every task in a run. See design-decisions.md for why this trades
    precision for robustness and simplicity.
    """

    def __init__(
        self,
        project_root: Union[str, Path],
        ignore_patterns: Optional[Iterable[str]] = None,
    ):
        """Initializes the bundler.

        Args:
            project_root: The directory to bundle.
            ignore_patterns: `fnmatch` patterns to exclude, matched against
                each path component of every file under `project_root`. If
                omitted, uses sensible defaults (`.git`, `__pycache__`,
                `.venv`, etc.) plus anything listed in a
                `.bettmensch_aiignore` file at `project_root`, if one
                exists. Passing an explicit value here bypasses both the
                defaults and that file entirely.
        """

        self.project_root = Path(project_root).resolve()
        self.ignore_patterns = tuple(
            ignore_patterns
            if ignore_patterns is not None
            else self._default_ignore_patterns()
        )

    def _default_ignore_patterns(self) -> List[str]:
        """Builds the default ignore pattern list: the built-in defaults,
        plus any patterns listed in a `.bettmensch_aiignore` file at
        `project_root`.

        Returns:
            The combined list of ignore patterns.
        """

        patterns = list(_DEFAULT_IGNORE_PATTERNS)
        ignore_file = self.project_root / _IGNORE_FILE_NAME

        if ignore_file.exists():
            for line in ignore_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)

        return patterns

    def _is_ignored(self, path: Path) -> bool:
        """Checks whether any path component of `path` matches an ignore
        pattern.

        Args:
            path: The (absolute) path to check; must be under
                `project_root`.

        Returns:
            Whether `path` should be excluded from the bundle.
        """

        relative_parts = path.relative_to(self.project_root).parts

        return any(
            fnmatch.fnmatch(part, pattern)
            for part in relative_parts
            for pattern in self.ignore_patterns
        )

    def iter_files(self) -> Iterator[Path]:
        """Yields every file under `project_root` not excluded by the
        ignore patterns.

        Returns:
            An iterator of absolute paths.
        """

        for path in self.project_root.rglob("*"):
            if path.is_file() and not self._is_ignored(path):
                yield path

    def bundle(self, archive_path: str) -> None:
        """Zips every non-ignored file under `project_root` into a single
        archive at `archive_path`, with archive members named relative to
        `project_root`.

        Args:
            archive_path: The local path to write the zip archive to.
        """

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file_path in self.iter_files():
                archive.write(
                    file_path, arcname=file_path.relative_to(self.project_root)
                )

    def bundle_and_upload(
        self,
        artifact_store: Union[BaseArtifactStore, "ArtifactClient"],
        pipeline_name: str,
        pipeline_run_id: uuid.UUID,
    ) -> str:
        """Bundles `project_root` and uploads it to `artifact_store`. All
        tasks in the same pipeline run share this one key.

        Args:
            artifact_store: The store (or `ArtifactClient`) to upload the
                bundle to.
            pipeline_name: The name of the pipeline being run.
            pipeline_run_id: The id of this run of the pipeline.

        Returns:
            The key the code bundle was uploaded under.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = str(Path(tmp_dir) / "code_bundle.zip")
            self.bundle(archive_path)

            key = artifact_store.key(
                pipeline_name,
                pipeline_run_id,
                CODE_BUNDLE_TASK_NAME,
                CODE_BUNDLE_ARTIFACT_NAME,
            )
            artifact_store.upload(archive_path, key)

        return key

    @staticmethod
    def download_and_extract(
        artifact_store: Union[BaseArtifactStore, "ArtifactClient"],
        key: str,
        destination_dir: Union[str, Path],
    ) -> None:
        """Downloads the code bundle stored at `key` and extracts it into
        `destination_dir`. The counterpart to `bundle_and_upload`; does not
        itself do anything with `sys.path` or execute anything - that is a
        task entrypoint's job, not this utility's.

        Args:
            artifact_store: The store (or `ArtifactClient`) the bundle was
                uploaded to.
            key: The key the bundle was uploaded under (as returned by
                `bundle_and_upload`).
            destination_dir: The local directory to extract the bundle
                into. Created if it doesn't already exist.
        """

        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = str(Path(tmp_dir) / "code_bundle.zip")
            artifact_store.download(key, archive_path)

            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(destination_dir)
