import uuid

import pytest
from bettmensch_ai.pipelines.artifact_store import (
    LocalArtifactStore,
    LocalArtifactStoreConfig,
)
from bettmensch_ai.pipelines.code_bundler import CodeBundler


@pytest.fixture
def project_root(tmp_path):
    root = tmp_path / "project"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "helper.py").write_text("def helper():\n    return 1\n")

    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "junk.pyc").write_text("junk")

    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]")

    (root / ".env").write_text("SECRET=1")

    return root


@pytest.fixture
def store(tmp_path):
    return LocalArtifactStore(
        LocalArtifactStoreConfig(root_dir=str(tmp_path / "artifact_store"))
    )


def relative_paths(bundler):
    return {
        path.relative_to(bundler.project_root).as_posix()
        for path in bundler.iter_files()
    }


def test_default_ignore_patterns_exclude_pycache_and_git(project_root):
    bundler = CodeBundler(project_root)

    assert relative_paths(bundler) == {"pkg/__init__.py", "pkg/helper.py", ".env"}


def test_custom_ignore_file_excludes_additional_patterns(project_root):
    (project_root / ".bettmensch_aiignore").write_text("*.env\n")

    bundler = CodeBundler(project_root)

    assert relative_paths(bundler) == {
        "pkg/__init__.py",
        "pkg/helper.py",
        ".bettmensch_aiignore",
    }


def test_explicit_ignore_patterns_override_defaults_and_ignore_file(project_root):
    (project_root / ".bettmensch_aiignore").write_text("*.env\n")

    bundler = CodeBundler(project_root, ignore_patterns=["pkg"])

    # Explicit patterns replace the defaults entirely, so __pycache__/.git
    # are back, but the ignore file is never consulted at all.
    assert relative_paths(bundler) == {
        ".env",
        ".bettmensch_aiignore",
        "__pycache__/junk.pyc",
        ".git/config",
    }


def test_bundle_produces_a_zip_with_relative_arcnames(project_root, tmp_path):
    import zipfile

    bundler = CodeBundler(project_root)
    archive_path = str(tmp_path / "bundle.zip")
    bundler.bundle(archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    assert names == {"pkg/__init__.py", "pkg/helper.py", ".env"}


def test_bundle_and_upload_is_idempotent_for_the_same_run(project_root, store):
    bundler = CodeBundler(project_root)
    pipeline_run_id = uuid.uuid4()

    key_1 = bundler.bundle_and_upload(store, "my-pipeline", pipeline_run_id)
    key_2 = bundler.bundle_and_upload(store, "my-pipeline", pipeline_run_id)

    assert key_1 == key_2


def test_bundle_and_upload_differs_across_runs(project_root, store):
    bundler = CodeBundler(project_root)

    key_1 = bundler.bundle_and_upload(store, "my-pipeline", uuid.uuid4())
    key_2 = bundler.bundle_and_upload(store, "my-pipeline", uuid.uuid4())

    assert key_1 != key_2


def test_download_and_extract_round_trips_the_bundle(project_root, store, tmp_path):
    bundler = CodeBundler(project_root)
    key = bundler.bundle_and_upload(store, "my-pipeline", uuid.uuid4())

    destination = tmp_path / "extracted"
    CodeBundler.download_and_extract(store, key, destination)

    assert (destination / "pkg" / "__init__.py").exists()
    assert (destination / "pkg" / "helper.py").read_text() == (
        "def helper():\n    return 1\n"
    )
