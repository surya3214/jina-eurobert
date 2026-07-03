from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from datasets import Dataset, DownloadMode, load_dataset

from jina_eurobert.datasets_registry import MANIFEST_FILENAME, read_manifest

DATASETS_DIR_ENV = "JINA_EUROBERT_DATASETS_DIR"


def resolve_datasets_dir(
    datasets_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Path | None:
    if datasets_dir is not None:
        return Path(datasets_dir).expanduser().resolve()
    if config is not None:
        raw = config.get("data", {}).get("datasets_dir")
        if raw:
            return Path(raw).expanduser().resolve()
    env_dir = os.environ.get(DATASETS_DIR_ENV)
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return None


def local_files_only_setting(config: dict[str, Any] | None = None) -> bool:
    if config is None:
        return True
    return bool(config.get("data", {}).get("local_files_only", True))


def _manifest_for(datasets_dir: Path) -> dict[str, dict[str, Any]]:
    return read_manifest(datasets_dir / MANIFEST_FILENAME)


def _offline_error(repo_id: str, datasets_dir: Path, revision: str | None) -> RuntimeError:
    local_dir = datasets_dir / repo_id.replace("/", "__")
    revision_flag = f" --revision {revision}" if revision else ""
    command = (
        f"hf download {repo_id} --repo-type dataset{revision_flag} "
        f"--local-dir {local_dir}"
    )
    return RuntimeError(
        f"Dataset {repo_id!r} is not available locally under {datasets_dir}. "
        f"Download it first:\n  {command}\n"
        f"Or run: python scripts/download_datasets.py --output-dir {datasets_dir}"
    )


def _load_local_dataset(
    local_path: Path,
    split: str,
    *,
    config_name: str | None = None,
    trust_remote_code: bool = True,
) -> Dataset:
    kwargs: dict[str, Any] = {
        "split": split,
        "trust_remote_code": trust_remote_code,
        "download_mode": DownloadMode.REUSE_DATASET_IF_EXISTS,
    }
    path = str(local_path)
    if config_name:
        return load_dataset(path, config_name, **kwargs)
    return load_dataset(path, **kwargs)


def load_hf_split(
    repo_id: str,
    split: str,
    *,
    datasets_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
    config_name: str | None = None,
    local_files_only: bool | None = None,
    revision: str | None = None,
    trust_remote_code: bool = True,
) -> Dataset:
    """Load a dataset split from a local snapshot when available."""
    resolved_dir = resolve_datasets_dir(datasets_dir, config)
    strict_local = local_files_only if local_files_only is not None else local_files_only_setting(config)
    dataset_configs = (config or {}).get("data", {}).get("dataset_configs", {})
    resolved_config = config_name or dataset_configs.get(repo_id)

    if resolved_dir is not None:
        manifest = _manifest_for(resolved_dir)
        entry = manifest.get(repo_id)
        if entry is not None:
            local_path = resolved_dir / entry["local_dir"]
            if not local_path.exists():
                if strict_local:
                    raise _offline_error(repo_id, resolved_dir, entry.get("revision"))
            else:
                split_config = entry.get("config") or resolved_config
                return _load_local_dataset(
                    local_path,
                    split,
                    config_name=split_config,
                    trust_remote_code=trust_remote_code,
                )
        elif strict_local:
            raise _offline_error(repo_id, resolved_dir, revision)

    if strict_local:
        raise RuntimeError(
            f"Dataset {repo_id!r} requires a local datasets directory. "
            f"Set data.datasets_dir in config or {DATASETS_DIR_ENV}."
        )

    kwargs: dict[str, Any] = {
        "path": repo_id,
        "split": split,
        "trust_remote_code": trust_remote_code,
    }
    if resolved_dir is not None:
        entry = _manifest_for(resolved_dir).get(repo_id)
        split_config = (entry or {}).get("config") or resolved_config
    else:
        split_config = resolved_config
    if split_config:
        kwargs["name"] = split_config
    if revision:
        kwargs["revision"] = revision
    return load_dataset(**kwargs)


@contextmanager
def local_datasets_context(
    datasets_dir: str | Path,
    manifest: dict[str, dict[str, Any]] | None = None,
) -> Iterator[None]:
    """Patch datasets.load_dataset to read from local snapshots during MTEB eval."""
    import datasets as datasets_module

    resolved_dir = Path(datasets_dir).expanduser().resolve()
    manifest = manifest or _manifest_for(resolved_dir)
    original = datasets_module.load_dataset

    def patched(*args: Any, **kwargs: Any) -> Any:
        path = kwargs.get("path") or (args[0] if args else None)
        if path in manifest:
            local_path = resolved_dir / manifest[path]["local_dir"]
            entry = manifest[path]
            config_name = entry.get("config") or kwargs.get("name")
            kwargs = dict(kwargs)
            kwargs["path"] = str(local_path)
            kwargs["local_files_only"] = True
            kwargs.pop("revision", None)
            if config_name:
                kwargs["name"] = config_name
            if args:
                positional = [str(local_path)]
                if config_name:
                    positional.append(config_name)
                return original(*positional, *args[1:], **kwargs)
            return original(**kwargs)
        return original(*args, **kwargs)

    datasets_module.load_dataset = patched
    try:
        yield
    finally:
        datasets_module.load_dataset = original
