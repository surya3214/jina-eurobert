from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mteb

MANIFEST_FILENAME = "manifest.json"

TRAINING_DATASETS: list[dict[str, Any]] = [
    {
        "repo_id": "sentence-transformers/gooaq",
        "revision": "main",
        "splits": ["train"],
    },
    {
        "repo_id": "sentence-transformers/natural-questions",
        "revision": "main",
        "splits": ["train"],
    },
    {
        "repo_id": "sentence-transformers/stsb",
        "revision": "main",
        "splits": ["train"],
    },
    {
        "repo_id": "sentence-transformers/msmarco-bm25",
        "revision": "main",
        "splits": ["train"],
        "config": "triplet",
    },
]

BENCHMARK_ALIASES = {
    "MTEB(multilingual, v2)": "MTEB(Multilingual, v2)",
    "MTEB(MULTILINGUAL, v2)": "MTEB(Multilingual, v2)",
}


def safe_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def resolve_benchmark(benchmark_name: str):
    canonical = BENCHMARK_ALIASES.get(benchmark_name, benchmark_name)
    return mteb.get_benchmark(canonical)


def filter_retrieval_sts_tasks(tasks: list) -> list:
    filtered = [task for task in tasks if task.metadata.type in {"Retrieval", "STS"}]
    return filtered or list(tasks)


def _dataset_metadata(task) -> dict[str, Any] | None:
    dataset = getattr(task.metadata, "dataset", None)
    if dataset is None:
        return None
    if isinstance(dataset, dict):
        return dataset
    return {
        "path": getattr(dataset, "path", None),
        "revision": getattr(dataset, "revision", None),
        "name": getattr(dataset, "name", None),
    }


def collect_mteb_datasets(benchmarks: list[str]) -> dict[str, str]:
    """Return repo_id -> revision for unique HF datasets used by MTEB tasks."""
    datasets: dict[str, str] = {}
    for benchmark_name in benchmarks:
        tasks = filter_retrieval_sts_tasks(list(resolve_benchmark(benchmark_name).tasks))
        for task in tasks:
            metadata = _dataset_metadata(task)
            if not metadata:
                continue
            repo_id = metadata.get("path")
            if not repo_id:
                continue
            revision = metadata.get("revision") or "main"
            datasets[repo_id] = revision
    return datasets


def training_datasets_from_config(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build training dataset entries from config lists, deduplicated by repo_id."""
    data_cfg = config.get("data", {})
    repo_ids: list[str] = []
    for key in ("pair_datasets", "triplet_datasets", "sts_datasets"):
        value = data_cfg.get(key, [])
        if isinstance(value, str):
            value = [value]
        repo_ids.extend(value)

    entries: dict[str, dict[str, Any]] = {}
    defaults = {item["repo_id"]: item for item in TRAINING_DATASETS}
    dataset_configs = data_cfg.get("dataset_configs", {})
    for repo_id in dict.fromkeys(repo_ids):
        default = defaults.get(
            repo_id,
            {"repo_id": repo_id, "revision": "main", "splits": ["train"]},
        )
        entry = {
            "revision": default["revision"],
            "local_dir": safe_dir_name(repo_id),
            "splits": list(default["splits"]),
        }
        config_name = dataset_configs.get(repo_id, default.get("config"))
        if config_name:
            entry["config"] = config_name
        entries[repo_id] = entry
    return entries


def build_manifest(
    *,
    training: dict[str, dict[str, Any]] | None = None,
    mteb_datasets: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    if training:
        manifest.update(training)
    if mteb_datasets:
        for repo_id, revision in mteb_datasets.items():
            manifest[repo_id] = {
                "revision": revision,
                "local_dir": safe_dir_name(repo_id),
                "splits": [],
            }
    return manifest


def read_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def write_manifest(manifest: dict[str, dict[str, Any]], path: str | Path) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def manifest_path_for(datasets_dir: str | Path) -> Path:
    return Path(datasets_dir) / MANIFEST_FILENAME
