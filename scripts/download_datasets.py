#!/usr/bin/env python3
"""Download Hugging Face datasets for offline training and MTEB evaluation."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from jina_eurobert.config import load_config
from jina_eurobert.datasets_registry import (
    build_manifest,
    collect_mteb_datasets,
    manifest_path_for,
    read_manifest,
    safe_dir_name,
    training_datasets_from_config,
    write_manifest,
)


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def _hf_cli_available() -> bool:
    return shutil.which("hf") is not None


def download_repo_hf_cli(repo_id: str, revision: str, local_dir: Path) -> None:
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "hf",
        "download",
        repo_id,
        "--repo-type",
        "dataset",
        "--revision",
        revision,
        "--local-dir",
        str(local_dir),
    ]
    subprocess.run(cmd, check=True)


def download_repo_snapshot(repo_id: str, revision: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
    )


def download_repo(repo_id: str, revision: str, local_dir: Path, force: bool) -> None:
    if local_dir.exists() and not force:
        return
    if force and local_dir.exists():
        shutil.rmtree(local_dir)
    if _hf_cli_available():
        download_repo_hf_cli(repo_id, revision, local_dir)
    else:
        download_repo_snapshot(repo_id, revision, local_dir)


def verify_local_load(repo_id: str, local_dir: Path, splits: list[str], config_name: str | None = None) -> None:
    from datasets import load_dataset

    if not splits:
        return
    split = splits[0]
    if config_name:
        dataset = load_dataset(str(local_dir), config_name, split=split)
    else:
        dataset = load_dataset(str(local_dir), split=split)
    if len(dataset) == 0:
        raise RuntimeError(f"{repo_id} loaded 0 rows from {local_dir}")


def collect_entries(
    config_path: str | None,
    benchmarks: list[str] | None,
    training_only: bool,
    mteb_only: bool,
) -> dict[str, dict]:
    training: dict[str, dict] = {}
    mteb_datasets: dict[str, str] = {}

    if not mteb_only:
        config = load_config(config_path)
        training = training_datasets_from_config(config)

    if not training_only:
        if benchmarks is None:
            config = load_config(config_path)
            benchmarks = config.get("eval", {}).get("benchmarks", ["MTEB(eng, v2)"])
        mteb_datasets = collect_mteb_datasets(benchmarks)

    return build_manifest(training=training or None, mteb_datasets=mteb_datasets or None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="data/hf_datasets")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--benchmark", action="append", default=None, help="MTEB benchmark to include")
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--mteb-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download even if local dir exists")
    parser.add_argument("--dry-run", action="store_true", help="List datasets without downloading")
    parser.add_argument("--manifest-only", action="store_true", help="Write manifest.json only")
    args = parser.parse_args()

    if args.training_only and args.mteb_only:
        parser.error("Use at most one of --training-only and --mteb-only")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_path_for(output_dir)

    entries = collect_entries(
        config_path=args.config,
        benchmarks=args.benchmark,
        training_only=args.training_only,
        mteb_only=args.mteb_only,
    )

    if args.dry_run or args.manifest_only:
        write_manifest(entries, manifest_file)
        print(f"{'Would download' if args.dry_run else 'Wrote manifest for'} {len(entries)} datasets:")
        for repo_id, entry in sorted(entries.items()):
            local_dir = output_dir / entry["local_dir"]
            status = "present" if local_dir.exists() else "missing"
            print(f"  {repo_id} @ {entry['revision']} -> {local_dir} [{status}]")
        if args.manifest_only:
            print(f"Wrote {manifest_file}")
        return

    manifest = read_manifest(manifest_file)
    manifest.update(entries)
    write_manifest(manifest, manifest_file)

    print(f"Downloading {len(entries)} datasets to {output_dir}")
    for repo_id, entry in sorted(entries.items()):
        revision = entry["revision"]
        local_dir = output_dir / entry["local_dir"]
        print(f"  {repo_id} @ {revision} -> {local_dir}")
        download_repo(repo_id, revision, local_dir, force=args.force)
        verify_local_load(repo_id, local_dir, entry.get("splits", []), entry.get("config"))
        manifest[repo_id] = entry
        write_manifest(manifest, manifest_file)

    print("\nSummary:")
    for repo_id, entry in sorted(manifest.items()):
        local_dir = output_dir / entry["local_dir"]
        print(f"  {repo_id}: {_format_size(_dir_size(local_dir))} at {local_dir}")
    print(f"\nManifest: {manifest_file}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
