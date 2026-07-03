#!/usr/bin/env python3
"""Evaluate distilled EuroBERT on MTEB retrieval and STS benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mteb
import torch
from sentence_transformers import SentenceTransformer

from jina_eurobert.config import load_config
from jina_eurobert.models import build_student_model


class PrefixRoutingModel:
    """Wrap SentenceTransformer to route prompts by MTEB task type."""

    def __init__(self, model: SentenceTransformer, max_seq_length: int = 512):
        self.model = model
        self.max_seq_length = max_seq_length

    def encode(
        self,
        sentences,
        task_name: str | None = None,
        prompt_type: str | None = None,
        batch_size: int = 32,
        **kwargs,
    ):
        prompt_name = prompt_type or kwargs.pop("prompt_name", None)
        if prompt_name is None and task_name is not None:
            task = mteb.get_task(task_name)
            if task.metadata.type == "STS":
                prompt_name = "document"
        kwargs.setdefault("batch_size", batch_size)
        kwargs.setdefault("normalize_embeddings", True)
        return self.model.encode(
            sentences,
            prompt_name=prompt_name,
            truncate_dim=kwargs.pop("truncate_dim", None),
            batch_size=batch_size,
            show_progress_bar=kwargs.pop("show_progress_bar", True),
            convert_to_numpy=kwargs.pop("convert_to_numpy", True),
            **kwargs,
        )


def load_tasks(
    config: dict,
    benchmark: str | None,
    task_names: list[str] | None,
) -> list:
    if task_names:
        return list(mteb.get_tasks(tasks=task_names))

    if benchmark:
        return list(mteb.get_benchmark(benchmark).tasks)

    tasks = []
    for benchmark_name in config.get("eval", {}).get("benchmarks", ["MTEB(eng, v2)"]):
        tasks.extend(mteb.get_benchmark(benchmark_name).tasks)
    return tasks


def filter_retrieval_sts_tasks(tasks: list) -> list:
    filtered = [task for task in tasks if task.metadata.type in {"Retrieval", "STS"}]
    return filtered or list(tasks)


def summarize_results(results: dict) -> dict[str, float]:
    summary: dict[str, float] = {}
    for task_name, task_result in results.items():
        payload = task_result.to_dict() if hasattr(task_result, "to_dict") else {}
        scores = payload.get("scores", {})
        test_scores = scores.get("test", scores)
        if isinstance(test_scores, list) and test_scores:
            test_scores = test_scores[0]
        if isinstance(test_scores, dict):
            for metric, value in test_scores.items():
                if isinstance(value, (int, float)):
                    summary[f"{task_name}/{metric}"] = float(value)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default=None, help="MTEB benchmark name")
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific MTEB task names")
    parser.add_argument("--output-dir", type=str, default="output/mteb_results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--truncate-dim", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("eval", {})
    student_cfg = config["student"]

    tasks = filter_retrieval_sts_tasks(load_tasks(config, args.benchmark, args.tasks))
    if args.smoke_test:
        tasks = tasks[:1]
        print(f"Smoke test: selected task {tasks[0].metadata.name} ({tasks[0].metadata.type})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_student_model(
        model_name=args.model_path,
        max_seq_length=eval_cfg.get("max_seq_length", student_cfg["max_seq_length"]),
        trust_remote_code=student_cfg.get("trust_remote_code", True),
        device=device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    routed_model = PrefixRoutingModel(model, max_seq_length=model.max_seq_length)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        routed_model,
        encode_kwargs={
            "batch_size": args.batch_size,
            "normalize_embeddings": True,
            "truncate_dim": args.truncate_dim,
        },
        output_folder=str(output_dir),
    )

    summary_path = output_dir / "summary.json"
    serializable = summarize_results(results)
    summary_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"Wrote MTEB results to {output_dir}")


if __name__ == "__main__":
    main()
