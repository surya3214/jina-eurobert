#!/usr/bin/env python3
"""Train distilled EuroBERT base model with multi-objective losses on 8xA100."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments

from jina_eurobert.collators import DistillationDataCollator
from jina_eurobert.config import load_config, matryoshka_dims
from jina_eurobert.data import build_training_mixture, load_teacher_embedding_index
from jina_eurobert.losses import CombinedDistillationLoss
from jina_eurobert.models import build_student_model
from jina_eurobert.trainer import DistillationTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config["training"]
    hardware_cfg = config["hardware"]
    student_cfg = config["student"]

    output_dir = Path(args.output_dir or config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_index = load_teacher_embedding_index(config["data"]["teacher_embeddings_dir"])
    if teacher_index and os.environ.get("RANK", "0") == "0":
        print(f"Loaded {len(teacher_index)} teacher embeddings from index.")

    train_dataset = build_training_mixture(
        config,
        teacher_index=teacher_index,
        max_samples_per_source=args.max_samples,
        smoke_test=args.smoke_test,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if hardware_cfg.get("bf16", True) and torch.cuda.is_available() else torch.float32

    model = build_student_model(
        model_name=student_cfg["model"],
        max_seq_length=student_cfg["max_seq_length"],
        trust_remote_code=student_cfg.get("trust_remote_code", True),
        device=device,
        dtype=dtype,
    )

    loss = CombinedDistillationLoss(
        model=model,
        matryoshka_dims=matryoshka_dims(config),
        loss_weights=config["loss_weights"],
    )

    max_steps = 10 if args.smoke_test else (args.max_steps or training_cfg["steps"])
    per_device_batch_size = 2 if args.smoke_test else hardware_cfg["per_device_batch_size"]

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=hardware_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=training_cfg["lr"],
        weight_decay=training_cfg.get("weight_decay", 0.01),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        bf16=hardware_cfg.get("bf16", True) and torch.cuda.is_available(),
        logging_steps=1 if args.smoke_test else training_cfg.get("logging_steps", 50),
        save_steps=training_cfg.get("save_steps", 2000),
        save_total_limit=3,
        dataloader_drop_last=not args.smoke_test,
        ddp_find_unused_parameters=False,
        report_to=[],
        seed=training_cfg.get("seed", 42),
    )

    query_prefix = config["prefixes"]["query"]
    document_prefix = config["prefixes"]["document"]
    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        data_collator=DistillationDataCollator(
            preprocess_fn=model.preprocess,
            prompts={
                "distill": {"anchor": query_prefix, "positive": document_prefix},
                "retrieval": {
                    "anchor": query_prefix,
                    "positive": document_prefix,
                    "negative": document_prefix,
                },
                "sts": {"anchor": document_prefix, "positive": document_prefix},
            },
        ),
    )
    trainer.train()
    final_path = output_dir / "final"
    model.save(str(final_path))
    if os.environ.get("RANK", "0") == "0":
        print(f"Saved distilled model to {final_path}")


if __name__ == "__main__":
    main()
