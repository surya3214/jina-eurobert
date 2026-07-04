from __future__ import annotations

import torch

from jina_eurobert.config import load_config
from jina_eurobert.data import build_training_mixture, load_teacher_embedding_index, teacher_index_key, text_hash
from jina_eurobert.losses import CombinedDistillationLoss, MRLEmbedDistillLoss


class _DummyModel:
  device = torch.device("cpu")

  def __call__(self, sentence_feature):
    batch_size = sentence_feature["sentence_embedding"].shape[0] if "sentence_embedding" in sentence_feature else 2
    dim = 768
    return {"sentence_embedding": torch.randn(batch_size, dim, requires_grad=True)}


def test_text_hash_is_stable():
    assert text_hash("hello") == text_hash("hello")
    assert text_hash("hello") != text_hash("world")


def test_teacher_index_key_uses_prompt():
    assert teacher_index_key("hello", "query") != teacher_index_key("hello", "document")


def test_precompute_smoke_writes_parquet(tmp_path):
    from scripts.precompute_teacher_mrl import write_smoke_test_embeddings

    dims = [32, 768]
    write_smoke_test_embeddings(tmp_path, dims)
    index = load_teacher_embedding_index(tmp_path)
    assert len(index) >= 2


def test_smoke_training_mixture_schema():
    config = load_config()
    mixture = build_training_mixture(config, smoke_test=True)
    assert set(mixture.keys()) == {"distill", "retrieval", "sts"}
    assert mixture["distill"].column_names == ["anchor", "positive", "teacher_anchor", "teacher_positive"]
    assert mixture["retrieval"].column_names == ["anchor", "positive", "negative"]
    assert mixture["sts"].column_names == ["anchor", "positive", "score"]


def test_matryoshka_dims_scalar_config():
    from jina_eurobert.config import matryoshka_dims

    assert matryoshka_dims({"matryoshka_dims": 768}) == [768]


def test_build_training_mixture_scalar_matryoshka_dims():
    config = load_config()
    config["matryoshka_dims"] = 768
    mixture = build_training_mixture(config, smoke_test=True)
    assert set(mixture.keys()) == {"distill", "retrieval", "sts"}


def test_resolve_mteb_benchmark_alias():
    from scripts.eval_mteb import resolve_benchmark

    benchmark = resolve_benchmark("MTEB(multilingual, v2)")
    assert benchmark.name == "MTEB(Multilingual, v2)"


def test_mrl_distill_loss_with_teacher_labels():
    model = _DummyModel()
    loss_fn = MRLEmbedDistillLoss(model, matryoshka_dims=768)  # type: ignore[arg-type]
    features = [
        {"sentence_embedding": torch.randn(2, 768, requires_grad=True)},
        {"sentence_embedding": torch.randn(2, 768, requires_grad=True)},
    ]
    labels = torch.randn(2, 2, 768)
    value = loss_fn(features, labels)
    assert torch.is_tensor(value)
    assert value.ndim == 0
    value.backward()
    assert value.item() >= 0


def test_model_device_with_data_parallel():
    import torch.nn as nn

    from jina_eurobert.device import model_device

    inner = nn.Linear(4, 4)
    wrapped = nn.DataParallel(inner)
    assert model_device(wrapped) == next(inner.parameters()).device


def test_student_forward_is_dataparallel_safe():
    import torch.nn as nn

    from jina_eurobert.models import build_student_model

    model = build_student_model(device="cpu", dtype=torch.float32)
    features = model.preprocess(["Query: test sentence"], prompt="Query: ")
    output = model(features)
    assert set(output.keys()) == {"sentence_embedding"}
    assert torch.is_tensor(output["sentence_embedding"])

    dp = nn.DataParallel(model)
    dp_output = dp(features)
    assert set(dp_output.keys()) == {"sentence_embedding"}
    assert torch.is_tensor(dp_output["sentence_embedding"])


def test_combined_loss_with_data_parallel_model():
    import torch.nn as nn

    from jina_eurobert.device import model_device
    from jina_eurobert.models import build_student_model

    model = build_student_model(device="cpu", dtype=torch.float32)
    wrapped = nn.DataParallel(model)
    loss_fn = CombinedDistillationLoss(
        model=wrapped,
        matryoshka_dims=[32, 768],
        loss_weights={"distill_mrl": 0.5, "infonce": 0.25, "cosent": 0.15, "gor": 0.1},
    )
    features = [
        model.preprocess(["Query: q1", "Query: q2"], prompt="Query: "),
        model.preprocess(["Document: d1", "Document: d2"], prompt="Document: "),
    ]
    labels = torch.randn(2, 2, 768)
    loss_fn.set_batch_type("distill")
    value = loss_fn(features, labels)
    assert torch.is_tensor(value)
    assert value.ndim == 0
    assert value.device == model_device(wrapped)


def test_combined_loss_routing():
    model = _DummyModel()
    loss_fn = CombinedDistillationLoss(
        model=model,  # type: ignore[arg-type]
        matryoshka_dims=[32, 768],
        loss_weights={"distill_mrl": 0.5, "infonce": 0.25, "cosent": 0.15, "gor": 0.1},
    )
    features = [
        {"sentence_embedding": torch.randn(2, 768, requires_grad=True)},
        {"sentence_embedding": torch.randn(2, 768, requires_grad=True)},
    ]

    loss_fn.set_batch_type("distill")
    distill_value = loss_fn(features, torch.randn(2, 2, 768))
    assert distill_value.item() >= 0

    loss_fn.set_batch_type("retrieval")
    retrieval_features = features + [
        {"sentence_embedding": torch.randn(2, 768, requires_grad=True)},
    ]
    retrieval_value = loss_fn(retrieval_features, None)
    assert retrieval_value.item() >= 0

    loss_fn.set_batch_type("sts")
    sts_value = loss_fn(features, torch.tensor([4.0, 3.5]))
    assert sts_value.item() >= 0


def test_manifest_lists_training_datasets():
    from jina_eurobert.datasets_registry import TRAINING_DATASETS

    repo_ids = {entry["repo_id"] for entry in TRAINING_DATASETS}
    assert repo_ids == {
        "sentence-transformers/gooaq",
        "sentence-transformers/natural-questions",
        "sentence-transformers/stsb",
        "sentence-transformers/msmarco-bm25",
    }


def test_load_hf_split_from_local_dir(tmp_path):
    from datasets import Dataset, DatasetDict

    from jina_eurobert.datasets_registry import write_manifest
    from jina_eurobert.hf_datasets import load_hf_split

    local_dir = tmp_path / "test__repo"
    DatasetDict({"train": Dataset.from_dict({"text": ["a", "b"]})}).save_to_disk(str(local_dir))
    write_manifest(
        {
            "test/repo": {
                "revision": "main",
                "local_dir": local_dir.name,
                "splits": ["train"],
            }
        },
        tmp_path / "manifest.json",
    )

    dataset = load_hf_split("test/repo", "train", datasets_dir=tmp_path, local_files_only=True)
    assert len(dataset) == 2
    assert dataset[0]["text"] == "a"


def test_training_manifest_includes_msmarco_config():
    from jina_eurobert.config import load_config
    from jina_eurobert.datasets_registry import training_datasets_from_config

    entries = training_datasets_from_config(load_config())
    msmarco = entries["sentence-transformers/msmarco-bm25"]
    assert msmarco["config"] == "triplet"
    assert msmarco["local_dir"] == "sentence-transformers__msmarco-bm25"


def test_load_msmarco_triplet_from_local_dir():
    from pathlib import Path

    import pytest

    from jina_eurobert.data import load_msmarco_triplet_dataset

    snapshot = Path("/tmp/msmarco_bm25/sentence-transformers__msmarco-bm25")
    if not (snapshot / "triplet").exists():
        pytest.skip("msmarco-bm25 snapshot not available")

    dataset = load_msmarco_triplet_dataset(
        max_samples=2,
        datasets_dir=Path("/tmp/msmarco_bm25"),
        config={"data": {"local_files_only": True}},
    )
    assert dataset.column_names == ["anchor", "positive", "negative"]
    assert len(dataset) == 2


def test_mteb_dataset_collection():
    from jina_eurobert.datasets_registry import collect_mteb_datasets

    datasets = collect_mteb_datasets(["MTEB(eng, v2)"])
    assert datasets
    assert all(repo_id.startswith("mteb/") for repo_id in datasets)


def test_collator_infers_dataset_name_without_lazy_column():
    from jina_eurobert.collators import DistillationDataCollator
    from jina_eurobert.device import infer_dataset_name

    assert infer_dataset_name({"anchor": "a", "positive": "b", "negative": "c"}) == "retrieval"
    assert infer_dataset_name({"anchor": "a", "positive": "b", "score": 3.5}) == "sts"
    assert infer_dataset_name({"anchor": "a", "positive": "b", "teacher_anchor": [1.0]}) == "distill"

    class DummyPreprocess:
        def __call__(self, inputs, prompt=None, task=None):
            batch_size = len(inputs)
            return {
                "input_ids": [[1, 2]] * batch_size,
                "attention_mask": [[1, 1]] * batch_size,
            }

    collator = DistillationDataCollator(preprocess_fn=DummyPreprocess(), prompts={})
    batch = collator(
        [
            {"anchor": "q", "positive": "d", "negative": "n"},
            {"anchor": "q2", "positive": "d2", "negative": "n2"},
        ]
    )
    assert batch["dataset_name"] == "retrieval"
    assert "negative_input_ids" in batch
    assert "label" not in batch


def test_summarize_training_mixture():
    from datasets import Dataset, DatasetDict

    from jina_eurobert.data import summarize_training_mixture

    mixture = DatasetDict(
        {
            "distill": Dataset.from_dict({"anchor": ["a"], "positive": ["b"]}),
            "retrieval": Dataset.from_dict({"anchor": ["a", "b"], "positive": ["c", "d"]}),
        }
    )
    assert summarize_training_mixture(mixture) == {"distill": 1, "retrieval": 2}


def test_prepare_sts_dataset_normalizes_scores():
    from datasets import Dataset

    from jina_eurobert.data import prepare_sts_dataset

    dataset = Dataset.from_dict({"anchor": ["a"], "positive": ["b"], "score": [4.0]})
    prepared = prepare_sts_dataset(dataset)
    assert prepared[0]["score"] == 0.8


def test_model_feature_inputs_keeps_prompt_length():
    from jina_eurobert.losses import _model_feature_inputs

    features = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "prompt_length": 3,
        "modality": "text",
    }
    model_inputs = _model_feature_inputs(features)
    assert model_inputs["prompt_length"] == 3
    assert set(model_inputs.keys()) == {"input_ids", "attention_mask", "prompt_length"}


def test_embeddings_from_features_is_immutable_across_calls():
    from jina_eurobert.losses import _embeddings_from_features
    from jina_eurobert.models import build_student_model

    model = build_student_model(device="cpu", dtype=torch.float32)
    model.eval()
    features = [model.preprocess(["Query: topic 0", "Query: topic 1"], prompt="Query: ")]
    before = set(features[0].keys())
    _embeddings_from_features(model, features)
    _embeddings_from_features(model, features)
    assert set(features[0].keys()) == before
    assert "token_embeddings" not in features[0]
    assert "sentence_embedding" not in features[0]


def test_combined_loss_finite_on_student_model():
    from jina_eurobert.config import load_config, matryoshka_dims
    from jina_eurobert.models import build_student_model

    model = build_student_model(device="cpu", dtype=torch.float32)
    model.eval()
    loss_fn = CombinedDistillationLoss(
        model=model,
        matryoshka_dims=matryoshka_dims(load_config()),
        loss_weights=load_config()["loss_weights"],
    )
    features = [
        model.preprocess(["Query: q1", "Query: q2"], prompt="Query: "),
        model.preprocess(["Document: d1", "Document: d2"], prompt="Document: "),
    ]
    loss_fn.set_batch_type("sts")
    value = loss_fn(features, torch.tensor([4.0, 3.0]))
    assert torch.isfinite(value).item()
    assert 0.0 <= float(value) <= 2.0
