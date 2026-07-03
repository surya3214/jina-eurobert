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


def test_mrl_distill_loss_with_teacher_labels():
    model = _DummyModel()
    loss_fn = MRLEmbedDistillLoss(model, matryoshka_dims=[32, 768])  # type: ignore[arg-type]
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
