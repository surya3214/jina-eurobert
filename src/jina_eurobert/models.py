from __future__ import annotations

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
from transformers import modeling_rope_utils as rope_utils


def _register_eurobert_default_rope() -> None:
    """EuroBERT custom code uses rope_type='default', which newer transformers omit."""

    if "default" in rope_utils.ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(config, device=None, seq_len=None):  # noqa: ARG001
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rope_theta = getattr(config, "rope_theta", 250000.0)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
        )
        return inv_freq, 1.0

    rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def build_student_model(
    model_name: str = "EuroBERT/EuroBERT-210m",
    max_seq_length: int = 512,
    trust_remote_code: bool = True,
    device: str | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
) -> SentenceTransformer:
    """Build EuroBERT student with last-token pooling and Query/Document prefixes."""
    _register_eurobert_default_rope()

    model_kwargs: dict = {"trust_remote_code": trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    transformer = Transformer(
        model_name,
        max_seq_length=max_seq_length,
        model_kwargs=model_kwargs,
        config_kwargs={"trust_remote_code": trust_remote_code},
        processor_kwargs={"trust_remote_code": trust_remote_code},
    )
    pooling = Pooling(
        transformer.get_embedding_dimension(),
        pooling_mode="lasttoken",
    )
    model = SentenceTransformer(modules=[transformer, pooling], device=device)
    model.max_seq_length = max_seq_length
    model.prompts = {
        "query": "Query: ",
        "document": "Document: ",
    }
    model.default_prompt_name = None
    return model


def build_teacher_model(
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    max_seq_length: int = 512,
    device: str | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
) -> SentenceTransformer:
    """Load frozen Qwen3 teacher with left padding for last-token pooling."""
    model = SentenceTransformer(
        model_name,
        model_kwargs={
            "torch_dtype": dtype,
            "attn_implementation": "sdpa",
        },
        tokenizer_kwargs={"padding_side": "left"},
        device=device,
    )
    model.max_seq_length = max_seq_length
    return model


def truncate_embeddings(embeddings: torch.Tensor, dim: int) -> torch.Tensor:
    truncated = embeddings[..., :dim]
    return F.normalize(truncated, p=2, dim=-1)


def encode_with_prefix(
    model: SentenceTransformer,
    sentences: list[str],
    prompt_name: str | None = None,
    truncate_dim: int | None = None,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> torch.Tensor:
    kwargs: dict = {
        "batch_size": batch_size,
        "show_progress_bar": show_progress_bar,
        "convert_to_tensor": True,
        "normalize_embeddings": True,
    }
    if prompt_name is not None:
        kwargs["prompt_name"] = prompt_name
    if truncate_dim is not None:
        kwargs["truncate_dim"] = truncate_dim
    return model.encode(sentences, **kwargs)
