from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
from sentence_transformers.util import batch_to_device
from transformers import AutoModel, modeling_rope_utils as rope_utils

from jina_eurobert.device import model_device


def patch_dataparallel_safe_forward(model: SentenceTransformer) -> SentenceTransformer:
    """Return only tensor outputs from forward so DataParallel gather_map succeeds.

    SentenceTransformer modules may attach non-tensor metadata (e.g. prompt_length,
    modality) to the feature dict. DataParallel cannot gather those values across GPUs.
    """
    original_forward = model.forward

    def forward(input: dict, **kwargs):  # type: ignore[no-untyped-def]
        output = original_forward(input, **kwargs)
        return {"sentence_embedding": output["sentence_embedding"]}

    model.forward = forward  # type: ignore[method-assign]
    return model


def _register_eurobert_default_rope() -> None:
    """EuroBERT custom code uses rope_type='default', which newer transformers omit."""

    if "default" in rope_utils.ROPE_INIT_FUNCTIONS:
        return

    # Transformers 5.x renamed the base RoPE initializer to "proportional".
    rope_utils.ROPE_INIT_FUNCTIONS["default"] = rope_utils.ROPE_INIT_FUNCTIONS["proportional"]


def _strip_eurobert_base_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("model.") for key in state):
        return {key.removeprefix("model."): value for key, value in state.items() if key.startswith("model.")}
    return state


def _load_safetensors_file(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))


def _load_local_checkpoint_state(model_dir: Path) -> dict[str, torch.Tensor]:
    single_path = model_dir / "model.safetensors"
    if single_path.is_file():
        return _load_safetensors_file(single_path)

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        state: dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(weight_map.values())):
            shard_path = model_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing EuroBERT shard {shard_path}")
            state.update(_load_safetensors_file(shard_path))
        return state

    pytorch_path = model_dir / "pytorch_model.bin"
    if pytorch_path.is_file():
        return torch.load(pytorch_path, map_location="cpu", weights_only=True)

    raise FileNotFoundError(
        f"No EuroBERT checkpoint found in {model_dir}. "
        "Expected model.safetensors, model.safetensors.index.json, or pytorch_model.bin."
    )


def _resolve_eurobert_checkpoint_path(model_name: str) -> Path:
    local_path = Path(model_name).expanduser()
    if local_path.is_dir():
        return local_path
    if local_path.is_file() and local_path.suffix == ".safetensors":
        return local_path
    return Path(hf_hub_download(model_name, "model.safetensors"))


def _eurobert_checkpoint_state_dict(model_name: str) -> dict[str, torch.Tensor]:
    checkpoint_path = _resolve_eurobert_checkpoint_path(model_name)
    if checkpoint_path.is_dir():
        state = _load_local_checkpoint_state(checkpoint_path)
    else:
        state = _load_safetensors_file(checkpoint_path)
    return _strip_eurobert_base_state_dict(state)


def _reload_eurobert_pretrained_weights(
    transformer: Transformer,
    model_name: str,
    *,
    dtype: torch.dtype | None,
    attn_implementation: str,
) -> None:
    """Rebuild EuroBERT with config-init + checkpoint weights.

    Transformers 5.5 can leave randomly initialized weights (and bad RoPE state)
    after ``from_pretrained`` for EuroBERT. Building from config and loading the
    safetensors checkpoint avoids that corruption.
    """
    model_kwargs: dict = {
        "trust_remote_code": True,
        "attn_implementation": attn_implementation,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    auto_model = AutoModel.from_config(transformer.config, **model_kwargs)
    state_dict = _eurobert_checkpoint_state_dict(model_name)
    missing, unexpected = auto_model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(
            "Failed to load EuroBERT pretrained weights; "
            f"missing {len(missing)} parameter(s), e.g. {missing[:3]}"
        )
    if unexpected:
        raise RuntimeError(
            "Failed to load EuroBERT pretrained weights; "
            f"unexpected {len(unexpected)} parameter(s), e.g. {unexpected[:3]}"
        )

    reference_key = "layers.0.self_attn.q_proj.weight"
    if reference_key not in state_dict:
        raise RuntimeError(f"EuroBERT checkpoint is missing expected weight {reference_key!r}.")
    loaded = auto_model.state_dict()[reference_key]
    expected = state_dict[reference_key].to(device=loaded.device, dtype=loaded.dtype)
    if not torch.allclose(loaded, expected):
        raise RuntimeError("EuroBERT pretrained weights were not applied to the student model.")

    transformer.model = auto_model


def _student_forward_smoke_test(model: SentenceTransformer) -> None:
    model.eval()
    device = model_device(model)
    with torch.no_grad():
        features = model.preprocess(["Query: weight check"], prompt="Query: ")
        features = batch_to_device(features, device)
        embeddings = model(features)["sentence_embedding"]
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("EuroBERT student model produced non-finite embeddings after weight load.")


def build_student_model(
    model_name: str = "EuroBERT/EuroBERT-210m",
    max_seq_length: int = 512,
    trust_remote_code: bool = True,
    device: str | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
) -> SentenceTransformer:
    """Build EuroBERT student with last-token pooling and Query/Document prefixes."""
    _register_eurobert_default_rope()

    use_cuda = device is not None and str(device).startswith("cuda")
    attn_implementation = "sdpa" if use_cuda else "eager"
    model_kwargs: dict = {
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_implementation,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    transformer = Transformer(
        model_name,
        max_seq_length=max_seq_length,
        model_kwargs=model_kwargs,
        config_kwargs={"trust_remote_code": trust_remote_code},
        processor_kwargs={"trust_remote_code": trust_remote_code},
    )
    _reload_eurobert_pretrained_weights(
        transformer,
        model_name,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    pooling = Pooling(
        transformer.get_embedding_dimension(),
        pooling_mode="lasttoken",
        include_prompt=False,
    )
    model = SentenceTransformer(modules=[transformer, pooling], device=device)
    if device is not None:
        model.to(device)
    _student_forward_smoke_test(model)
    model.max_seq_length = max_seq_length
    model.prompts = {
        "query": "Query: ",
        "document": "Document: ",
    }
    model.default_prompt_name = None
    return patch_dataparallel_safe_forward(model)


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
