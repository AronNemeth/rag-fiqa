"""Download and verify pinned Hugging Face and Ollama models."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from rag_fiqa.config import (
    GENERATION_MODEL_NAME,
    HF_MODELS,
    MODELS_MANIFEST_PATH,
    SUPPORTED_GENERATION_MODELS,
)
from rag_fiqa.utils import get_logger

logger = get_logger(__name__)

# Skip non-PyTorch framework weights to avoid redundant multi-hundred-MB downloads.
HF_IGNORE_PATTERNS: list[str] = [
    "*.msgpack",
    "*.h5",
    "*.ot",
    "*.tflite",
    "onnx/*",
    "openvino/*",
]


def download_hf_model(
    role: str,
    repo_id: str,
    revision: str,
    cache_dir: Path | None = None,
) -> dict[str, str]:
    """Download a pinned Hugging Face model snapshot into the cache.

    Args:
        role: Logical pipeline role (e.g. 'embedding' or 'reranker').
        repo_id: Hugging Face repository identifier.
        revision: Pinned Git commit SHA on Hugging Face Hub.
        cache_dir: Optional custom cache directory overriding HF_HOME/HF_HUB_CACHE.

    Returns:
        Metadata dictionary recording the role, repo_id, revision, and resolved snapshot path.

    Raises:
        RuntimeError: If the Hugging Face Hub download fails.
    """
    logger.info("Ensuring %s model %s (revision %s) is cached", role, repo_id, revision)
    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            ignore_patterns=HF_IGNORE_PATTERNS,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Failed to download Hugging Face model '{repo_id}' at revision '{revision}': {exc}"
        ) from exc

    logger.info("Verified %s model %s at %s", role, repo_id, snapshot_path)
    return {
        "role": role,
        "repo_id": repo_id,
        "revision": revision,
        "snapshot_path": str(snapshot_path),
    }


def pull_ollama_model(model_name: str = GENERATION_MODEL_NAME) -> dict[str, str]:
    """Pull an Ollama generation model via the Ollama CLI and record its digest.

    Args:
        model_name: Ollama model tag to pull. Must be one of SUPPORTED_GENERATION_MODELS.

    Returns:
        Metadata dictionary with the model name and recorded CLI details.

    Raises:
        ValueError: If model_name is not in SUPPORTED_GENERATION_MODELS.
        RuntimeError: If the ollama binary is missing or the pull command fails.
    """
    if model_name not in SUPPORTED_GENERATION_MODELS:
        supported = ", ".join(SUPPORTED_GENERATION_MODELS)
        raise ValueError(f"Unsupported generation model '{model_name}'. Supported models: {supported}")

    ollama_bin = shutil.which("ollama")
    if ollama_bin is None:
        raise RuntimeError(
            "Ollama CLI ('ollama') was not found on PATH. "
            "Install Ollama from https://ollama.com or pass --skip-ollama to download only Hugging Face models."
        )

    logger.info("Pulling Ollama generation model %s", model_name)
    try:
        subprocess.run([ollama_bin, "pull", model_name], check=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Ollama failed to pull model '{model_name}' (exit code {exc.returncode}). "
            "Verify that the Ollama daemon is running (`ollama serve`) and network access is available."
        ) from exc

    digest = _resolve_ollama_digest(ollama_bin, model_name)
    logger.info("Verified Ollama model %s (digest: %s)", model_name, digest or "unknown")
    return {
        "role": "generation",
        "provider": "ollama",
        "model": model_name,
        "digest": digest,
    }


def _resolve_ollama_digest(ollama_bin: str, model_name: str) -> str:
    """Extract the local digest for a pulled Ollama model from `ollama list`."""
    try:
        result = subprocess.run(
            [ollama_bin, "list"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError, OSError:
        return ""

    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and (parts[0] == model_name or parts[0] == f"{model_name}:latest"):
            return parts[1]
    return ""


def setup_models(
    roles: tuple[str, ...] = ("embedding", "reranker"),
    cache_dir: Path | None = None,
    generation_model: str = GENERATION_MODEL_NAME,
    skip_ollama: bool = False,
    manifest_path: Path = MODELS_MANIFEST_PATH,
) -> dict[str, Any]:
    """Download configured models and record their revisions in a manifest file.

    Args:
        roles: Hugging Face model roles from HF_MODELS to download.
        cache_dir: Optional directory for the Hugging Face Hub cache.
        generation_model: Ollama generation model tag to pull.
        skip_ollama: If True, skip pulling the Ollama generation model.
        manifest_path: Path where the JSON manifest will be written.

    Returns:
        The manifest dictionary written to manifest_path.
    """
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            existing = {}

    hf_records: dict[str, dict[str, str]] = dict(existing.get("huggingface_models", {}))
    for role in roles:
        if role not in HF_MODELS:
            raise ValueError(f"Unknown Hugging Face model role '{role}'. Expected one of: {tuple(HF_MODELS)}")
        spec = HF_MODELS[role]
        hf_records[role] = download_hf_model(
            role=role,
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            cache_dir=cache_dir,
        )

    generation_record: dict[str, str] | None = existing.get("generation_model")
    if not skip_ollama:
        generation_record = pull_ollama_model(generation_model)

    manifest: dict[str, Any] = {
        "huggingface_cache_dir": str(cache_dir)
        if cache_dir is not None
        else "default (HF_HOME / ~/.cache/huggingface)",
        "huggingface_models": hf_records,
        "generation_model": generation_record,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote model manifest to %s", manifest_path)
    return manifest
