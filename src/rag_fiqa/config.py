"""Python configs."""

from pathlib import Path

# project directories and paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_NAME = "fiqa"
PREPARED_DATA_DIR = DATA_DIR / "processed" / DATASET_NAME
DATASET_ARCHIVE = RAW_DATA_DIR / f"{DATASET_NAME}.zip"

# dataset download settings
DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
DATASET_MD5 = "17918ed23cd04fb15047f73e6c3bd9d9"
DOWNLOAD_TIMEOUT_SECONDS = 60
QRELS_SPLITS = ("train", "dev", "test")

# logging settings
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# chunking settings (only based on char length)
CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 120
CHUNKS_DIR = DATA_DIR / "chunks" / DATASET_NAME
CHUNK_CORPUS_PATH = PREPARED_DATA_DIR / "corpus_deduplicated.jsonl"

# model and artifact settings
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
HF_MODELS: dict[str, dict[str, str]] = {
    "embedding": {
        "repo_id": EMBEDDING_MODEL_ID,
        "revision": EMBEDDING_MODEL_REVISION,
    },
    "reranker": {
        "repo_id": RERANKER_MODEL_ID,
        "revision": RERANKER_MODEL_REVISION,
    },
}
GENERATION_MODEL_NAME = "qwen3:4b"
GENERATION_MODEL_ALTERNATIVES = ("qwen2.5:3b",)
SUPPORTED_GENERATION_MODELS = (GENERATION_MODEL_NAME, *GENERATION_MODEL_ALTERNATIVES)
MODELS_MANIFEST_PATH = DATA_DIR / "models" / "models_manifest.json"
