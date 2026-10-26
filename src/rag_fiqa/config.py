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

# chunking settings
CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 32
CHUNK_CORPUS_PATH = PREPARED_DATA_DIR / "corpus_deduplicated.jsonl"

# indexing and vector store settings
INDEX_DIR = DATA_DIR / "index" / DATASET_NAME
QDRANT_PATH = INDEX_DIR / "qdrant"
QDRANT_COLLECTION_NAME = "fiqa_chunks"
INDEX_MANIFEST_PATH = INDEX_DIR / "index_manifest.json"
EMBEDDING_BATCH_SIZE = 64
SUBSET_RANDOM_SEED = 42

# retrieval and reranking settings
RETRIEVAL_TOP_K_CHUNKS = 30
EVAL_TOP_K_DOCUMENTS = 20
RERANK_TOP_K_CHUNKS = 5

# evaluation settings
QUERIES_PATH = PREPARED_DATA_DIR / "queries.jsonl"
QRELS_DIR = PREPARED_DATA_DIR / "qrels"
EVAL_DIR = DATA_DIR / "eval" / DATASET_NAME
ABSTENTION_QUERIES_PATH = DATA_DIR / "eval" / "abstention_queries.json"
EVAL_SPLITS = ("dev", "test")
EVAL_NUM_QUERIES = 50
EVAL_GENERATION_SAMPLE = 10
EVAL_SEED = 42

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
GENERATION_MODEL_NAME = "qwen3:1.7b"
GENERATION_MODEL_ALTERNATIVES = ("qwen3:4b", "qwen3:1.7b")
SUPPORTED_GENERATION_MODELS = (GENERATION_MODEL_NAME, *GENERATION_MODEL_ALTERNATIVES)
MODELS_MANIFEST_PATH = DATA_DIR / "models" / "models_manifest.json"

# generation settings
OLLAMA_HOST = "http://localhost:11434"
GENERATION_TEMPERATURE = 0.0
GENERATION_MAX_TOKENS = 250
GENERATION_NUM_CTX = 2048
GENERATION_TIMEOUT_SECONDS = 60.0
MIN_RERANK_SCORE = -5.0

# api and serving settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "FiQA Financial Q&A RAG API"
API_VERSION = "0.1.0"
