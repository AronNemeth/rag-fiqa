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
