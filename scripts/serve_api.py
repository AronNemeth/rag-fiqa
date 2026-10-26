"""Start the FastAPI backend server for the FiQA financial Q&A application."""

import argparse

import uvicorn

from rag_fiqa.config import API_HOST, API_PORT
from rag_fiqa.utils import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the API server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=API_HOST, help=f"Host interface to bind (default: {API_HOST}).")
    parser.add_argument("--port", type=int, default=API_PORT, help=f"Port number to listen on (default: {API_PORT}).")
    parser.add_argument("--reload", action="store_true", help="Enable automatic reload on code changes.")
    return parser.parse_args()


def main() -> None:
    """Configure logging and launch the Uvicorn ASGI server."""
    args = parse_args()
    setup_logging()
    uvicorn.run("rag_fiqa.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
